import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import type OperationNode from "../../../OperationNode.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    uniq,
    int64Vec,
    zeroTensor,
    bool,
    makeTensorConst,
    scalarInt64,
    asConcreteValueNode,
} from "../../../Utils.js";
import type { LoopCtx, BuildResult, LoopBuilder } from "../BuildLoop.js";
import { unsqueezeIdx, broadcastShapes, getMatDims } from "../BuildLoop.js";

// Handlers needed here
import handleElementWiseOperation from "../handlers/ElementWiseOperations.js";
import handleMatMul from "../handlers/MatMul.js";
import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import inferShapes from "@specs-feup/onnx-flow/Onnx/InferShapes";

export default class MatMulBuilder implements LoopBuilder {
    canHandle(chain: OperationNode.Class[]): boolean {
        return chain.some((op) => op.type === "MatMul");
    }

    build(
        chain: OperationNode.Class[],
        outer: OnnxGraph.Class,
        opts: { fuse: boolean; recurse: boolean; coalesce: boolean },
    ): BuildResult {
        const matmulIndex = chain.findIndex((op) => op.type === "MatMul");
        const lastOp = chain.at(-1)!;
        let outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();

        const fallbackElemTy = lastOp.getOutgoers.first()?.literalType ?? DataType.FLOAT;

        let elemTy =
            outTensor && outTensor.literalType !== DataType.UNDEFINED
                ? outTensor.literalType
                : fallbackElemTy;

        // If the operation works on 8-bit integers, we MUST accumulate in INT32
        // to avoid immediate overflow. We also update the output tensor type
        // so the graph remains consistent.
        if (elemTy === DataType.INT8 || elemTy === DataType.UINT8) {
            elemTy = DataType.INT32;
            if (outTensor) {
                outTensor.setLiteralType(elemTy);
            }
        }

        inferShapes(outer);

        const mm = chain[matmulIndex];
        const lhsRaw = mm.getInputs()![0];
        const lhs = asConcreteValueNode(lhsRaw);
        const rhsRaw = mm.getInputs()![1];
        const rhs = asConcreteValueNode(rhsRaw);

        // Use shared helper to normalise vector/matrix shapes
        const { K, N, A2, B2, ...dims } = getMatDims(lhs.shape, rhs.shape);
        let { M } = dims;

        const lhsShape = lhs.shape;
        if (lhsShape.length >= 2) {
            const mCandidate = Number(lhsShape[lhsShape.length - 2]);
            if (Number.isFinite(mCandidate) && mCandidate > 0) {
                M = mCandidate;
            }
        }

        // Leading batch dims from both inputs (may be empty)
        const aBatch = A2.length > 2 ? (A2.slice(0, -2) as number[]) : [];
        const bBatch = B2.length > 2 ? (B2.slice(0, -2) as number[]) : [];

        // ONNX / NumPy broadcast of batch dims
        const batchDimsStatic = broadcastShapes([aBatch, bBatch]);

        const batchDims = batchDimsStatic as KnownShape;

        // Batch product (treat non-positive/dynamic as 1 in the loop trip count)
        const batchProd = (batchDimsStatic.length ? batchDimsStatic : [1])
            .map((d) => {
                const n = Number(d);
                if (!Number.isFinite(n) || n <= 0) return 1;
                return n;
            })
            .reduce((p, d) => p * d, 1);

        // Loop config
        const totalIters = batchProd * M * K * N;
        const carryLen = batchProd * M * N;
        const finalOutShape = [...batchDims, M, N];

        // Ensure we always have an outer output tensor node for this chain
        if (!outTensor) {
            outTensor = outer
                .addNode(uniq(outer, `out_${lastOp.id}`))
                .init(new TensorNode.Builder(elemTy, finalOutShape, "intermediate"))
                .as(TensorNode);

            outer
                .addEdge(lastOp, outTensor)
                .init(new OnnxEdge.Builder(elemTy, finalOutShape))
                .as(OnnxEdge);
        }

        const matmulDims = { M, K, N, batchProd, batchDims };

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const iter = body
            .addNode(uniq(body, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        body.addNode(uniq(body, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        // carry buffer flat [M*N]
        const carry = body
            .addNode(uniq(body, "carry"))
            .init(new TensorNode.Builder(elemTy, [carryLen], "input"))
            .as(TensorNode);

        const axes = makeTensorConst(body, "axes", int64Vec([0]));
        // Flat index (i,j,k) decoding + cached unsqueezed indices provided by handler
        const ctx: LoopCtx = {
            opMap: new Map(),
            iter,
            unsqIdx: null, // provided per-path; not used for coalesced MatMul
            carry,
            axes,
            outShape: finalOutShape,
            coalesce: opts.coalesce,
            matmulDims,
            iU: null,
            jU: null,
            kU: null,
            flatU: null,
            kIdx: null,
            kM1: null,
            gateByK:
                opts.coalesce &&
                chain
                    .slice(matmulIndex + 1)
                    .some(
                        (op) =>
                            op.type === "Add" ||
                            op.type === "Sub" ||
                            op.type === "Mul" ||
                            op.type === "Div",
                    ),
            running: null,
        };

        ctx.outShape = finalOutShape;

        const handlers: Record<
            string,
            (op: OperationNode.Class, g: OnnxGraph.Class, ctx: LoopCtx) => TensorNode.Class
        > = {
            MatMul: handleMatMul,
            Add: handleElementWiseOperation,
            Sub: handleElementWiseOperation,
            Mul: handleElementWiseOperation,
            Div: handleElementWiseOperation,
            Relu: handleElementWiseOperation,
            Sigmoid: handleElementWiseOperation,
            Tanh: handleElementWiseOperation,
            Exp: handleElementWiseOperation,
            Sum: handleElementWiseOperation,
            Min: handleElementWiseOperation,
            Max: handleElementWiseOperation,
        };

        let indicesOut: ConcreteValueNode | null = null;

        for (const op of chain) {
            const h = handlers[op.type];
            const out = h(op, body, ctx);
            ctx.opMap.set(op, [op, out]);
            if (op.type === "MatMul") {
                indicesOut = ctx.flatU ?? ctx.unsqIdx!;
            }
        }

        let lastOut = ctx.opMap.get(lastOp)![1];
        if (lastOut.shape.length === 0) {
            lastOut = unsqueezeIdx(body, lastOut, ctx.axes, "updateUnsq");
        }

        inferShapes(outer);
        inferShapes(body);

        // Loop inputs
        const trip = makeTensorConst(outer, `trip_count_${chain[0].id}`, scalarInt64(totalIters));
        const cond = makeTensorConst(outer, `cond_${chain[0].id}`, bool(true));
        const v_initial = makeTensorConst(outer, "init_carry", zeroTensor(elemTy, [carryLen]));

        return {
            body,
            ctx,
            lastOut,
            indicesOut: indicesOut!,
            elemTy,
            outShape: finalOutShape,
            outTensor,
            trip,
            cond,
            v_initial,
        };
    }
}
