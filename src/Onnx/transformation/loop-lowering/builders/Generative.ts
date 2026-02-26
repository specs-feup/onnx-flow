import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import type { TensorProto } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { uniq, int64Vec, bool, makeTensorConst } from "../../../Utils.js";
import type { LoopCtx, BuildResult, LoopBuilder } from "../BuildLoop.js";
import { unsqueezeIdx } from "../BuildLoop.js";

// Handlers needed here (Range + we allow trailing elementwise/transpose)
import handleRange from "../handlers/Range.js";
import handleElementWiseOperation from "../handlers/ElementWiseOperations.js";
import handleTranspose from "../handlers/Transpose.js";
import inferShapes from "@specs-feup/onnx-flow/Onnx/InferShapes";
import ConstantNode from "@specs-feup/onnx-flow/Onnx/ConstantNode";

export default class GenerativeBuilder implements LoopBuilder {
    canHandle(chain: OperationNode.Class[]): boolean {
        return chain.some((op) => op.type === "Range");
    }

    build(
        chain: OperationNode.Class[],
        outer: OnnxGraph.Class,
        _opts: { fuse: boolean; recurse: boolean; coalesce: boolean },
    ): BuildResult {
        const lastOp = chain.at(-1)!;
        let outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();

        // Find the Range op in the chain
        const rangeOp = chain.find((op) => op.type === "Range");
        if (!rangeOp) {
            throw new Error("GenerativeBuilder: expected a Range op in the chain");
        }

        // Use the Range's start input to define the element type
        const rangeInputs = rangeOp.getInputs()!;
        const startNodeRaw = rangeInputs[0];
        const startNode = startNodeRaw.is(TensorNode)
            ? startNodeRaw.as(TensorNode)
            : startNodeRaw.as(ConstantNode);
        const elemTy = startNode.literalType;

        // out shape is unknown-length 1D (Range defines its length at runtime)
        const outShape: (number | string)[] = [-1];

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const iter = body
            .addNode(uniq(body, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        body.addNode(uniq(body, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        // unknown length carry → declare [-1], *no* initializer
        const carry = body
            .addNode(uniq(body, "carry"))
            .init(new TensorNode.Builder(elemTy, [-1], "input"))
            .as(TensorNode);

        const axes = makeTensorConst(body, "axes", int64Vec([0]));
        const unsq = body
            .addNode(uniq(body, "unsq"))
            .init(new OperationNode.Builder("Unsqueeze", [iter, axes]))
            .as(OperationNode);
        const unsqOut = body
            .addNode(uniq(body, "unsq_out"))
            .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
            .as(TensorNode);
        body.addEdge(unsq, unsqOut)
            .init(new OnnxEdge.Builder(unsqOut.literalType, unsqOut.shape))
            .as(OnnxEdge);

        const ctx: LoopCtx = {
            opMap: new Map(),
            iter,
            unsqIdx: unsqOut,
            carry,
            axes,
            outShape,
            coalesce: false,
        };

        const handlers: Record<
            string,
            (op: OperationNode.Class, g: OnnxGraph.Class, ctx: LoopCtx) => TensorNode.Class
        > = {
            Range: handleRange,
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
            Transpose: handleTranspose,
        };

        for (const op of chain) {
            const h = handlers[op.type];
            if (!h) throw new Error(`GenerativeBuilder: unsupported op ${op.type}`);
            const out = h(op, body, ctx);
            ctx.opMap.set(op, [op, out]);
        }

        let lastOut = ctx.opMap.get(lastOp)![1];
        if (lastOut.shape.length === 0) {
            lastOut = unsqueezeIdx(body, lastOut, ctx.axes, "updateUnsq");
        }

        inferShapes(outer);
        inferShapes(body);

        // Compute trip_count, cond, v_initial for Range at OUTER graph level
        const [startT, limitT, deltaT] = rangeOp
            .getInputs()!
            .map((n) => (n.is(TensorNode) ? n.as(TensorNode) : n.as(ConstantNode)));

        const subN = outer
            .addNode(uniq(outer, `range_sub_${chain[0].id}`))
            .init(new OperationNode.Builder("Sub", [limitT, startT]))
            .as(OperationNode);
        const subOut = outer
            .addNode(uniq(outer, `range_sub_out_${chain[0].id}`))
            .init(new TensorNode.Builder(startT.literalType, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(subN, subOut)
            .init(new OnnxEdge.Builder(subOut.literalType, subOut.shape))
            .as(OnnxEdge);

        const subCastN = outer
            .addNode(uniq(outer, `range_subF_${chain[0].id}`))
            .init(new OperationNode.Builder("Cast", [subOut], { to: DataType.FLOAT }))
            .as(OperationNode);
        const subF = outer
            .addNode(uniq(outer, `range_subF_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(subCastN, subF)
            .init(new OnnxEdge.Builder(subF.literalType, subF.shape))
            .as(OnnxEdge);

        const deltaCastN = outer
            .addNode(uniq(outer, `range_deltaF_${chain[0].id}`))
            .init(new OperationNode.Builder("Cast", [deltaT], { to: DataType.FLOAT }))
            .as(OperationNode);
        const deltaF = outer
            .addNode(uniq(outer, `range_deltaF_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(deltaCastN, deltaF)
            .init(new OnnxEdge.Builder(deltaF.literalType, deltaF.shape))
            .as(OnnxEdge);

        const divN = outer
            .addNode(uniq(outer, `range_divF_${chain[0].id}`))
            .init(new OperationNode.Builder("Div", [subF, deltaF]))
            .as(OperationNode);
        const divF = outer
            .addNode(uniq(outer, `range_divF_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(divN, divF)
            .init(new OnnxEdge.Builder(divF.literalType, divF.shape))
            .as(OnnxEdge);

        const ceilN = outer
            .addNode(uniq(outer, `range_ceilF_${chain[0].id}`))
            .init(new OperationNode.Builder("Ceil", [divF]))
            .as(OperationNode);
        const ceilF = outer
            .addNode(uniq(outer, `range_ceilF_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(ceilN, ceilF)
            .init(new OnnxEdge.Builder(ceilF.literalType, ceilF.shape))
            .as(OnnxEdge);

        const zeroF = makeTensorConst(outer, `range_zeroF_${chain[0].id}`, {
            dataType: DataType.FLOAT,
            dims: [],
            floatData: [0],
        } as TensorProto);

        const maxN = outer
            .addNode(uniq(outer, `range_maxF_${chain[0].id}`))
            .init(new OperationNode.Builder("Max", [ceilF, zeroF]))
            .as(OperationNode);
        const maxF = outer
            .addNode(uniq(outer, `range_maxF_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(maxN, maxF)
            .init(new OnnxEdge.Builder(maxF.literalType, maxF.shape))
            .as(OnnxEdge);

        const tripCastN = outer
            .addNode(uniq(outer, `range_trip_${chain[0].id}`))
            .init(new OperationNode.Builder("Cast", [maxF], { to: DataType.INT64 }))
            .as(OperationNode);
        const tripScalar = outer
            .addNode(uniq(outer, `range_trip_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(tripCastN, tripScalar)
            .init(new OnnxEdge.Builder(tripScalar.literalType, tripScalar.shape))
            .as(OnnxEdge);

        const axes0 = makeTensorConst(outer, `axes0_${chain[0].id}`, int64Vec([0]));
        const tripUnsq = outer
            .addNode(uniq(outer, `range_trip_unsq_${chain[0].id}`))
            .init(new OperationNode.Builder("Unsqueeze", [tripScalar, axes0]))
            .as(OperationNode);
        const tripVec = outer
            .addNode(uniq(outer, `range_trip_vec_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(tripUnsq, tripVec)
            .init(new OnnxEdge.Builder(tripVec.literalType, tripVec.shape))
            .as(OnnxEdge);

        // 1) ConstantOfShape -> float zeros (no 'value' attr; ONNX default)
        const cos = outer
            .addNode(uniq(outer, `range_init_${chain[0].id}`))
            .init(new OperationNode.Builder("ConstantOfShape", [tripVec], {}))
            .as(OperationNode);

        const zerosF = outer
            .addNode(uniq(outer, `range_initF_out_${chain[0].id}`))
            .init(new TensorNode.Builder(DataType.FLOAT, [-1], "intermediate"))
            .as(TensorNode);
        outer
            .addEdge(cos, zerosF)
            .init(new OnnxEdge.Builder(zerosF.literalType, zerosF.shape))
            .as(OnnxEdge);

        // 2) Cast zeros to elemTy if needed
        let v_initial: TensorNode.Class;
        if (elemTy === DataType.FLOAT) {
            v_initial = zerosF;
        } else {
            const castInit = outer
                .addNode(uniq(outer, `range_init_cast_${chain[0].id}`))
                .init(new OperationNode.Builder("Cast", [zerosF], { to: elemTy }))
                .as(OperationNode);

            v_initial = outer
                .addNode(uniq(outer, `range_init_out_${chain[0].id}`))
                .init(new TensorNode.Builder(elemTy, [-1], "intermediate"))
                .as(TensorNode);

            outer
                .addEdge(castInit, v_initial)
                .init(new OnnxEdge.Builder(v_initial.literalType, v_initial.shape))
                .as(OnnxEdge);
        }

        const trip = tripScalar; // scalar
        const cond = makeTensorConst(outer, `cond_${chain[0].id}`, bool(true));

        // Ensure we always have an outer output tensor node for this generative chain
        if (!outTensor) {
            const shapeForOut =
                outShape && outShape.length && typeof outShape[0] === "number"
                    ? [outShape[0] as number]
                    : [];

            outTensor = outer
                .addNode(uniq(outer, `out_${lastOp.id}`))
                .init(new TensorNode.Builder(elemTy, shapeForOut, "intermediate"))
                .as(TensorNode);

            outer
                .addEdge(lastOp, outTensor)
                .init(new OnnxEdge.Builder(elemTy, shapeForOut))
                .as(OnnxEdge);
        }

        return {
            body,
            ctx,
            lastOut,
            indicesOut: unsqOut,
            elemTy,
            outShape,
            outTensor,
            trip,
            cond,
            v_initial,
        };
    }
}
