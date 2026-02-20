import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import type { Shape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    uniq,
    int64Vec,
    zeroTensor,
    bool,
    toStaticShape,
    makeTensorConst,
    scalarInt64,
    asStaticDims,
    getAttr,
} from "../../../Utils.js";

import type { LoopCtx, BuildResult, LoopBuilder } from "../BuildLoop.js";
import { unsqueezeIdx, resolveFusedInput } from "../BuildLoop.js";
import handleTranspose from "../handlers/Transpose.js";
import ConstantNode from "@specs-feup/onnx-flow/Onnx/ConstantNode";
import RegionArgumentNode from "@specs-feup/onnx-flow/Onnx/RegionArgumentNode";

function toScalar(
    g: OnnxGraph.Class,
    t: TensorNode.Class | ConstantNode.Class | RegionArgumentNode.Class,
    tag: string,
): TensorNode.Class | ConstantNode.Class | RegionArgumentNode.Class {
    if (t.shape && t.shape.length === 0) return t;
    const shapeConst = makeTensorConst(g, uniq(g, `${tag}_shape`), int64Vec([]));
    const reshape = g
        .addNode(uniq(g, `${tag}_reshape`))
        .init(new OperationNode.Builder("Reshape", [t, shapeConst]))
        .as(OperationNode);
    const out = g
        .addNode(uniq(g, `${tag}_out`))
        .init(new TensorNode.Builder(t.literalType, [], "intermediate"))
        .as(TensorNode);
    g.addEdge(reshape, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
}

/**
 * Dedicated builder for Transpose chains:
 * - CHAIN: [Transpose]
 * - CHAIN: [Transpose, Add] (2D broadcast case)
 */
export default class TransposeBuilder implements LoopBuilder {
    canHandle(chain: OperationNode.Class[]): boolean {
        // Pure Transpose
        if (chain.length === 1 && chain[0].type === "Transpose") {
            return true;
        }

        // Fused [Transpose, Add] chain
        if (chain.length === 2 && chain[0].type === "Transpose" && chain[1].type === "Add") {
            return true;
        }

        return false;
    }

    build(
        chain: OperationNode.Class[],
        outer: OnnxGraph.Class,
        opts: { fuse: boolean; recurse: boolean; coalesce: boolean },
    ): BuildResult {
        const transposeOp = chain[0];
        const hasAdd = chain.length === 2;
        const addOp = hasAdd ? chain[1] : null;
        const finalOp = hasAdd ? addOp! : transposeOp;

        // ---- Element type / final output tensor ---------------------------
        let outTensor = finalOp.getOutgoers.targets.filterIs(TensorNode).first();

        let elemTy: DataType = DataType.FLOAT;
        if (outTensor) {
            elemTy =
                outTensor.literalType === DataType.UNDEFINED
                    ? (finalOp.getOutgoers.first()?.literalType ?? DataType.FLOAT)
                    : outTensor.literalType;
        } else {
            const xInNodeRaw = transposeOp
                .getInputs()
                ?.find((n) => n.is(TensorNode) || n.is(ConstantNode) || n.is(RegionArgumentNode));
            const xInNode = xInNodeRaw.is(TensorNode)
                ? xInNodeRaw.as(TensorNode)
                : xInNodeRaw.is(ConstantNode)
                  ? xInNodeRaw.as(ConstantNode)
                  : xInNodeRaw.as(RegionArgumentNode);
            if (xInNode) {
                elemTy = xInNode.literalType;
            }
        }

        // ---- Compute transpose output shape from input+perm ---------------
        const xInRaw = transposeOp
            .getInputs()
            ?.find((n) => n.is(TensorNode) || n.is(ConstantNode) || n.is(RegionArgumentNode));
        const xIn = xInRaw.is(TensorNode)
            ? xInRaw.as(TensorNode)
            : xInRaw.is(ConstantNode)
              ? xInRaw.as(ConstantNode)
              : xInRaw.as(RegionArgumentNode);

        const inShape = xIn ? toStaticShape(xIn.shape as Shape) : [];
        let outShape: (number | string)[] = [];

        if (inShape && inShape.length > 0) {
            const rank = inShape.length;

            // Use the *same* perm-reading logic as the handler
            let perm = getAttr(transposeOp, "perm");
            if (!Array.isArray(perm) || perm.length !== rank) {
                // ONNX default for Transpose: reverse axes
                perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
            }

            // Safely cast perm to an array of numbers
            const permArray = perm as number[];

            outShape = permArray.map((p) => inShape[p] ?? 1);
        } else {
            // Very conservative fallback
            outShape = [];
        }

        // Ensure we always have an outer output tensor node for this chain
        if (!outTensor) {
            const fallbackShape =
                outShape && outShape.length ? outShape : ([] as (number | string)[]);
            outTensor = outer
                .addNode(uniq(outer, `out_${finalOp.id}`))
                .init(new TensorNode.Builder(elemTy, fallbackShape, "intermediate"))
                .as(TensorNode);

            outer
                .addEdge(finalOp, outTensor)
                .init(new OnnxEdge.Builder(elemTy, fallbackShape))
                .as(OnnxEdge);
        }

        // ---- Trip count and carry length ----------------------------------
        const safeOut = asStaticDims(outShape as (number | string)[]);
        const totalIters =
            safeOut.length <= 1 ? (safeOut[0] ?? 1) : safeOut.reduce((a, b) => a * b, 1);
        const carryLen = totalIters;

        // ---- Body graph skeleton ------------------------------------------
        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const iter = body
            .addNode(uniq(body, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);

        const _condIn = body
            .addNode(uniq(body, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        const carry = body
            .addNode(uniq(body, "carry"))
            .init(new TensorNode.Builder(elemTy, [carryLen], "input"))
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
            coalesce: opts.coalesce,
        };

        // ---- 1) Transpose scalar for this iteration -----------------------
        // handleTranspose now strictly returns a scalar []
        const tpScalar = handleTranspose(transposeOp, body, ctx);
        ctx.opMap.set(transposeOp, [transposeOp, tpScalar]);

        let lastOut: TensorNode.Class = tpScalar;

        // ---- 2) Optional Add after Transpose (CHAIN: [Transpose, Add]) ----
        if (hasAdd && addOp) {
            const addInputs = addOp.getInputs?.() ?? [];
            let otherInput = null;

            // Find the Add input that is *not* produced by the Transpose op
            for (const inp of addInputs) {
                if (!inp.is(TensorNode) && !inp.is(ConstantNode)) continue;
                const t = inp.is(TensorNode) ? inp.as(TensorNode) : inp.as(ConstantNode);

                if (t.getIncomers.length > 0) {
                    const prod = t.getIncomers[0].source;
                    if (prod.is && prod.is(OperationNode) && prod.id === transposeOp.id) {
                        // This path comes from the Transpose op; skip it
                        continue;
                    }
                }

                otherInput = inp;
                break;
            }

            if (!otherInput) {
                throw new Error("TransposeBuilder: could not find non-transpose input for Add");
            }

            // Broadcast-aware scalar for the other input at this iter
            let otherScalar = resolveFusedInput(
                body,
                otherInput,
                ctx,
                addOp,
                /*flatten*/ false,
                /*returnGather*/ true,
            );

            // Force to scalar [] so we have strictly Rank 0 + Rank 0
            otherScalar = toScalar(body, otherScalar, "add_other_scalar");

            // Now build Add: scalar + scalar → scalar
            const addBodyNode = body
                .addNode(uniq(body, "add_loop"))
                .init(new OperationNode.Builder("Add", [tpScalar, otherScalar]))
                .as(OperationNode);

            const addOut = body
                .addNode(uniq(body, "add_loop_out"))
                .init(new TensorNode.Builder(elemTy, [], "intermediate"))
                .as(TensorNode);

            body.addEdge(addBodyNode, addOut).init(new OnnxEdge.Builder(elemTy, [])).as(OnnxEdge);

            ctx.opMap.set(addOp, [addOp, addOut]);
            lastOut = addOut;
        }

        // lastOut is now guaranteed to be scalar [].
        // We must Unsqueeze it to [1] to match the rank of ScatterElements indices (unsqOut [1]).
        lastOut = unsqueezeIdx(body, lastOut, ctx.axes, hasAdd ? "updateUnsq_add" : "updateUnsq");

        // ---- Loop inputs for outer graph ----------------------------------
        const trip = makeTensorConst(outer, `trip_count_${chain[0].id}`, scalarInt64(totalIters));
        const cond = makeTensorConst(outer, `cond_${chain[0].id}`, bool(true));
        const v_initial = makeTensorConst(outer, "init_carry", zeroTensor(elemTy, [carryLen]));

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
