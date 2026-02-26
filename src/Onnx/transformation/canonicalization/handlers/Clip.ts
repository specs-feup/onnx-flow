import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import type { DataType } from "../../../OnnxTypes.js";
import { toArrayLike, uniq, maybeRemoveOrphanConstant } from "../../../Utils.js";
import ConstantNode from "@specs-feup/onnx-flow/Onnx/ConstantNode";

// --- Handler ---
export default function clipHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
    if (op.type !== "Clip") return false;

    const ins = op.getInputs?.() ?? [];
    if (ins.length < 1) {
        throw new Error(`[ClipHandler] Node ${op.id} missing required input (data).`);
    }

    const Xn = ins[0];
    if (!Xn?.is?.(TensorNode) && !Xn?.is?.(ConstantNode)) {
        throw new Error(`[ClipHandler] Node ${op.id} input[0] invalid.`);
    }
    const X = Xn.is(TensorNode) ? Xn.as(TensorNode) : Xn.as(ConstantNode);
    const dtype = X.literalType as DataType;

    // Get output tensor Y
    const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
    if (outs.length !== 1) return false;
    const Y = outs[0];

    // --- Gather min/max ONLY from inputs (Inputs 1 and 2 are optional in ONNX) ---
    let minT: TensorNode.Class | ConstantNode.Class | undefined;
    let maxT: TensorNode.Class | ConstantNode.Class | undefined;

    if (ins[1]?.is?.(TensorNode)) {
        minT = ins[1].as(TensorNode);
    } else if (ins[1]?.is?.(ConstantNode)) {
        minT = ins[1].as(ConstantNode);
    }

    if (ins[2]?.is?.(TensorNode)) {
        maxT = ins[2].as(TensorNode);
    } else if (ins[2]?.is?.(ConstantNode)) {
        maxT = ins[2].as(ConstantNode);
    }

    // Build: cur = X; if (min) cur = Max(cur, min); if (max) cur = Min(cur, max)
    let cur: TensorNode.Class | ConstantNode.Class = X;

    const maxOp = g
        .addNode(uniq(g, `clip_max_${op.id}`))
        .init(new OperationNode.Builder("Max", [cur, minT!], {}))
        .as(OperationNode);
    const maxOut = g
        .addNode(uniq(g, `clip_max_out_${op.id}`))
        .init(
            new TensorNode.Builder(
                dtype,
                Array.isArray(X.shape) ? X.shape.slice() : [-1],
                "intermediate",
            ),
        )
        .as(TensorNode);
    g.addEdge(maxOp, maxOut).init(new OnnxEdge.Builder(dtype, maxOut.shape)).as(OnnxEdge);
    cur = maxOut;

    const minOp = g
        .addNode(uniq(g, `clip_min_${op.id}`))
        .init(new OperationNode.Builder("Min", [cur, maxT!], {}))
        .as(OperationNode);
    g.addEdge(minOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    cur = Y;

    // If neither min nor max existed (degenerate), just Identity to Y
    if (cur === X) {
        const id = g
            .addNode(uniq(g, `clip_id_${op.id}`))
            .init(new OperationNode.Builder("Identity", [X], {}))
            .as(OperationNode);
        g.addEdge(id, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    } else if (cur !== Y) {
        // Had only min or only max -> connect last op to Y
        const lastOp = toArrayLike<OperationNode.Class>(
            cur.getIncomers?.sources?.filterIs?.(OperationNode),
        )[0];
        g.addEdge(lastOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    }

    g.getNodeById(op.id)?.remove();

    // Clean up unused min/max constants or initializers
    maybeRemoveOrphanConstant(
        g,
        ins[1]?.is?.(TensorNode)
            ? ins[1].as(TensorNode)
            : ins[1]?.is?.(ConstantNode)
              ? ins[1].as(ConstantNode)
              : undefined,
    );
    maybeRemoveOrphanConstant(
        g,
        ins[2]?.is?.(TensorNode)
            ? ins[2].as(TensorNode)
            : ins[2]?.is?.(ConstantNode)
              ? ins[2].as(ConstantNode)
              : undefined,
    );

    return true;
}
