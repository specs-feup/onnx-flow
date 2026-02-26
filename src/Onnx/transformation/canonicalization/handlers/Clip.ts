import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import type { ConcreteValueNode, DataType } from "../../../OnnxTypes.js";
import {
    toArrayLike,
    uniq,
    maybeRemoveOrphanConstant,
    isConcreteValueNode,
    asConcreteValueNode,
    tryAsConcreteValueNode,
} from "../../../Utils.js";

// --- Handler ---
export default function clipHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
    if (op.type !== "Clip") return false;

    const ins = op.getInputs() ?? [];
    if (ins.length < 1) {
        throw new Error(`[ClipHandler] Node ${op.id} missing required input (data).`);
    }

    const Xn = ins[0];
    if (!isConcreteValueNode(Xn)) {
        throw new Error(`[ClipHandler] Node ${op.id} input[0] invalid.`);
    }
    const X = asConcreteValueNode(Xn);
    const dtype = X.literalType as DataType;

    // Get output tensor Y
    const outs = toArrayLike<TensorNode.Class>(op.getOutgoers.targets.filterIs(TensorNode));
    if (outs.length !== 1) return false;
    const Y = outs[0];

    // --- Gather min/max ONLY from inputs (Inputs 1 and 2 are optional in ONNX) ---
    const minT: ConcreteValueNode | undefined = tryAsConcreteValueNode(ins[1]);
    const maxT: ConcreteValueNode | undefined = tryAsConcreteValueNode(ins[2]);

    // Build: cur = X; if (min) cur = Max(cur, min); if (max) cur = Min(cur, max)
    let cur: ConcreteValueNode = X;

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
            cur.getIncomers.sources.filterIs(OperationNode),
        )[0];
        g.addEdge(lastOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    }

    g.getNodeById(op.id)?.remove();

    // Clean up unused min/max constants or initializers
    maybeRemoveOrphanConstant(g, tryAsConcreteValueNode(ins[1]));
    maybeRemoveOrphanConstant(g, tryAsConcreteValueNode(ins[2]));

    return true;
}
