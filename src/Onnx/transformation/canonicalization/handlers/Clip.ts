import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, toArrayLike, uniq, maybeRemoveOrphanConstant } from "../../../Utils.js";

// --- Handler ---
export default function clipHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "Clip") return false;

  const ins = op.getInputs?.() ?? [];
  const Xn = ins[0];
  if (!Xn?.is?.(TensorNode)) return false;
  const X = Xn.as(TensorNode);
  const dtype = X.literalType as DataType;

  // Get output tensor Y
  const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Gather min/max from inputs (preferred) or attributes (fallback)
  const a = (op as any).getAttributes?.() ?? (op as any).attributes ?? {};

  let minT: TensorNode.Class | undefined;
  let maxT: TensorNode.Class | undefined;

  // input[1] = min?, input[2] = max?
  if (ins[1]?.is?.(TensorNode)) minT = ins[1].as(TensorNode);
  if (ins[2]?.is?.(TensorNode)) maxT = ins[2].as(TensorNode);

  // If missing, use attributes if present (older opsets)
  if (!minT && (a.min !== undefined)) {
    const minV = Number(a.min);
    const minConst = g.addNode(uniq(g, `clip_min_${op.id}`))
      .init(new TensorNode.Builder(dtype, [], "constant", makeTensorProto(dtype, [], [minV])))
      .as(TensorNode);
    minT = minConst;
  }
  if (!maxT && (a.max !== undefined)) {
    const maxV = Number(a.max);
    const maxConst = g.addNode(uniq(g, `clip_max_${op.id}`))
      .init(new TensorNode.Builder(dtype, [], "constant", makeTensorProto(dtype, [], [maxV])))
      .as(TensorNode);
    maxT = maxConst;
  }

  // Build: cur = X; if (min) cur = Max(cur, min); if (max) cur = Min(cur, max)
  let cur: TensorNode.Class = X;

  if (minT) {
    const maxOp = g.addNode(uniq(g, `clip_max_${op.id}`))
      .init(new OperationNode.Builder("Max", [cur, minT], {}))
      .as(OperationNode);
    const maxOut = g.addNode(uniq(g, `clip_max_out_${op.id}`))
      .init(new TensorNode.Builder(dtype, Array.isArray(X.shape) ? X.shape.slice() : [], "intermediate"))
      .as(TensorNode);
    g.addEdge(maxOp, maxOut).init(new OnnxEdge.Builder(dtype, maxOut.shape)).as(OnnxEdge);
    cur = maxOut;
  }

  if (maxT) {
    const minOp = g.addNode(uniq(g, `clip_min_${op.id}`))
      .init(new OperationNode.Builder("Min", [cur, maxT], {}))
      .as(OperationNode);
    // final edge directly to Y (avoid duplicate outputs)
    g.addEdge(minOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    cur = Y;
  }

  // If neither min nor max existed (degenerate), just Identity to Y
  if (cur === X) {
    const id = g.addNode(uniq(g, `clip_id_${op.id}`))
      .init(new OperationNode.Builder("Identity", [X], {}))
      .as(OperationNode);
    g.addEdge(id, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
  } else if (cur !== Y) {
    // Had only min or only max → connect last op to Y
    const lastOp = toArrayLike<OperationNode.Class>(cur.getIncomers?.sources?.filterIs?.(OperationNode))[0];
    if (lastOp) g.addEdge(lastOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
  }

  // Remove original Clip op
  g.getNodeById(op.id).remove();

  // Clean up unused min/max constants or initializers
  maybeRemoveOrphanConstant(g, ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined);
  maybeRemoveOrphanConstant(g, ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined);

  return true;
}
