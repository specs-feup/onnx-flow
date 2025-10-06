import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../Utilities.js";

// --- small utils (local) ---
function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}
function toArrayLike<T = any>(nc: any): T[] {
  return nc?.toArray?.() ?? nc ?? [];
}
function getOnnxName(tn?: TensorNode.Class): string | undefined {
  if (!tn) return undefined;
  const a: any = tn as any;
  return a.extraAttrs?.onnxName ?? a.name ?? a.id ?? a.getName?.();
}
function removeInitializerByName(g: OnnxGraph.Class, name?: string) {
  if (!name) return;
  const anyG: any = g as any;
  const model = anyG?.rawModel ?? anyG?.model;
  const graph = model?.graph ?? anyG?.graph;
  if (!graph) return;
  for (const f of ["initializer","sparse_initializer","input","value_info"]) {
    if (Array.isArray(graph[f])) graph[f] = graph[f].filter((x: any) => x?.name !== name);
  }
}
function maybeRemoveOrphanConstant(g: OnnxGraph.Class, tn?: TensorNode.Class) {
  if (!tn) return;
  const isConstLike =
    (tn as any).type === "constant" ||
    (tn as any).constantValue != null ||
    (tn as any).originalInitializer != null ||
    (tn as any).initializer != null;
  if (!isConstLike) return;
  const consumers = toArrayLike<OperationNode.Class>(tn.getOutgoers?.targets?.filterIs?.(OperationNode));
  if (consumers.length > 0) return;

  // remove upstream Constant op if it's now orphan
  const srcOps = toArrayLike<OperationNode.Class>(tn.getIncomers?.sources?.filterIs?.(OperationNode));
  for (const src of srcOps) {
    if (src.type !== "Constant") continue;
    const outs = toArrayLike<TensorNode.Class>(src.getOutgoers?.targets?.filterIs?.(TensorNode));
    const stillUsed = outs.some(t => toArrayLike(src.getOutgoers?.targets?.filterIs?.(OperationNode)).length > 0);
    if (!stillUsed) src.remove();
  }

  const onnxName = getOnnxName(tn);
  tn.remove();
  removeInitializerByName(g, onnxName);
}

// --- main handler ---
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
