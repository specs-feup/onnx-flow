import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { uniq, addEdge, toArrayLike, makeI64ShapeConst } from "../../../Utils.js";

/**
 * Softmax(X, axis)  ≡  exp(X - reduce_max(X, axis)) / reduce_sum(exp(...), axis)
 * We build:
 *   m   = ReduceMax(X, axes=[axis], keepdims=1)
 *   sh  = Sub(X, m)
 *   ex  = Exp(sh)
 *   den = ReduceSum(ex, axes=[axis], keepdims=1)
 *   Y   = Div(ex, den)
 *
 * Notes:
 * - Reductions use opset13-style axes as a 1-D INT64 tensor input.
 * - We set explicit shapes for intermediates to keep the reduce builder happy.
 */
export default function softmaxHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "Softmax") return false;

  // ---- Inputs / outputs
  const ins = op.getInputs?.() ?? [];
  if (ins.length < 1 || !ins[0]?.is?.(TensorNode)) return false;

  const X = ins[0].as(TensorNode);
  const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // ---- Rank and axis
  const inShape = Array.isArray(X.shape) ? [...X.shape] : [];
  const rank = inShape.length;

  // axis attribute (default -1 per opset >= 13)
  const attrs = (op as any).getAttributes?.() ?? (op as any).attributes ?? {};
  let axis = Number(attrs.axis ?? -1);
  if (rank > 0 && axis < 0) axis = (axis + rank) % rank;

  // Helper: shapes for intermediates
  // M and DEN have keepdims=1 at `axis`; SH and EX match X
  const xShape = inShape.length ? [...inShape] : [];     // may contain -1 / undefined; that's fine
  const redShape =
    rank === 0 ? [] :
    (() => {
      const s = [...inShape];
      if (s.length) s[Math.max(0, axis)] = 1; // keepdims=1 at axis
      return s;
    })();

  // ---- m = ReduceMax(X, axes=[axis], keepdims=1)
  const rmax = g.addNode(uniq(g, `sm_rmax_${op.id}`))
    .init(new OperationNode.Builder("ReduceMax", [X], { keepdims: 1, axes: [axis] }))
    .as(OperationNode);
  const M = g.addNode(uniq(g, `sm_maxT_${op.id}`))
    .init(new TensorNode.Builder(X.literalType as DataType, redShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, rmax, M, X.literalType as DataType, M.shape);

  // ---- sh = Sub(X, m)
  const sub = g.addNode(uniq(g, `sm_sub_${op.id}`))
    .init(new OperationNode.Builder("Sub", [X, M], {}))
    .as(OperationNode);
  const SH = g.addNode(uniq(g, `sm_shT_${op.id}`))
    .init(new TensorNode.Builder(X.literalType as DataType, xShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, sub, SH, X.literalType as DataType, SH.shape);

  // ---- ex = Exp(sh)
  const exp = g.addNode(uniq(g, `sm_exp_${op.id}`))
    .init(new OperationNode.Builder("Exp", [SH], {}))
    .as(OperationNode);
  const EX = g.addNode(uniq(g, `sm_exT_${op.id}`))
    .init(new TensorNode.Builder(X.literalType as DataType, xShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, exp, EX, X.literalType as DataType, EX.shape);

  // ---- den = ReduceSum(ex, axes=[axis], keepdims=1)
  const rsum = g.addNode(uniq(g, `sm_rsum_${op.id}`))
    .init(new OperationNode.Builder("ReduceSum", [EX, makeI64ShapeConst(g, `sm_axes_${op.id}`, [axis])], { keepdims: 1 }))
    .as(OperationNode);
  const DEN = g.addNode(uniq(g, `sm_denT_${op.id}`))
    .init(new TensorNode.Builder(X.literalType as DataType, redShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, rsum, DEN, X.literalType as DataType, DEN.shape);

  // ---- Y = Div(ex, den)
  const div = g.addNode(uniq(g, `sm_div_${op.id}`))
    .init(new OperationNode.Builder("Div", [EX, DEN], {}))
    .as(OperationNode);
  g.addEdge(div, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);

  // ---- Remove original node
  g.getNodeById(op.id)?.remove?.();

  return true;
}
