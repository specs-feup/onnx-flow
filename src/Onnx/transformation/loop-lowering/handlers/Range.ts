import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { uniq } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopCtx, resolveFusedInput } from "../BuildLoop.js";

/* ============================== HANDLER ================================== */

export default function handleRange(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx
): TensorNode.Class {
  // Resolve inputs as Tensors in the *body* graph
  const [start, limit, delta] = op.getInputs()!.map(inp =>
    resolveFusedInput(g, inp, ctx, op, /*flatten*/ false, /*returnGather*/ false)
  );

  // Compute: y = start + (Cast(iter) * delta)   (all scalar [])
  const yTy = start.literalType;

  const iterCastN = g.addNode(uniq(g, `range_iter_cast_${op.id}`))
    .init(new OperationNode.Builder("Cast", [ctx.iter], { to: yTy }))
    .as(OperationNode);
  const iterCast = g.addNode(uniq(g, `range_iter_cast_out_${op.id}`))
    .init(new TensorNode.Builder(yTy, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(iterCastN, iterCast).init(new OnnxEdge.Builder(iterCast.literalType, iterCast.shape)).as(OnnxEdge);

  const mulN = g.addNode(uniq(g, `range_mul_${op.id}`))
    .init(new OperationNode.Builder("Mul", [iterCast, delta]))
    .as(OperationNode);
  const mul = g.addNode(uniq(g, `range_mul_out_${op.id}`))
    .init(new TensorNode.Builder(yTy, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(mulN, mul).init(new OnnxEdge.Builder(mul.literalType, mul.shape)).as(OnnxEdge);

  const addN = g.addNode(uniq(g, `range_add_${op.id}`))
    .init(new OperationNode.Builder("Add", [start, mul]))
    .as(OperationNode);
  const y = g.addNode(uniq(g, `range_y_${op.id}`))
    .init(new TensorNode.Builder(yTy, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(addN, y).init(new OnnxEdge.Builder(y.literalType, y.shape)).as(OnnxEdge);

  return y; // scalar []; your end-of-body code will Unsqueeze to [1] for ScatterElements if needed
}