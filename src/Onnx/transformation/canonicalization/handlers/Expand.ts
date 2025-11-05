import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { DataType } from "../../../OnnxTypes.js";
import { uniq, addEdge, toArrayLike } from "../../../Utils.js";

export default function expandHandler(
  g: OnnxGraph.Class,
  op: OperationNode.Class
): boolean {
  if (op.type !== "Expand") return false;

  const ins = op.getInputs?.() ?? [];
  if (ins.length < 2) return false;

  const data = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
  const shape = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  if (!data || !shape) return false;

  const outs = toArrayLike<TensorNode.Class>(
    op.getOutgoers?.targets?.filterIs?.(TensorNode)
  );
  if (outs.length !== 1) return false;
  const Y = outs[0];

  const inDtype = (data.literalType ?? DataType.FLOAT) as DataType;

  // --- ConstantOfShape with NO 'value' attribute (defaults to 0.0 float)
  const cosOp = g
    .addNode(uniq(g, `Expand_fill_${op.id}`))
    .init(new OperationNode.Builder("ConstantOfShape", [shape], {}))
    .as(OperationNode);

  const cosOut = g
    .addNode(uniq(g, `Expand_fill_out_${op.id}`))
    .init(new TensorNode.Builder(DataType.FLOAT, Array.isArray(Y.shape) ? [...Y.shape] : [], "intermediate"))
    .as(TensorNode);

  addEdge(g, cosOp, cosOut, DataType.FLOAT, cosOut.shape);

  // --- If needed, Cast fill to input dtype
  const fillForAdd =
    inDtype === DataType.FLOAT
      ? cosOut
      : (() => {
          const castOp = g
            .addNode(uniq(g, `Expand_cast_${op.id}`))
            .init(new OperationNode.Builder("Cast", [cosOut], { to: inDtype }))
            .as(OperationNode);

          const castOut = g
            .addNode(uniq(g, `Expand_cast_out_${op.id}`))
            .init(new TensorNode.Builder(inDtype, Array.isArray(Y.shape) ? [...Y.shape] : [], "intermediate"))
            .as(TensorNode);

          addEdge(g, castOp, castOut, inDtype, castOut.shape);
          return castOut;
        })();

  // Add(data, fill) => Y (broadcast = expand)
  const addOp = g
    .addNode(uniq(g, `Expand_add_${op.id}`))
    .init(new OperationNode.Builder("Add", [data, fillForAdd], {}))
    .as(OperationNode);

  addEdge(g, addOp, Y, Y.literalType as DataType, Y.shape);

  g.getNodeById(op.id)?.remove();
  return true;
}
