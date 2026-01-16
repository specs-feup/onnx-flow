import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { toArrayLike, uniq, addEdge, scalarOfType, constI64 } from "../../../Utils.js";

/**
 * QuantizeLinear(x, scale, zero_point)
 * Formula: y = saturate(round(x / scale) + zero_point)
 *
 * Decomposition:
 * 1. Expand Scale (S) and ZeroPoint (Z) to match Shape(X).
 * 2. Div:  temp = x / S
 * 3. Round: temp = Round(temp)
 * 4. Add:  temp = temp + Z_float
 * 5. Clip: temp = Clip(temp, min_T, max_T)
 * 6. Cast: y = Cast(temp, to=T)
 */
export default function quantizeLinearHandler(
  g: OnnxGraph.Class,
  op: OperationNode.Class
): boolean {
  if (op.type !== "QuantizeLinear") return false;

  const ins = op.getInputs?.() ?? [];
  if (ins.length < 2) return false;

  const X = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
  const S = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  // Zero point is optional.
  const Z = ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined;

  if (!X || !S) return false;

  const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Target type comes from Z (if present) or Y.
  const targetType = Z ? Z.literalType : (Y.literalType ?? DataType.UINT8);
  const floatT = X.literalType ?? DataType.FLOAT;

  const a = op.getAttributes?.() ?? op.attributes ?? {};
  const axisAttr = Number(a.axis ?? 1);

  // 1. Prepare Inputs (Scale is float, Z needs cast to float)
  let Zf: TensorNode.Class;
  if (Z) {
    const castZ = g.addNode(uniq(g, `QL_CastZ_${op.id}`))
      .init(new OperationNode.Builder("Cast", [Z], { to: floatT }))
      .as(OperationNode);
    Zf = g.addNode(uniq(g, `QL_Zf_${op.id}`))
      .init(new TensorNode.Builder(floatT, Z.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, castZ, Zf, floatT, Z.shape);
  } else {
    Zf = scalarOfType(g, `QL_Zzero_${op.id}`, 0, floatT);
  }

  // 2. Broadcast S and Z to X's shape (Scalar or Per-Axis)
  const rank = X.shape?.length ?? 0;
  const shapeXop = g.addNode(uniq(g, `QL_ShapeX_${op.id}`))
    .init(new OperationNode.Builder("Shape", [X], {}))
    .as(OperationNode);
  const shapeX = g.addNode(uniq(g, `QL_shapeX_${op.id}`))
    .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
    .as(TensorNode);
  addEdge(g, shapeXop, shapeX, DataType.INT64, [rank]);

  const sRank = S.shape?.length ?? 0;
  // Heuristic: if S has rank 1 and X has rank > 1, assume per-axis if not 1-element
  const isPerAxis = sRank === 1 && rank > 1; 

  let Sx: TensorNode.Class = S;
  let Zx: TensorNode.Class = Zf;

  if (isPerAxis) {
    const axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
    
    // Unsqueeze on all dims EXCEPT 'axis'
    const axesVals = [];
    for (let i = 0; i < rank; i++) {
      if (i !== axis) axesVals.push(i);
    }
    const axes = constI64(g, `QL_axes_${op.id}`, axesVals);

    // Unsqueeze S
    const uSop = g.addNode(uniq(g, `QL_unsqS_${op.id}`))
      .init(new OperationNode.Builder("Unsqueeze", [S, axes]))
      .as(OperationNode);
    const Sranked = g.addNode(uniq(g, `QL_Srank_${op.id}`))
      .init(new TensorNode.Builder(floatT, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, uSop, Sranked, floatT, []);

    // Unsqueeze Z
    const uZop = g.addNode(uniq(g, `QL_unsqZ_${op.id}`))
      .init(new OperationNode.Builder("Unsqueeze", [Zf, axes]))
      .as(OperationNode);
    const Zranked = g.addNode(uniq(g, `QL_Zrank_${op.id}`))
      .init(new TensorNode.Builder(floatT, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, uZop, Zranked, floatT, []);

    // Expand S
    const expSop = g.addNode(uniq(g, `QL_ExpandS_${op.id}`))
      .init(new OperationNode.Builder("Expand", [Sranked, shapeX], {}))
      .as(OperationNode);
    Sx = g.addNode(uniq(g, `QL_Sx_${op.id}`))
      .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expSop, Sx, floatT, X.shape);

    // Expand Z
    const expZop = g.addNode(uniq(g, `QL_ExpandZ_${op.id}`))
      .init(new OperationNode.Builder("Expand", [Zranked, shapeX], {}))
      .as(OperationNode);
    Zx = g.addNode(uniq(g, `QL_Zx_${op.id}`))
      .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expZop, Zx, floatT, X.shape);
  } else {
    // Per-tensor (Scalar)
    const expSop = g.addNode(uniq(g, `QL_ExpandS_${op.id}`))
      .init(new OperationNode.Builder("Expand", [S, shapeX], {}))
      .as(OperationNode);
    Sx = g.addNode(uniq(g, `QL_Sx_${op.id}`))
      .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expSop, Sx, floatT, X.shape);

    const expZop = g.addNode(uniq(g, `QL_ExpandZ_${op.id}`))
      .init(new OperationNode.Builder("Expand", [Zf, shapeX], {}))
      .as(OperationNode);
    Zx = g.addNode(uniq(g, `QL_Zx_${op.id}`))
      .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expZop, Zx, floatT, X.shape);
  }

  // 3. Div = X / Scale
  const divOp = g.addNode(uniq(g, `QL_Div_${op.id}`))
    .init(new OperationNode.Builder("Div", [X, Sx], {}))
    .as(OperationNode);
  const divOut = g.addNode(uniq(g, `QL_DivOut_${op.id}`))
    .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, divOp, divOut, floatT, X.shape);

  // 4. Round
  const roundOp = g.addNode(uniq(g, `QL_Round_${op.id}`))
    .init(new OperationNode.Builder("Round", [divOut], {}))
    .as(OperationNode);
  const roundOut = g.addNode(uniq(g, `QL_RoundOut_${op.id}`))
    .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, roundOp, roundOut, floatT, X.shape);

  // 5. Add Zero Point
  const addOp = g.addNode(uniq(g, `QL_Add_${op.id}`))
    .init(new OperationNode.Builder("Add", [roundOut, Zx], {}))
    .as(OperationNode);
  const addOut = g.addNode(uniq(g, `QL_AddOut_${op.id}`))
    .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, addOp, addOut, floatT, X.shape);

  // 6. Clip (Saturate)
  let minVal = 0;
  let maxVal = 255;
  if (targetType === DataType.INT8) {
    minVal = -128;
    maxVal = 127;
  }

  const minT = scalarOfType(g, `QL_min_${op.id}`, minVal, floatT);
  const maxT = scalarOfType(g, `QL_max_${op.id}`, maxVal, floatT);

  const clipOp = g.addNode(uniq(g, `QL_Clip_${op.id}`))
    .init(new OperationNode.Builder("Clip", [addOut, minT, maxT], {}))
    .as(OperationNode);
  const clipOut = g.addNode(uniq(g, `QL_ClipOut_${op.id}`))
    .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, clipOp, clipOut, floatT, X.shape);

  // 7. Cast to target type
  const finalCastOp = g.addNode(uniq(g, `QL_FinalCast_${op.id}`))
    .init(new OperationNode.Builder("Cast", [clipOut], { to: targetType }))
    .as(OperationNode);
  
  // Wire to original Y
  addEdge(g, finalCastOp, Y, targetType, Y.shape);

  // Remove original node
  g.getNodeById(op.id).remove();

  return true;
}