import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { makeTensorProto } from "../../Utilities.js";


/* ------------------------------- utils -------------------------------- */
function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}

function toArrayLike<T = any>(nc: any): T[] {
  return nc?.toArray?.() ?? nc ?? [];
}

// rank-0 scalar constant
function scalarOfType(
  g: OnnxGraph.Class,
  name: string,
  v: number,
  dtype: DataType
): TensorNode.Class {
  const proto = makeTensorProto(dtype, [], [v]);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, [], "constant", proto))
    .as(TensorNode);
}

// 1-D INT64 constant
function constI64(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [vals.length], "constant", proto))
    .as(TensorNode);
}

function addEdge(
  g: OnnxGraph.Class,
  srcOp: OperationNode.Class,
  dstTensor: TensorNode.Class,
  dtype: DataType,
  shape?: Array<number | String | undefined>
) {
  g.addEdge(srcOp, dstTensor)
    .init(new OnnxEdge.Builder(dtype, shape ?? dstTensor.shape))
    .as(OnnxEdge);
}

/* ------------------------------ handler ------------------------------- */
/**
 * DequantizeLinear(x, scale[, zero_point], axis)
 * → Cast(x, floatT) ; Cast(zp, floatT or use 0) ; (Unsqueeze per-axis) ; Sub ; Mul
 *
 * Fires when:
 *  - x/scale (and optional zp) are tensors,
 *  - For per-axis case, rank(x) is known and scale/zp are 1-D with concrete length.
 * Otherwise returns false to avoid breaking dynamic models.
 */
export default function dequantizeLinearHandler(
  g: OnnxGraph.Class,
  op: OperationNode.Class
): boolean {
  if (op.type !== "DequantizeLinear") return false;

  // Inputs in topo order
  const ins = op.getInputs?.() ?? [];
  if (ins.length < 2) return false;

  const X = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
  const S = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  const Z = ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined;
  if (!X || !S) return false;

  // Single output tensor Y
  const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Attributes
  const a = (op as any).getAttributes?.() ?? (op as any).attributes ?? {};
  const axisAttr = Number(a.axis ?? 0); // ONNX default axis is 0

  // Determine float target dtype for dequantized outputs
  // Prefer Y's literal type if it's a float; otherwise fallback to FLOAT
  const yType = (Y.literalType ?? DataType.FLOAT) as DataType;
  const floatTypes = new Set([DataType.FLOAT, DataType.FLOAT16, DataType.BFLOAT16, DataType.DOUBLE]);
  const floatT: DataType = floatTypes.has(yType) ? yType : DataType.FLOAT;

  // Rank/shape info
  const xShape = X.shape ?? [];
  const rank = xShape.length;

  // Identify per-tensor vs per-axis (we only special-case per-axis when S is rank-1)
  const sRank = S.shape?.length ?? undefined;
  const zRank = Z ? (Z.shape?.length ?? undefined) : undefined;
  const perTensor = sRank === 0;          // scalar scale
  const perAxis   = sRank === 1;          // 1-D scale along 'axis'

  // If per-axis we need rank known to build Unsqueeze axes
  if (perAxis && rank === undefined) return false;

  // Normalize axis if per-axis
  const axis = perAxis ? (axisAttr < 0 ? axisAttr + rank : axisAttr) : 0;
  if (perAxis && (axis < 0 || axis >= rank)) return false;

  // Basic dtype sanity: scale must be float-ish; if it's not, bail
  if (!floatTypes.has(S.literalType as DataType)) return false;

  // Cast X to floatT
  const castX = g.addNode(uniq(g, `DQL_CastX_${op.id}`))
    .init(new OperationNode.Builder("Cast", [X], { to: floatT }))
    .as(OperationNode);
  const Xf = g.addNode(uniq(g, `DQL_Xf_${op.id}`))
    .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, castX, Xf, floatT, X.shape);

  // Prepare zero_point branch
  let Zb: TensorNode.Class;
  if (Z) {
    // Cast zp to floatT
    const castZ = g.addNode(uniq(g, `DQL_CastZ_${op.id}`))
      .init(new OperationNode.Builder("Cast", [Z], { to: floatT }))
      .as(OperationNode);
    const Zf = g.addNode(uniq(g, `DQL_Zf_${op.id}`))
      .init(new TensorNode.Builder(floatT, Z.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, castZ, Zf, floatT, Z.shape);
    Zb = Zf;
  } else {
    // zp default = 0
    Zb = scalarOfType(g, `DQL_Zzero_${op.id}`, 0, floatT);
  }

  // Prepare scale branch
  let Sb: TensorNode.Class = S;

  // For per-axis: Unsqueeze S and Zb to match rank(X) on all dims except 'axis'
  if (perAxis) {
    // Validate expected shapes where possible
    const sLen = typeof S.shape?.[0] === "number" ? (S.shape![0] as number) : undefined;
    const xAxisDim = typeof xShape?.[axis] === "number" ? (xShape![axis] as number) : undefined;
    if (sLen !== undefined && xAxisDim !== undefined && sLen !== xAxisDim) {
      // incompatible per-axis length
      return false;
    }
    // If zp is 1-D, keep; if scalar or 1-D both are fine after Unsqueeze.
    // Build axes list: all dims except 'axis'
    const axesVals: number[] = [];
    for (let i = 0; i < rank; i++) if (i !== axis) axesVals.push(i);

    const axesConst = constI64(g, `DQL_Axes_${op.id}`, axesVals);

    // Unsqueeze S
    const uS = g.addNode(uniq(g, `DQL_UnsqS_${op.id}`))
      .init(new OperationNode.Builder("Unsqueeze", [S, axesConst], {}))
      .as(OperationNode);
    const Sbu = g.addNode(uniq(g, `DQL_Sb_${op.id}`))
      .init(new TensorNode.Builder(S.literalType as DataType, xShape, "intermediate"))
      .as(TensorNode);
    addEdge(g, uS, Sbu, S.literalType as DataType, xShape);
    Sb = Sbu;

    // If Zb is not scalar already, Unsqueeze it the same way
    const zIsScalar = (Z?.shape?.length ?? 0) === 0;
    if (!zIsScalar || Z === undefined) {
      const uZ = g.addNode(uniq(g, `DQL_UnsqZ_${op.id}`))
        .init(new OperationNode.Builder("Unsqueeze", [Zb, axesConst], {}))
        .as(OperationNode);
      const Zbu = g.addNode(uniq(g, `DQL_Zb_${op.id}`))
        .init(new TensorNode.Builder(floatT, xShape, "intermediate"))
        .as(TensorNode);
      addEdge(g, uZ, Zbu, floatT, xShape);
      Zb = Zbu;
    }
  }

  // Sub: (float(x) - (Zb))
  const sub = g.addNode(uniq(g, `DQL_Sub_${op.id}`))
    .init(new OperationNode.Builder("Sub", [Xf, Zb], {}))
    .as(OperationNode);
  const D = g.addNode(uniq(g, `DQL_D_${op.id}`))
    .init(new TensorNode.Builder(floatT, xShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, sub, D, floatT, xShape);

  // Mul: D * Sb
  const mul = g.addNode(uniq(g, `DQL_Mul_${op.id}`))
    .init(new OperationNode.Builder("Mul", [D, Sb], {}))
    .as(OperationNode);

  // Final edge to Y
  g.addEdge(mul, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);

  // Remove original DequantizeLinear
  g.getNodeById(op.id).remove();

  return true;
}
