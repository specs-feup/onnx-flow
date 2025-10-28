import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
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

function tensorOnesConst(
  g: OnnxGraph.Class,
  name: string,
  dtype: DataType,
  shape: number[]
): TensorNode.Class {
  const size = shape.reduce((a, b) => a * b, 1);
  const ones = new Array<number>(size).fill(1);
  const proto = makeTensorProto(dtype, shape, ones);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, shape, "constant", proto))
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
 * AveragePool(N,C,H,W) →
 *   Wones = Constant [C,1,kH,kW] of ones (dtype=X)
 *   SumOut = Conv(X, Wones, group=C, strides, pads)
 *   if count_include_pad==1:
 *       Y = SumOut / (kH*kW)                // scalar divisor
 *   else:
 *       OnesX   = Expand(1, Shape(X))       // per-position ones
 *       Denom   = Conv(OnesX, Wones, group=C, strides, pads)
 *       Y       = SumOut / Denom
 *
 * Fires only for:
 *   - rank-4 NCHW input, known C (channels)
 *   - ceil_mode == 0
 */
export default function averagePoolHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "AveragePool") return false;

  // Inputs
  const ins = op.getInputs?.() ?? [];
  if (ins.length !== 1) return false;

  const X = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
  if (!X) return false;

  // Single output Y
  const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Shapes / dtype checks
  const xShape = X.shape ?? [];
  if (xShape.length !== 4) return false; // expect NCHW
  const C = xShape[1];
  if (typeof C !== "number") return false; // need channels to build [C,1,kH,kW]

  const dtype = (X.literalType ?? DataType.FLOAT) as DataType;

  // Attributes
  const a = (op as any).getAttributes?.() ?? (op as any).attributes ?? {};
  const kernel = (a.kernel_shape ?? []) as number[];
  const strides = (a.strides ?? [1, 1]) as number[];
  const pads = (a.pads ?? [0, 0, 0, 0]) as number[];
  const ceilMode = Number(a.ceil_mode ?? 0);
  const countIncludePad = Number(a.count_include_pad ?? 0);

  if (!Array.isArray(kernel) || kernel.length !== 2) return false;
  if (!Array.isArray(strides) || strides.length !== 2) return false;
  if (!Array.isArray(pads) || pads.length !== 4) return false;
  if (ceilMode !== 0) return false; // our Conv-based lowering aligns with floor mode

  const [kH, kW] = kernel;
  const [sH, sW] = strides;
  const [pT, pL, pB, pR] = pads;

  // Build ones kernel: [C, 1, kH, kW]
  const Wones = tensorOnesConst(g, `AvgPool_Wones_${op.id}`, dtype, [C, 1, kH, kW]);

  // SumOut = Conv(X, Wones) with group=C, strides, pads
  const convSum = g.addNode(uniq(g, `AvgPool_ConvSum_${op.id}`))
    .init(new OperationNode.Builder("Conv", [X, Wones], {
      group: C,
      strides: [sH, sW],
      pads: [pT, pL, pB, pR],
    }))
    .as(OperationNode);

  const SumOut = g.addNode(uniq(g, `AvgPool_SumOut_${op.id}`))
    .init(new TensorNode.Builder(dtype, Y.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, convSum, SumOut, dtype, Y.shape);

  let finalProducer: OperationNode.Class | undefined;

  if (countIncludePad === 1) {
    // Divide by constant kH*kW
    const divConst = scalarOfType(g, `AvgPool_div_${op.id}`, kH * kW, dtype);
    const div = g.addNode(uniq(g, `AvgPool_DivC_${op.id}`))
      .init(new OperationNode.Builder("Div", [SumOut, divConst], {}))
      .as(OperationNode);
    finalProducer = div;
  } else {
    // Build OnesX = Expand(1, Shape(X))
    const shapeX = g.addNode(uniq(g, `AvgPool_ShapeX_${op.id}`))
      .init(new OperationNode.Builder("Shape", [X], {}))
      .as(OperationNode);
    const ShX = g.addNode(uniq(g, `AvgPool_ShX_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [4], "intermediate"))
      .as(TensorNode);
    addEdge(g, shapeX, ShX, DataType.INT64, [4]);

    const oneScalar = scalarOfType(g, `AvgPool_one_${op.id}`, 1, dtype);
    const expand = g.addNode(uniq(g, `AvgPool_Expand_${op.id}`))
      .init(new OperationNode.Builder("Expand", [oneScalar, ShX], {}))
      .as(OperationNode);
    const OnesX = g.addNode(uniq(g, `AvgPool_OnesX_${op.id}`))
      .init(new TensorNode.Builder(dtype, X.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expand, OnesX, dtype, X.shape);

    // Denom = Conv(OnesX, Wones) with same attrs
    const convMask = g.addNode(uniq(g, `AvgPool_ConvMask_${op.id}`))
      .init(new OperationNode.Builder("Conv", [OnesX, Wones], {
        group: C,
        strides: [sH, sW],
        pads: [pT, pL, pR, pB] /* placeholder, will overwrite below */,
      }))
      .as(OperationNode);
    // note: ensure pads order matches [pT, pL, pB, pR]
    (convMask as any).attributes.pads = [pT, pL, pB, pR];

    const Denom = g.addNode(uniq(g, `AvgPool_Denom_${op.id}`))
      .init(new TensorNode.Builder(dtype, Y.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, convMask, Denom, dtype, Y.shape);

    // Y = SumOut / Denom
    const div = g.addNode(uniq(g, `AvgPool_DivM_${op.id}`))
      .init(new OperationNode.Builder("Div", [SumOut, Denom], {}))
      .as(OperationNode);
    finalProducer = div;
  }

  // Wire final producer → Y
  if (finalProducer) {
    g.addEdge(finalProducer, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
  }

  // Remove original AveragePool op
  g.getNodeById(op.id).remove();

  return true;
}
