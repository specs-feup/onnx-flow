import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { addEdge, scalarOfType, tensorOnesConst, toArrayLike, uniq } from "../../../Utils.js";
// import Graph from "@specs-feup/flow/graph/Graph"; // only needed if you actually finish 1D lowering

function averagePool1DHandler(
  g: OnnxGraph.Class,
  op: OperationNode.Class,
  X: TensorNode.Class,
  Y: TensorNode.Class,
  dtype: DataType,
): boolean {
  // 1D AvgPool lowering is not finished / tested. Keep the original node.
  return false;
}

/* ------------------------------ Handler ------------------------------- */
/**
 * AveragePool(N,C,H,W) → Conv-based lowering.
 */
export default function averagePoolHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "AveragePool") return false;

  // Inputs
  const ins = op.getInputs?.() ?? [];
  if (ins.length !== 1) return false;

  const X = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
  if (!X) return false;

  // Single output Y
  const outs = toArrayLike<TensorNode.Class>(
    op.getOutgoers?.targets?.filterIs?.(TensorNode),
  );
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Shapes / dtype checks
  const xShape = X.shape ?? [];
  const rank = xShape.length;

  // Support 1D (NCW) and 2D (NCHW) only
  if (rank !== 3 && rank !== 4) return false;

  const C = xShape[1];
  if (typeof C !== "number") return false; // need channels to build ones kernel

  const dtype = (X.literalType ?? DataType.FLOAT) as DataType;

  // 1D: currently we just leave it alone (handler returns false)
  if (rank === 3) {
    const ok = averagePool1DHandler(g, op, X, Y, dtype);
    return ok;
  }

  // ---------- Attributes ----------
  const a = op.getAttributes?.() ?? op.attributes ?? {};

  const autoPad = (a.auto_pad ?? "NOTSET") as string;
  const ceilMode = Number(a.ceil_mode ?? 0);
  const countIncludePad = Number(a.count_include_pad ?? 0);

  const rawKernel = (a.kernel_shape ?? []) as number[];
  const rawStrides = (a.strides ?? []) as number[];
  const rawPads = (a.pads ?? []) as number[];

  // ---------- SPECIAL CASE: KWS tiled global-like AvgPool ----------
  // This should *not* be rewritten to Conv; we want the loop builder to see it.
  const is2D =
    Array.isArray(rawKernel) &&
    rawKernel.length === 2 &&
    Array.isArray(rawStrides) &&
    rawStrides.length === 2;

  if (is2D) {
    const kH = Number(rawKernel[0]);
    const kW = Number(rawKernel[1]);
    const sH = Number(rawStrides[0] ?? 1);
    const sW = Number(rawStrides[1] ?? 1);

    const pads = (rawPads ?? []).map(Number);
    const allPadsZero = pads.length === 0 || pads.every((p) => p === 0);

    // Matches the KWS AvgPool:
    //   - auto_pad = NOTSET
    //   - ceil_mode = 0
    //   - no pads
    //   - kernel_shape == strides (non-overlapping tiles)
    const looksLikeTiledGlobalPool =
      autoPad === "NOTSET" &&
      ceilMode === 0 &&
      allPadsZero &&
      kH === sH &&
      kW === sW;

    if (looksLikeTiledGlobalPool) {
      // DO NOT rewrite to Conv, leave the AveragePool in the graph so that
      // AveragePoolBuilder (loop-lowering) picks it up.
      return false;
    }
  }

  let kH = 1;
  let kW: number;
  let sH = 1;
  let sW: number;

  // --- kernel / strides normalisation ---------------------------------------
  if (is2D) {
    // kernel [kH, kW]
    if (!Array.isArray(rawKernel) || rawKernel.length !== 2) return false;
    [kH, kW] = rawKernel;

    // strides: allow [], [s], or [sH,sW]
    if (!Array.isArray(rawStrides)) return false;
    if (rawStrides.length === 0) {
      [sH, sW] = [1, 1];
    } else if (rawStrides.length === 1) {
      [sH, sW] = [rawStrides[0], rawStrides[0]];
    } else if (rawStrides.length === 2) {
      [sH, sW] = rawStrides;
    } else {
      return false;
    }
  } else {
    // 1D: kernel [kW], strides [] or [sW]
    if (!Array.isArray(rawKernel) || rawKernel.length !== 1) return false;
    kW = rawKernel[0];

    if (!Array.isArray(rawStrides)) return false;
    if (rawStrides.length === 0) {
      sW = 1;
    } else if (rawStrides.length === 1) {
      sW = rawStrides[0];
    } else {
      return false;
    }
  }

  // --- pads / ceil_mode handling --------------------------------------------
  // We treat auto_pad=VALID as explicit pads=0 (same semantics).
  const useAutoPad = autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER";

  let pT = 0;
  let pL = 0;
  let pB = 0;
  let pR = 0;

  if (useAutoPad) {
    // Let ConvBuilder handle SAME_* padding; we only support ceil_mode=0 here.
    if (ceilMode !== 0) return false;
  } else {
    // NOTSET / VALID / unspecified: use explicit pads
    const pads =
      rawPads.length > 0
        ? rawPads.map(Number)
        : is2D
        ? [0, 0, 0, 0]
        : [0, 0];

    if (is2D) {
      if (pads.length !== 4) return false;
      [pT, pL, pB, pR] = pads;
    } else {
      if (pads.length !== 2) return false;
      [pL, pR] = pads;
    }

    // ceil_mode emulation: bump bottom/right padding by (stride - 1)
    if (ceilMode === 1) {
      if (is2D) {
        pB += sH - 1;
        pR += sW - 1;
      } else {
        pR += sW - 1;
      }
    } else if (ceilMode !== 0) {
      return false;
    }
  }


  // Build ones kernel: [C, 1, kH, kW] for 2D, [C, 1, kW] for 1D
  const onesShape = is2D ? [C, 1, kH, kW] : [C, 1, kW];
  const Wones = tensorOnesConst(g, `AvgPool_Wones_${op.id}`, dtype, onesShape);

  // Common Conv attributes for Sum and Denom
  const sumAttrs: any = { group: C };
  if (useAutoPad) {
    sumAttrs.auto_pad = autoPad;
    sumAttrs.strides = is2D ? [sH, sW] : [sW];
  } else {
    sumAttrs.strides = is2D ? [sH, sW] : [sW];
    sumAttrs.pads = is2D ? [pT, pL, pB, pR] : [pL, pR];
  }

  // SumOut = Conv(X, Wones)
  const convSum = g
    .addNode(uniq(g, `AvgPool_ConvSum_${op.id}`))
    .init(new OperationNode.Builder("Conv", [X, Wones], sumAttrs))
    .as(OperationNode);

  const SumOut = g
    .addNode(uniq(g, `AvgPool_SumOut_${op.id}`))
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
    const shapeX = g
      .addNode(uniq(g, `AvgPool_ShapeX_${op.id}`))
      .init(new OperationNode.Builder("Shape", [X], {}))
      .as(OperationNode);

    // Shape rank matches X (3 for NCW, 4 for NCHW)
    const ShX = g
      .addNode(uniq(g, `AvgPool_ShX_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
      .as(TensorNode);
    addEdge(g, shapeX, ShX, DataType.INT64, [rank]);

    const oneScalar = scalarOfType(g, `AvgPool_one_${op.id}`, 1, dtype);
    const expand = g.addNode(uniq(g, `AvgPool_Expand_${op.id}`))
      .init(new OperationNode.Builder("Expand", [oneScalar, ShX], {}))
      .as(OperationNode);
    const OnesX = g.addNode(uniq(g, `AvgPool_OnesX_${op.id}`))
      .init(new TensorNode.Builder(dtype, X.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expand, OnesX, dtype, X.shape);

    // Denom = Conv(OnesX, Wones) with same attrs as SumOut
    const maskAttrs: any = { group: C };
    if (useAutoPad) {
      maskAttrs.auto_pad = autoPad;
      maskAttrs.strides = is2D ? [sH, sW] : [sW];
    } else {
      maskAttrs.strides = is2D ? [sH, sW] : [sW];
      maskAttrs.pads = is2D ? [pT, pL, pB, pR] : [pL, pR];
    }

    const convMask = g
      .addNode(uniq(g, `AvgPool_ConvMask_${op.id}`))
      .init(new OperationNode.Builder("Conv", [OnesX, Wones], maskAttrs))
      .as(OperationNode);

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
