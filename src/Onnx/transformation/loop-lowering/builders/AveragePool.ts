import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import {
  uniq,
  makeTensorConst,
  int64Vec,
  scalarInt64,
  bool,
  tensorOnesConst,
  zeroTensor,
} from "@specs-feup/onnx-flow/Onnx/Utils";
import {
  LoopBuilder,
  BuildResult,
  unsqueezeIdx,
  LoopCtx,
  decodeMixedRadix,
} from "../BuildLoop.js";
import inferShapes from "@specs-feup/onnx-flow/Onnx/InferShapes";

function resolveShape(t: TensorNode.Class): number[] {
  // If the tensor already has a shape, just use it.
  if (t.shape && t.shape.length) {
    return t.shape as number[];
  }

  // Try incoming edges first
  const incs = t.getIncomers ?? [];
  for (const e of incs) {
    if (e.shape && e.shape.length) {
      const s = e.shape.slice();
      // Cache it on the tensor so later passes see it as well
      t.setShape(s);
      return s;
    }
  }

  // Fallback: try outgoing edges (sometimes only consumers got a shape)
  const outs = t.getOutgoers ?? [];
  for (const e of outs) {
    if (e.shape && e.shape.length) {
      const s = e.shape.slice();

      t.setShape(s);
      return s;
    }
  }

  // Still unknown
  return [];
}

function getAttr<T>(
  op: OperationNode.Class,
  name: string,
  def: T
): T | number[] | undefined {
  if (!op.attributes) return def;
  const attr = op.attributes.find((a) => a.name === name);
  if (!attr) return def;

  if (attr.type === "INTS") return attr.ints ?? def;
  if (attr.type === "INT") return attr.i ?? def;
  if (attr.type === "FLOAT") return attr.f ?? def;

  return def;
}

class AveragePoolBuilder implements LoopBuilder {
  canHandle(chain: OperationNode.Class[]): boolean {
    if (chain.length !== 1) return false;
    const op = chain[0];

    // Keep supporting GlobalAveragePool in general.
    if (op.type === "GlobalAveragePool") {
      return true;
    }

    if (op.type !== "AveragePool") return false;

    const a = op.getAttributes?.() ??  op.attributes ?? {};

    const autoPad = (a.auto_pad ?? "NOTSET") as string;
    const ceilMode = Number(a.ceil_mode ?? 0);
    const kernelShape = (a.kernel_shape ?? []) as number[];
    const strides = (a.strides ?? []) as number[];
    const pads = (a.pads ?? []) as number[];

    // Only 2D spatial; if no proper kernel/stride, let other passes handle it.
    if (kernelShape.length !== 2 || strides.length !== 2) return false;

    const [kH, kW] = kernelShape.map(Number);
    const [sH, sW] = strides.map(Number);
    const allPadsZero = pads.length === 0 || pads.every((p) => Number(p) === 0);

    // Match *exactly* the KWS tiled global-like pooling:
    const looksLikeKwsTiled =
      autoPad === "NOTSET" &&
      ceilMode === 0 &&
      allPadsZero &&
      kH === sH &&
      kW === sW;

    return looksLikeKwsTiled;
  }

  build(
    chain: OperationNode.Class[],
    outer: OnnxGraph.Class,
    opts: { fuse: boolean; recurse: boolean; coalesce: boolean }
  ): BuildResult {
    const avg = chain[0];

    const inputsArr =
      avg.getInputs()?.filter((n) => n.is(TensorNode)).map((n) => n.as(TensorNode)) ??
      [];
    if (inputsArr.length < 1) {
      throw new Error("AveragePoolBuilder: AveragePool must have at least X as input");
    }

    const X = inputsArr[0];
    const output = avg.getOutgoers.targets.filterIs(TensorNode).first();
    const Y = output;

    const xShape = resolveShape(X);
    if (xShape.length < 3) {
      throw new Error(
        `AveragePoolBuilder: expected at least 3D input (N,C,H,...) but got ${xShape}`
      );
    }

    const N = xShape[0];
    const C = xShape[1];

    const a = avg.getAttributes?.() ??  avg.attributes ?? {};

    const autoPad = (a.auto_pad ?? "NOTSET") as string;
    const ceilMode = Number(a.ceil_mode ?? 0);
    const countIncludePad = Number(a.count_include_pad ?? 0);

    const kernelShape = (a.kernel_shape ?? []) as number[];
    const strides = (a.strides ?? []) as number[];
    const pads = (a.pads ?? []) as number[];

    const isGlobal = avg.type === "GlobalAveragePool";

    // For 2D spatial case (N, C, H, W)
    if (xShape.length !== 4) {
      throw new Error(
        `AveragePoolBuilder: currently only handles 2D spatial avg pool (N,C,H,W). got=${xShape}`
      );
    }

    const H = xShape[2];
    const W_in = xShape[3];

    let kH: number;
    let kW: number;
    let strideH: number;
    let strideW: number;
    let padTop: number;
    let padLeft: number;
    let padBottom: number;
    let padRight: number;

    if (isGlobal) {
      // GlobalAveragePool => kernel covers entire HxW
      kH = H;
      kW = W_in;
      strideH = 1;
      strideW = 1;
      padTop = 0;
      padLeft = 0;
      padBottom = 0;
      padRight = 0;
    } else {
      if (kernelShape.length !== 2) {
        throw new Error(
          `AveragePoolBuilder: kernel_shape must have 2 elements for 2D, got=${kernelShape}`
        );
      }
      kH = kernelShape[0];
      kW = kernelShape[1];

      // Strides default = 1
      strideH = strides.length > 0 ? strides[0] : 1;
      strideW = strides.length > 1 ? strides[1] : strideH;

      // Pads default = 0
      padTop = pads[0] ?? 0;
      padLeft = pads[1] ?? 0;
      padBottom = pads[2] ?? 0;
      padRight = pads[3] ?? 0;
    }

    const H_padded = H + padTop + padBottom;
    const W_padded = W_in + padLeft + padRight;

    let H_out = Math.floor((H_padded - kH) / strideH + 1);
    let W_out = Math.floor((W_padded - kW) / strideW + 1);

    // Fallback for edge cases (like the kws_ref AveragePool) where the
    // naive geometric formula would give <= 0. We clamp to at least 1 in
    // each spatial dimension and rely on the in-bounds checks inside the
    // loop body to ignore out-of-range positions.
    if (H_out <= 0) H_out = 1;
    if (W_out <= 0) W_out = 1;

    const yShape = [N, C, H_out, W_out];

    // totalIters = N * C * H_out * W_out; we rely on precomputed yShape or
    // inferred from outside, so we do not recompute from H_out/W_out (shape inference
    // already ran). This avoids the ConvBuilder "H_out/W_out <= 0" failure
    // for weird geometric combos.
    const totalIters = N * C * H_out * W_out;

    const elemTy = (Y.literalType ?? DataType.FLOAT) as DataType;
    const carryLen = totalIters;

    // ---- Loop context / inputs ----------------------------------------------
    const inputs = new Map<string, TensorNode.Class>();
    inputs.set(X.id, X);

    // ---- Build loop body graph ----------------------------------------------
    const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    // iter: INT64 scalar
    const iter = body
      .addNode(uniq(body, "iter"))
      .init(new TensorNode.Builder(DataType.INT64, [], "input"))
      .as(TensorNode);

    // cond_in: bool scalar
    const condIn = body
      .addNode(uniq(body, "cond_in"))
      .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
      .as(TensorNode);

    // full carry: [carryLen], shape matches Y flattened
    const carry = body
      .addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [carryLen], "input"))
      .as(TensorNode);

    // Captured X (same ID so BuildLoop can wire it)
    const X_in = body
      .addNode(X.id)
      .init(new TensorNode.Builder(X.literalType, X.shape, "intermediate"))
      .as(TensorNode);

    // axes=[0] constant & unsqueezed iter index for ScatterElements indices
    const axes0 = int64Vec([0]);
    const axesTensor = makeTensorConst(
      body,
      "axes0",
      DataType.INT64,
      "constant",
      axes0
    );

    const iterUnsq = unsqueezeIdx(body, iter, axesTensor, "avgpIdx");

    // Mixed-radix decode to (n, c, ho, wo)
    const dims = [N, C, H_out, W_out];
    const [r0Const, r1Const, r2Const, r3Const] = decodeMixedRadix(
      body,
      iter,
      dims,
      "rDecode"
    );
    const nIdx = r0Const; // [1]
    const cIdx = r1Const; // [1]
    const hoIdx = r2Const; // [1]
    const woIdx = r3Const; // [1]

    // Compute top-left corner (hStart, wStart) of the window in padded coords
    // hStart = ho * strideH - padTop
    // wStart = wo * strideW - padLeft

    const strideHConst = makeTensorConst(
      body,
      "strideH",
      DataType.INT64,
      "constant",
      scalarInt64(strideH)
    );
    const strideWConst = makeTensorConst(
      body,
      "strideW",
      DataType.INT64,
      "constant",
      scalarInt64(strideW)
    );
    const padTopConst = makeTensorConst(
      body,
      "padTop",
      DataType.INT64,
      "constant",
      scalarInt64(padTop)
    );
    const padLeftConst = makeTensorConst(
      body,
      "padLeft",
      DataType.INT64,
      "constant",
      scalarInt64(padLeft)
    );

    const hoMulStride = body
      .addNode(uniq(body, "ho_mul_stride"))
      .init(new OperationNode.Builder("Mul", [hoIdx, strideHConst]))
      .as(OperationNode);
    const hoMulStrideTensor = body
      .addNode(uniq(body, "ho_mul_stride_t"))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(hoIdx, hoMulStride)
      .init(new OnnxEdge.Builder(hoIdx.literalType, hoIdx.shape));
    body
      .addEdge(strideHConst, hoMulStride)
      .init(new OnnxEdge.Builder(strideHConst.literalType, strideHConst.shape));
    body
      .addEdge(hoMulStride, hoMulStrideTensor)
      .init(new OnnxEdge.Builder(hoMulStrideTensor.literalType, hoMulStrideTensor.shape));

    const hStartOp = body
      .addNode(uniq(body, "hStart_op"))
      .init(new OperationNode.Builder("Sub", [hoMulStrideTensor, padTopConst]))
      .as(OperationNode);
    const hStart = body
      .addNode(uniq(body, "hStart"))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(hoMulStrideTensor, hStartOp)
      .init(new OnnxEdge.Builder(hoMulStrideTensor.literalType, hoMulStrideTensor.shape));
    body
      .addEdge(padTopConst, hStartOp)
      .init(new OnnxEdge.Builder(padTopConst.literalType, padTopConst.shape));
    body
      .addEdge(hStartOp, hStart)
      .init(new OnnxEdge.Builder(hStart.literalType, hStart.shape));

    const woMulStride = body
      .addNode(uniq(body, "wo_mul_stride"))
      .init(new OperationNode.Builder("Mul", [woIdx, strideWConst]))
      .as(OperationNode);
    const woMulStrideTensor = body
      .addNode(uniq(body, "wo_mul_stride_t"))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(woIdx, woMulStride)
      .init(new OnnxEdge.Builder(woIdx.literalType, woIdx.shape));
    body
      .addEdge(strideWConst, woMulStride)
      .init(new OnnxEdge.Builder(strideWConst.literalType, strideWConst.shape));
    body
      .addEdge(woMulStride, woMulStrideTensor)
      .init(new OnnxEdge.Builder(woMulStrideTensor.literalType, woMulStrideTensor.shape));

    const wStartOp = body
      .addNode(uniq(body, "wStart_op"))
      .init(new OperationNode.Builder("Sub", [woMulStrideTensor, padLeftConst]))
      .as(OperationNode);
    const wStart = body
      .addNode(uniq(body, "wStart"))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(woMulStrideTensor, wStartOp)
      .init(new OnnxEdge.Builder(woMulStrideTensor.literalType, woMulStrideTensor.shape));
    body
      .addEdge(padLeftConst, wStartOp)
      .init(new OnnxEdge.Builder(padLeftConst.literalType, padLeftConst.shape));
    body
      .addEdge(wStartOp, wStart)
      .init(new OnnxEdge.Builder(wStart.literalType, wStart.shape));

    // Now we iterate over the kernel window kH x kW, summing contributions.
    // We'll do this by flattening the window into 1D, but conceptually:
    //
    //  for kh in [0..kH-1]:
    //    for kw in [0..kW-1]:
    //      h = hStart + kh
    //      w = wStart + kw
    //      if 0 <= h < H_padded and 0 <= w < W_padded:
    //        (map to unpadded coords, check if inside input)
    //        sum += X[n, c, h_unpadded, w_unpadded]
    //
    // We'll accumulate in sumVal, track "count" of included elements, and
    // then out = sumVal / count.

    // We flatten the inner kernel indices into a single range [0..kH*kW-1]
    const kernelSize = kH * kW;
    const kernelIndexConst = makeTensorConst(
      body,
      "kernelIndex",
      DataType.INT64,
      "constant",
      scalarInt64(kernelSize)
    );

    const zeroFloat = makeTensorConst(
      body,
      "zeroFloat",
      elemTy,
      "constant",
      zeroTensor(elemTy, [])   // shape [] scalar, element type = elemTy
    );

    const zeroInt = makeTensorConst(
      body,
      "zeroInt",
      DataType.INT64,
      "constant",
      scalarInt64(0)
    );


    const HConst = makeTensorConst(
      body,
      "H_const",
      DataType.INT64,
      "constant",
      scalarInt64(H)
    );
    const WConst = makeTensorConst(
      body,
      "W_const",
      DataType.INT64,
      "constant",
      scalarInt64(W_in)
    );

    const Hminus1Const = makeTensorConst(
      body,
      "H_minus1",
      DataType.INT64,
      "constant",
      scalarInt64(H - 1)
    );
    const Wminus1Const = makeTensorConst(
      body,
      "W_minus1",
      DataType.INT64,
      "constant",
      scalarInt64(W_in - 1)
    );


    // sumVal and countVal start at 0
    const sumInit = zeroFloat;
    const countInit = zeroInt;

    let sumVal = sumInit;
    let countVal = countInit;

    for (let kh = 0; kh < kH; kh++) {
      for (let kwi = 0; kwi < kW; kwi++) {
        const khConst = makeTensorConst(
          body,
          `kh_${kh}`,
          DataType.INT64,
          "constant",
          scalarInt64(kh)
        );
        const kwConst = makeTensorConst(
          body,
          `kw_${kwi}`,
          DataType.INT64,
          "constant",
          scalarInt64(kwi)
        );

        // h = hStart + kh
        const hOp = body
          .addNode(uniq(body, `h_op_${kh}`))
          .init(new OperationNode.Builder("Add", [hStart, khConst]))
          .as(OperationNode);
        const hVal = body
          .addNode(uniq(body, `h_val_${kh}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body
          .addEdge(hStart, hOp)
          .init(new OnnxEdge.Builder(hStart.literalType, hStart.shape));
        body
          .addEdge(khConst, hOp)
          .init(new OnnxEdge.Builder(khConst.literalType, khConst.shape));
        body
          .addEdge(hOp, hVal)
          .init(new OnnxEdge.Builder(hVal.literalType, hVal.shape));

        // w = wStart + kw
        const wOp = body
          .addNode(uniq(body, `w_op_${kwi}`))
          .init(new OperationNode.Builder("Add", [wStart, kwConst]))
          .as(OperationNode);
        const wVal = body
          .addNode(uniq(body, `w_val_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body
          .addEdge(wStart, wOp)
          .init(new OnnxEdge.Builder(wStart.literalType, wStart.shape));
        body
          .addEdge(kwConst, wOp)
          .init(new OnnxEdge.Builder(kwConst.literalType, kwConst.shape));
        body
          .addEdge(wOp, wVal)
          .init(new OnnxEdge.Builder(wVal.literalType, wVal.shape));

        // h >= 0
        const hGe0Op = body
          .addNode(uniq(body, `h_ge0_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("GreaterOrEqual", [hVal, zeroInt]))
          .as(OperationNode);
        const hGe0 = body
          .addNode(uniq(body, `h_ge0_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(hVal, hGe0Op).init(new OnnxEdge.Builder(hVal.literalType, hVal.shape));
        body.addEdge(hGe0Op, hGe0).init(new OnnxEdge.Builder(hGe0.literalType, hGe0.shape));

        // h < H
        const hLtHOp = body
          .addNode(uniq(body, `h_ltH_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Less", [hVal, HConst]))
          .as(OperationNode);
        const hLtH = body
          .addNode(uniq(body, `h_ltH_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(hVal, hLtHOp).init(new OnnxEdge.Builder(hVal.literalType, hVal.shape));
        body.addEdge(HConst, hLtHOp).init(new OnnxEdge.Builder(HConst.literalType, HConst.shape));
        body.addEdge(hLtHOp, hLtH).init(new OnnxEdge.Builder(hLtH.literalType, hLtH.shape));

        // w >= 0
        const wGe0Op = body
          .addNode(uniq(body, `w_ge0_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("GreaterOrEqual", [wVal, zeroInt]))
          .as(OperationNode);
        const wGe0 = body
          .addNode(uniq(body, `w_ge0_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(wVal, wGe0Op).init(new OnnxEdge.Builder(wVal.literalType, wVal.shape));
        body.addEdge(wGe0Op, wGe0).init(new OnnxEdge.Builder(wGe0.literalType, wGe0.shape));

        // w < W
        const wLtWOp = body
          .addNode(uniq(body, `w_ltW_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Less", [wVal, WConst]))
          .as(OperationNode);
        const wLtW = body
          .addNode(uniq(body, `w_ltW_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(wVal, wLtWOp).init(new OnnxEdge.Builder(wVal.literalType, wVal.shape));
        body.addEdge(WConst, wLtWOp).init(new OnnxEdge.Builder(WConst.literalType, WConst.shape));
        body.addEdge(wLtWOp, wLtW).init(new OnnxEdge.Builder(wLtW.literalType, wLtW.shape));

        // Combine to single inBounds bool: 0 <= h < H  &&  0 <= w < W
        const hInOp = body
          .addNode(uniq(body, `h_in_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("And", [hGe0, hLtH]))
          .as(OperationNode);
        const hIn = body
          .addNode(uniq(body, `h_in_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(hGe0, hInOp).init(new OnnxEdge.Builder(hGe0.literalType, hGe0.shape));
        body.addEdge(hLtH, hInOp).init(new OnnxEdge.Builder(hLtH.literalType, hLtH.shape));
        body.addEdge(hInOp, hIn).init(new OnnxEdge.Builder(hIn.literalType, hIn.shape));

        const wInOp = body
          .addNode(uniq(body, `w_in_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("And", [wGe0, wLtW]))
          .as(OperationNode);
        const wIn = body
          .addNode(uniq(body, `w_in_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(wGe0, wInOp).init(new OnnxEdge.Builder(wGe0.literalType, wGe0.shape));
        body.addEdge(wLtW, wInOp).init(new OnnxEdge.Builder(wLtW.literalType, wLtW.shape));
        body.addEdge(wInOp, wIn).init(new OnnxEdge.Builder(wIn.literalType, wIn.shape));

        const inBoundsOp = body
          .addNode(uniq(body, `inBounds_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("And", [hIn, wIn]))
          .as(OperationNode);
        const inBounds = body
          .addNode(uniq(body, `inBounds_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(hIn, inBoundsOp).init(new OnnxEdge.Builder(hIn.literalType, hIn.shape));
        body.addEdge(wIn, inBoundsOp).init(new OnnxEdge.Builder(wIn.literalType, wIn.shape));
        body.addEdge(inBoundsOp, inBounds).init(new OnnxEdge.Builder(inBounds.literalType, inBounds.shape));

        // Build dynamic indices [nIdx, cIdx, hVal, wVal] to gather X[n, c, hVal, wVal]
        // hClamped = max(0, min(hVal, H-1))
        const hMinOp = body
          .addNode(uniq(body, `h_min_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Min", [hVal, Hminus1Const]))
          .as(OperationNode);
        const hMin = body
          .addNode(uniq(body, `h_min_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(hVal, hMinOp).init(new OnnxEdge.Builder(hVal.literalType, hVal.shape));
        body.addEdge(Hminus1Const, hMinOp).init(new OnnxEdge.Builder(Hminus1Const.literalType, Hminus1Const.shape));
        body.addEdge(hMinOp, hMin).init(new OnnxEdge.Builder(hMin.literalType, hMin.shape));

        const hClampOp = body
          .addNode(uniq(body, `h_clamp_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Max", [hMin, zeroInt]))
          .as(OperationNode);
        const hClamped = body
          .addNode(uniq(body, `h_clamped_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(hMin, hClampOp).init(new OnnxEdge.Builder(hMin.literalType, hMin.shape));
        body.addEdge(zeroInt, hClampOp).init(new OnnxEdge.Builder(zeroInt.literalType, zeroInt.shape));
        body.addEdge(hClampOp, hClamped).init(new OnnxEdge.Builder(hClamped.literalType, hClamped.shape));

        // wClamped = max(0, min(wVal, W-1))
        const wMinOp = body
          .addNode(uniq(body, `w_min_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Min", [wVal, Wminus1Const]))
          .as(OperationNode);
        const wMin = body
          .addNode(uniq(body, `w_min_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(wVal, wMinOp).init(new OnnxEdge.Builder(wVal.literalType, wVal.shape));
        body.addEdge(Wminus1Const, wMinOp).init(new OnnxEdge.Builder(Wminus1Const.literalType, Wminus1Const.shape));
        body.addEdge(wMinOp, wMin).init(new OnnxEdge.Builder(wMin.literalType, wMin.shape));

        const wClampOp = body
          .addNode(uniq(body, `w_clamp_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Max", [wMin, zeroInt]))
          .as(OperationNode);
        const wClamped = body
          .addNode(uniq(body, `w_clamped_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(wMin, wClampOp).init(new OnnxEdge.Builder(wMin.literalType, wMin.shape));
        body.addEdge(zeroInt, wClampOp).init(new OnnxEdge.Builder(zeroInt.literalType, zeroInt.shape));
        body.addEdge(wClampOp, wClamped).init(new OnnxEdge.Builder(wClamped.literalType, wClamped.shape));


        // Unsqueeze scalars to shape [1] so that we can Concat them
        const nUnsq = unsqueezeIdx(body, nIdx, axesTensor, `nIdx_unsq_${kh}_${kwi}`);
        const cUnsq = unsqueezeIdx(body, cIdx, axesTensor, `cIdx_unsq_${kh}_${kwi}`);
        const hUnsq = unsqueezeIdx(body, hClamped, axesTensor, `hIdx_unsq_${kh}_${kwi}`);
        const wUnsq = unsqueezeIdx(body, wClamped, axesTensor, `wIdx_unsq_${kh}_${kwi}`);

        // Concat along axis 0 → shape [4]
        const idxCat = body
          .addNode(uniq(body, `idx_cat_${kh}_${kwi}`))
          .init(
            new OperationNode.Builder("Concat", [nUnsq, cUnsq, hUnsq, wUnsq], {
              axis: 0,
            }),
          )
          .as(OperationNode);
        const idxVec = body
          .addNode(uniq(body, `idx_vec_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [4], "intermediate"))
          .as(TensorNode);

        body
          .addEdge(nUnsq, idxCat)
          .init(new OnnxEdge.Builder(nUnsq.literalType, nUnsq.shape));
        body
          .addEdge(cUnsq, idxCat)
          .init(new OnnxEdge.Builder(cUnsq.literalType, cUnsq.shape));
        body
          .addEdge(hUnsq, idxCat)
          .init(new OnnxEdge.Builder(hUnsq.literalType, hUnsq.shape));
        body
          .addEdge(wUnsq, idxCat)
          .init(new OnnxEdge.Builder(wUnsq.literalType, wUnsq.shape));
        body
          .addEdge(idxCat, idxVec)
          .init(new OnnxEdge.Builder(idxVec.literalType, idxVec.shape));

        // Unsqueeze again to shape [1, 4] as required by GatherND
        const indices = unsqueezeIdx(body, idxVec, axesTensor, `indices_${kh}_${kwi}`);

        const gather = body
          .addNode(uniq(body, `gather_${kh}_${kwi}`))
          .init(new OperationNode.Builder("GatherND", [X_in, indices]))
          .as(OperationNode);

        const gathered = body
          .addNode(uniq(body, `gathered_${kh}_${kwi}`))
          .init(new TensorNode.Builder(elemTy, [], "intermediate"))
          .as(TensorNode);

        body
          .addEdge(X_in, gather)
          .init(new OnnxEdge.Builder(X_in.literalType, X_in.shape));
        body
          .addEdge(indices, gather)
          .init(new OnnxEdge.Builder(indices.literalType, indices.shape));
        body
          .addEdge(gather, gathered)
          .init(new OnnxEdge.Builder(gathered.literalType, gathered.shape));

        // Cast mask to elemTy and INT64
        const maskFloatOp = body
          .addNode(uniq(body, `mask_cast_f_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Cast", [inBounds], { to: elemTy }))
          .as(OperationNode);
        const maskFloat = body
          .addNode(uniq(body, `mask_cast_f_${kh}_${kwi}`))
          .init(new TensorNode.Builder(elemTy, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(inBounds, maskFloatOp).init(new OnnxEdge.Builder(inBounds.literalType, inBounds.shape));
        body.addEdge(maskFloatOp, maskFloat).init(new OnnxEdge.Builder(maskFloat.literalType, maskFloat.shape));

        const maskIntOp = body
          .addNode(uniq(body, `mask_cast_i_op_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Cast", [inBounds], { to: DataType.INT64 }))
          .as(OperationNode);
        const maskInt = body
          .addNode(uniq(body, `mask_cast_i_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(inBounds, maskIntOp).init(new OnnxEdge.Builder(inBounds.literalType, inBounds.shape));
        body.addEdge(maskIntOp, maskInt).init(new OnnxEdge.Builder(maskInt.literalType, maskInt.shape));

        // sumVal += gathered * maskFloat
        const contribOp = body
          .addNode(uniq(body, `sum_mul_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Mul", [gathered, maskFloat]))
          .as(OperationNode);
        const contrib = body
          .addNode(uniq(body, `contrib_${kh}_${kwi}`))
          .init(new TensorNode.Builder(elemTy, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(gathered, contribOp).init(new OnnxEdge.Builder(gathered.literalType, gathered.shape));
        body.addEdge(maskFloat, contribOp).init(new OnnxEdge.Builder(maskFloat.literalType, maskFloat.shape));
        body.addEdge(contribOp, contrib).init(new OnnxEdge.Builder(contrib.literalType, contrib.shape));

        const addOp = body
          .addNode(uniq(body, `sum_add_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Add", [sumVal, contrib]))
          .as(OperationNode);
        const sumNext = body
          .addNode(uniq(body, `sum_next_${kh}_${kwi}`))
          .init(new TensorNode.Builder(elemTy, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(sumVal, addOp).init(new OnnxEdge.Builder(sumVal.literalType, sumVal.shape));
        body.addEdge(contrib, addOp).init(new OnnxEdge.Builder(contrib.literalType, contrib.shape));
        body.addEdge(addOp, sumNext).init(new OnnxEdge.Builder(sumNext.literalType, sumNext.shape));
        sumVal = sumNext;

        // countVal += maskInt (0 or 1)
        const countAddOp = body
          .addNode(uniq(body, `count_add_${kh}_${kwi}`))
          .init(new OperationNode.Builder("Add", [countVal, maskInt]))
          .as(OperationNode);
        const countNext = body
          .addNode(uniq(body, `count_next_${kh}_${kwi}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body.addEdge(countVal, countAddOp).init(new OnnxEdge.Builder(countVal.literalType, countVal.shape));
        body.addEdge(maskInt, countAddOp).init(new OnnxEdge.Builder(maskInt.literalType, maskInt.shape));
        body.addEdge(countAddOp, countNext).init(new OnnxEdge.Builder(countNext.literalType, countNext.shape));
        countVal = countNext;
      }
    }

    // Final avg = sumVal / divisor
    let divisor: TensorNode.Class;
    if (countIncludePad) {
      // divide by kH*kW
      divisor = makeTensorConst(
        body,
        "divisor_kSize",
        DataType.INT64,
        "constant",
        scalarInt64(kernelSize)
      );
    } else {
      // use computed countVal
      divisor = countVal;
    }

    const divisorCast = body
      .addNode(uniq(body, "divisor_cast"))
      .init(new OperationNode.Builder("Cast", [divisor], { to: elemTy }))
      .as(OperationNode);
    const divisorCastTensor = body
      .addNode(uniq(body, "divisor_cast_t"))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(divisor, divisorCast)
      .init(new OnnxEdge.Builder(divisor.literalType, divisor.shape));
    body
      .addEdge(divisorCast, divisorCastTensor)
      .init(new OnnxEdge.Builder(divisorCastTensor.literalType, divisorCastTensor.shape));

    const avgOp = body
      .addNode(uniq(body, "avg_op"))
      .init(new OperationNode.Builder("Div", [sumVal, divisorCastTensor]))
      .as(OperationNode);
    const avgOut = body
      .addNode(uniq(body, "avg_out"))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(sumVal, avgOp)
      .init(new OnnxEdge.Builder(sumVal.literalType, sumVal.shape));
    body
      .addEdge(divisorCastTensor, avgOp)
      .init(new OnnxEdge.Builder(divisorCastTensor.literalType, divisorCastTensor.shape));
    body
      .addEdge(avgOp, avgOut)
      .init(new OnnxEdge.Builder(avgOut.literalType, avgOut.shape));

    // Tell BuildLoop what the update and indices are
    const indicesOut = iterUnsq;
    const lastOut = avgOut;

    // ctx: align with ConvBuilder
    const ctx: LoopCtx = {
      opMap: new Map(),
      iter,
      unsqIdx: iterUnsq,
      carry,
      axes: axesTensor,
      outShape: yShape,
      coalesce: false,
    };

    // ---- Outer: trip_count, cond, v_initial ------------------------------
    inferShapes(outer);
    const trip = makeTensorConst(
      outer,
      `avg_trip_${avg.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(totalIters)),
    );
    const cond = makeTensorConst(
      outer,
      `avg_cond_${avg.id}`,
      DataType.BOOL,
      "constant",
      bool(true),
    );
    const v_initial = makeTensorConst(
      outer,
      `avg_carry_init_${avg.id}`,
      elemTy,
      "constant",
      zeroTensor(elemTy, [carryLen]),
    );

    inferShapes(body);

    const outShape = yShape;
    const outTensor = Y;

    return {
      body,
      ctx,
      lastOut,
      indicesOut,
      elemTy,
      outShape,
      inputs,
      outTensor,
      trip,
      cond,
      v_initial,
    };
  }
}

export default AveragePoolBuilder;
