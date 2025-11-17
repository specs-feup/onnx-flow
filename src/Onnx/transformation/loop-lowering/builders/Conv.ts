import Graph from "@specs-feup/flow/graph/Graph";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { uniq, makeTensorConst, int64Vec, scalarInt64, bool, zeroTensor } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopBuilder, BuildResult, unsqueezeIdx, LoopCtx, decodeMixedRadix } from "../BuildLoop.js";

function resolveShape(t: TensorNode.Class): number[] {
  // If the tensor already has a shape, just use it.
  if (t.shape && t.shape.length) {
    return t.shape as number[];
  }

  // Try incoming edges first
  const incs = t.getIncomers ?? [];
  for (const e of incs) {
    if (e.shape && e.shape.length) {
      // Cache it on the tensor so later passes see it as well
      //t.shape = e.shape.slice();
      return t.shape as number[];
    }
  }

  // Fallback: try outgoing edges (sometimes only consumers got a shape)
  const outs = t.getOutgoers ?? [];
  for (const e of outs) {
    if (e.shape && e.shape.length) {
      //t.shape = e.shape.slice();
      return t.shape as number[];
    }
  }

  // Still unknown
  return [];
}


/**
 * Conv loop-lowering builder.
 *
 * Current restrictions (good enough for conv_simple + debugging):
 *  - 2D Conv, NCHW
 *  - N = 1, C = 1, M = 1
 *  - group = 1
 *  - strides = [1, 1]
 *  - pads = [0, 0, 0, 0]
 *  - dilations = [1, 1]
 *
 * For each loop iteration `iter` we compute exactly one output element Y[iter]
 * and scatter it into a 1D carry buffer of size H_out * W_out.
 */
export default class ConvBuilder implements LoopBuilder {
  canHandle(chain: OperationNode.Class[]): boolean {
    return chain.length === 1 && chain[0].type === "Conv";
  }

  build(
    chain: OperationNode.Class[],
    outer: OnnxGraph.Class,
    opts: { fuse: boolean; recurse: boolean; coalesce: boolean }
  ): BuildResult {
    const conv = chain[0];

    // ---- Basic tensor plumbing ------------------------------------------------
    const inputsArr =
      conv.getInputs()?.filter((n) => n.is(TensorNode)).map((n) => n.as(TensorNode)) ??
      [];
    if (inputsArr.length < 2) {
      throw new Error("ConvBuilder: Conv must have at least X and W as inputs");
    }

    const X = inputsArr[0];
    const W = inputsArr[1];
    const B = inputsArr.length >= 3 ? inputsArr[2] : undefined;

    const outTensor = conv.getOutgoers.targets.filterIs(TensorNode).first();

    const elemTy =
      outTensor.literalType === DataType.UNDEFINED
        ? conv.getOutgoers.first().literalType
        : outTensor.literalType;

    console.log("XSHAPE:", X.shape);
    const xShape = resolveShape(X);
    const wShape = resolveShape(W);

    const is2D = xShape.length === 4 && wShape.length === 4;
    const is1D = xShape.length === 3 && wShape.length === 3;

    if (!is1D && !is2D) {
      throw new Error(
        `ConvBuilder: only 1D or 2D Conv with NCW/NCHW layout is supported; got X=${xShape}, W=${wShape}`
      );
    }

    let N: number;
    let C: number;
    let H: number;
    let Win: number;
    let M: number;
    let Cw: number;
    let kH: number;
    let kW: number;

    if (is2D) {
      [N, C, H, Win] = xShape;
      [M, Cw, kH, kW] = wShape;
    } else {
      // 1D Conv: X [N, C, W], W [M, C/group, kW]
      const [N1, C1, W1] = xShape;
      const [M1, Cw1, kW1] = wShape;

      N = N1;
      C = C1;
      H = 1;          // fake spatial H dimension
      Win = W1;

      M = M1;
      Cw = Cw1;
      kH = 1;         // kernel height = 1
      kW = kW1;
    }

    // Bias shape: allow scalar [1] or per-output-channel [M]
    // Also accept common "expanded" forms like [1, M, 1, 1] or [M, 1, 1].
    const bShape = B ? resolveShape(B) : [];

    function classifyBiasShape(shape: number[]): "none" | "scalar" | "perChannel1D" | "perChannel4D" {
      if (!B) return "none";
      if (shape.length === 1 && shape[0] === 1) return "scalar";
      if (shape.length === 1 && shape[0] === M) return "perChannel1D";
      if (
        shape.length === 4 &&
        shape[0] === 1 &&
        shape[1] === M &&
        shape[2] === 1 &&
        shape[3] === 1
      ) {
        return "perChannel4D";
      }
      // You can easily add more forms here later (e.g., [M,1,1]) if needed.
      return "none";
    }

    const biasKind = B ? classifyBiasShape(bShape) : "none";

    if (B && biasKind === "none") {
      throw new Error(
        `ConvBuilder: unsupported bias shape; expected [1], [M] or [1,M,1,1]; got B=${bShape}, M=${M}`
      );
    }

    // ---- Attribute sanity + current restrictions -----------------------------
    const a = conv.getAttributes?.() ?? conv.attributes ?? {};

    // Strides & dilations
    let strides: number[] = Array.isArray(a.strides)
      ? a.strides.map(Number)
      : is1D
      ? [1]          // Conv1D default: [strideW]
      : [1, 1];

    let dilations: number[] = Array.isArray(a.dilations)
      ? a.dilations.map(Number)
      : is1D
      ? [1]          // Conv1D default: [dilW]
      : [1, 1];

    let strideH: number;
    let strideW: number;
    let dilH: number;
    let dilW: number;

    if (is1D) {
      // Only W dimension is “real”; H is the fake dimension = 1
      const sW = strides[0] ?? 1;
      const dW = dilations[0] ?? 1;

      strideH = 1;
      strideW = sW;
      dilH = 1;
      dilW = dW;

      // Keep these as 2D-style arrays for later logs/debug if needed
      strides = [strideH, strideW];
      dilations = [dilH, dilW];
    } else {
      // 2D Conv
      if (strides.length === 1) strides = [strides[0], strides[0]];
      if (dilations.length === 1) dilations = [dilations[0], dilations[0]];

      strideH = strides[0];
      strideW = strides[1];
      dilH = dilations[0];
      dilW = dilations[1];
    }

    // Effective kernel sizes (for SAME_* padding computation)
    const kEffH = dilH * (kH - 1) + 1;
    const kEffW = dilW * (kW - 1) + 1;

    const auto_pad = (a.auto_pad ?? "NOTSET") as string;

    // Helper to compute SAME_* pads for one spatial dimension
    function computeSamePads(
      inSize: number,
      effKernel: number,
      stride: number,
      isLower: boolean
    ): [number, number] {
      const out = Math.ceil(inSize / stride);
      const totalPad = Math.max((out - 1) * stride + effKernel - inSize, 0);
      // UPPER: more padding at the end; LOWER: more padding at the beginning
      const padHead = isLower ? Math.ceil(totalPad / 2) : Math.floor(totalPad / 2);
      const padTail = totalPad - padHead;
      return [padHead, padTail];
    }

    let pads: number[];

    if (Array.isArray(a.pads) && a.pads.length > 0) {
      if (is1D) {
        if (a.pads.length !== 2 && a.pads.length !== 4) {
          throw new Error(
            `ConvBuilder: Conv1D expects pads of length 2 or [0,pl,0,pr]; got pads=${a.pads}`
          );
        }
        if (a.pads.length === 2) {
          const [padLeft, padRight] = a.pads.map(Number);
          pads = [0, padLeft, 0, padRight];
        } else {
          pads = a.pads.map(Number);
        }
      } else {
        // 2D Conv
        if (a.pads.length !== 4) {
          throw new Error(
            `ConvBuilder: Conv2D expects pads of length 4; got pads=${a.pads}`
          );
        }
        pads = a.pads.map(Number);
      }
    } else if (auto_pad === "VALID" || auto_pad === "NOTSET" || !auto_pad) {
      pads = [0, 0, 0, 0];
    } else if (auto_pad === "SAME_UPPER" || auto_pad === "SAME_LOWER") {
      const isLower = auto_pad === "SAME_LOWER";

      let padTop = 0;
      let padBottom = 0;
      let padLeft = 0;
      let padRight = 0;

      if (!is1D) {
        const pb = computeSamePads(H, kEffH, strideH, isLower);
        padTop = pb[0];
        padBottom = pb[1];
      }

      const plr = computeSamePads(Win, kEffW, strideW, isLower);
      padLeft = plr[0];
      padRight = plr[1];

      pads = [padTop, padLeft, padBottom, padRight];
    } else {
      throw new Error(`ConvBuilder: unsupported auto_pad=${auto_pad}`);
    }

    const group = Number(a.group ?? 1);

    if (group < 1) {
      throw new Error(`ConvBuilder: invalid group=${group}`);
    }
    if (C % group !== 0) {
      throw new Error(`ConvBuilder: C=${C} not divisible by group=${group}`);
    }
    if (M % group !== 0) {
      throw new Error(`ConvBuilder: M=${M} not divisible by group=${group}`);
    }
    if (Cw * group !== C) {
      throw new Error(
        `ConvBuilder: expected W second dim = C/group; got X=${xShape}, W=${wShape}, group=${group}`
      );
    }

    // Pads: [top, left, bottom, right]
    const padTop = pads[0] ?? 0;
    const padLeft = pads[1] ?? 0;
    const padBottom = pads[2] ?? 0;
    const padRight = pads[3] ?? 0;

    const H_padded = H + padTop + padBottom;
    const W_padded = Win + padLeft + padRight;

    const H_out = Math.floor((H_padded - kEffH) / strideH + 1);
    const W_out = Math.floor((W_padded - kEffW) / strideW + 1);

    if (H_out <= 0 || W_out <= 0) {
      throw new Error(
        `ConvBuilder: invalid shapes, H_out=${H_out}, W_out=${W_out} (H=${H}, W=${Win}, kH=${kH}, kW=${kW}, pads=${pads}, strides=${strides}, dilations=${dilations})`
      );
    }

    const numOut = N * M * H_out * W_out;
    const carryLen = numOut;

    // Conv output shape: [N, M, H_out, W_out]
    const outShape = [N, M, H_out, W_out];

    // Make sure the graph tensor reflects this shape
    outTensor.setShape(outShape);

    // `inputs` map used by BuildLoop for captured outer inputs
    const inputs = new Map<string, TensorNode.Class>();
    inputs.set(X.id, X);
    inputs.set(W.id, W);
    if (B) inputs.set(B.id, B);

    // ---- Build loop body graph -----------------------------------------------
    const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    // iter: INT64 scalar
    const iter = body
      .addNode(uniq(body, "iter"))
      .init(new TensorNode.Builder(DataType.INT64, [], "input"))
      .as(TensorNode);

    // cond_in: BOOL scalar (ignored internally, just forwarded)
    const condIn = body
      .addNode(uniq(body, "cond_in"))
      .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
      .as(TensorNode);

    // carry: 1D buffer [carryLen] with same elemTy as Y
    const carry = body
      .addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [carryLen], "input"))
      .as(TensorNode);

    // Captured inputs (use same IDs as outer so BuildLoop can wire them)
    const X_in = body
    .addNode(X.id)
    .init(new TensorNode.Builder(X.literalType, X.shape, "intermediate"))
    .as(TensorNode);

    const W_in = body
    .addNode(W.id)
    .init(new TensorNode.Builder(W.literalType, W.shape, "intermediate"))
    .as(TensorNode);

    let B_in: TensorNode.Class | undefined;
    if (B) {
    B_in = body
        .addNode(B.id)
        .init(new TensorNode.Builder(B.literalType, B.shape, "intermediate"))
        .as(TensorNode);
    }

    // axes=[0] constant
    const axes0 = makeTensorConst(
      body,
      `conv_axes_${conv.id}`,
      DataType.INT64,
      "constant",
      int64Vec([0])
    );

    // Unsqueezed iteration index used as ScatterElements indices
    const iterUnsq = unsqueezeIdx(
      body,
      iter,
      axes0,
      `conv_unsq_iter_${conv.id}`
    ); // shape [1]

    // Optionally pad X_in spatially (H, W) before flattening
    let X_src = X_in;
    if (pads.some((p) => p !== 0)) {
      // ONNX Pad uses [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
      const padVec = [0, 0, padTop, padLeft, 0, 0, padBottom, padRight];

      const padsConst = makeTensorConst(
        body,
        `conv_pads_${conv.id}`,
        DataType.INT64,
        "constant",
        int64Vec(padVec)
      );
      const padOp = body
        .addNode(uniq(body, `conv_pad_${conv.id}`))
        .init(new OperationNode.Builder("Pad", [X_in, padsConst]))
        .as(OperationNode);
      const X_padded = body
        .addNode(uniq(body, `conv_x_padded_${conv.id}`))
        .init(
          new TensorNode.Builder(
            X.literalType,
            [N, C, H_padded, W_padded],
            "intermediate"
          )
        )
        .as(TensorNode);

      body
        .addEdge(X_in, padOp)
        .init(new OnnxEdge.Builder(X_in.literalType, X_in.shape))
        .as(OnnxEdge);
      body
        .addEdge(padsConst, padOp)
        .init(new OnnxEdge.Builder(padsConst.literalType, padsConst.shape))
        .as(OnnxEdge);
      body
        .addEdge(padOp, X_padded)
        .init(new OnnxEdge.Builder(X_padded.literalType, X_padded.shape))
        .as(OnnxEdge);

      X_src = X_padded;
    }

    // Flatten X_src to 1D [-1]
    const xFlatShapeConst = makeTensorConst(
      body,
      `conv_x_flat_shape_${conv.id}`,
      DataType.INT64,
      "constant",
      int64Vec([-1])
    );
    const xReshape = body
      .addNode(uniq(body, `conv_x_reshape_${conv.id}`))
      .init(new OperationNode.Builder("Reshape", [X_src, xFlatShapeConst]))
      .as(OperationNode);
    const X_flat = body
      .addNode(uniq(body, `conv_x_flat_${conv.id}`))
      .init(new TensorNode.Builder(X.literalType, [-1], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(X_src, xReshape)
      .init(new OnnxEdge.Builder(X_src.literalType, X_src.shape))
      .as(OnnxEdge);
    body
      .addEdge(xFlatShapeConst, xReshape)
      .init(new OnnxEdge.Builder(xFlatShapeConst.literalType, xFlatShapeConst.shape))
      .as(OnnxEdge);
    body
      .addEdge(xReshape, X_flat)
      .init(new OnnxEdge.Builder(X_flat.literalType, X_flat.shape))
      .as(OnnxEdge);

    // Flatten W to 1D [-1]
    const wFlatShapeConst = makeTensorConst(
      body,
      `conv_w_flat_shape_${conv.id}`,
      DataType.INT64,
      "constant",
      int64Vec([-1])
    );
    const wReshape = body
      .addNode(uniq(body, `conv_w_reshape_${conv.id}`))
      .init(new OperationNode.Builder("Reshape", [W_in, wFlatShapeConst]))
      .as(OperationNode);
    const W_flat = body
      .addNode(uniq(body, `conv_w_flat_${conv.id}`))
      .init(new TensorNode.Builder(W.literalType, [-1], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(W_in, wReshape)
      .init(new OnnxEdge.Builder(W_in.literalType, W_in.shape))
      .as(OnnxEdge);
    body
      .addEdge(wFlatShapeConst, wReshape)
      .init(
        new OnnxEdge.Builder(
          wFlatShapeConst.literalType,
          wFlatShapeConst.shape
        )
      )
      .as(OnnxEdge);
    body
      .addEdge(wReshape, W_flat)
      .init(new OnnxEdge.Builder(W_flat.literalType, W_flat.shape))
      .as(OnnxEdge);

    // Some scalar INT64 constants: W_out, W_padded, kW, etc.
    const WoutConst = makeTensorConst(
      body,
      `conv_Wout_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(W_out))
    );
    const WpadConst = makeTensorConst(
      body,
      `conv_Wpad_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(W_padded))
    );
    const HpadConst = makeTensorConst(
      body,
      `conv_Hpad_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(H_padded))
    );
    const CConst = makeTensorConst(
      body,
      `conv_C_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(C))
    );
    const kWConst = makeTensorConst(
      body,
      `conv_kW_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(kW))
    );
    const kHConst = makeTensorConst(
      body,
      `conv_kH_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(kH))
    );
    const strideHConst = makeTensorConst(
      body,
      `conv_strideH_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(strideH))
    );
    const strideWConst = makeTensorConst(
      body,
      `conv_strideW_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(strideW))
    );
    const dilHConst = makeTensorConst(
      body,
      `conv_dilH_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(dilH))
    );
    const dilWConst = makeTensorConst(
      body,
      `conv_dilW_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(dilW))
    );

    // Decode iter → (n, m, ho, wo) in row-major order over [N, M, H_out, W_out]
    const [nIdx, mIdx, ho, wo] = decodeMixedRadix(
      body,
      iter,
      [N, M, H_out, W_out],
      `conv_iter_${conv.id}`
    );

    // Group bookkeeping
    const C_per_group = Cw;                 // = C / group
    const M_per_group = M / group;
    const CwConst = makeTensorConst(
      body,
      `conv_Cw_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(C_per_group))
    );
    const MperGroupConst = makeTensorConst(
      body,
      `conv_MperG_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(M_per_group))
    );

    // gIdx = mIdx / (M/group)
    const gDivOp = body
      .addNode(uniq(body, `conv_div_g_${conv.id}`))
      .init(new OperationNode.Builder("Div", [mIdx, MperGroupConst]))
      .as(OperationNode);
    const gIdx = body
      .addNode(uniq(body, `conv_g_${conv.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(mIdx, gDivOp)
      .init(new OnnxEdge.Builder(mIdx.literalType, mIdx.shape))
      .as(OnnxEdge);
    body
      .addEdge(MperGroupConst, gDivOp)
      .init(new OnnxEdge.Builder(MperGroupConst.literalType, MperGroupConst.shape))
      .as(OnnxEdge);
    body
      .addEdge(gDivOp, gIdx)
      .init(new OnnxEdge.Builder(gIdx.literalType, gIdx.shape))
      .as(OnnxEdge);

    // gBase = gIdx * C_per_group  (start input channel for this group)
    const gMulCwOp = body
      .addNode(uniq(body, `conv_mul_gCw_${conv.id}`))
      .init(new OperationNode.Builder("Mul", [gIdx, CwConst]))
      .as(OperationNode);
    const gBase = body
      .addNode(uniq(body, `conv_gBase_${conv.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(gIdx, gMulCwOp)
      .init(new OnnxEdge.Builder(gIdx.literalType, gIdx.shape))
      .as(OnnxEdge);
    body
      .addEdge(CwConst, gMulCwOp)
      .init(new OnnxEdge.Builder(CwConst.literalType, CwConst.shape))
      .as(OnnxEdge);
    body
      .addEdge(gMulCwOp, gBase)
      .init(new OnnxEdge.Builder(gBase.literalType, gBase.shape))
      .as(OnnxEdge);

    // ---- Accumulate over input channels and kernel window --------------------
    let accVec: TensorNode.Class | null = null;

    for (let cRel = 0; cRel < C_per_group; cRel++) {
      const cRelConst = makeTensorConst(
        body,
        `conv_cRel_${conv.id}_${cRel}`,
        DataType.INT64,
        "constant",
        scalarInt64(Number(cRel))
      );

      // cAbs = gBase + cRel
      const cAbsAddOp = body
        .addNode(uniq(body, `conv_cAbs_add_${conv.id}_${cRel}`))
        .init(new OperationNode.Builder("Add", [gBase, cRelConst]))
        .as(OperationNode);
      const cAbs = body
        .addNode(uniq(body, `conv_cAbs_${conv.id}_${cRel}`))
        .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
        .as(TensorNode);
      body
        .addEdge(gBase, cAbsAddOp)
        .init(new OnnxEdge.Builder(gBase.literalType, gBase.shape))
        .as(OnnxEdge);
      body
        .addEdge(cRelConst, cAbsAddOp)
        .init(new OnnxEdge.Builder(cRelConst.literalType, cRelConst.shape))
        .as(OnnxEdge);
      body
        .addEdge(cAbsAddOp, cAbs)
        .init(new OnnxEdge.Builder(cAbs.literalType, cAbs.shape))
        .as(OnnxEdge);

      for (let kh = 0; kh < kH; kh++) {
        for (let kw = 0; kw < kW; kw++) {
          const khConst = makeTensorConst(
            body,
            `conv_kh_${conv.id}_${cRel}_${kh}_${kw}`,
            DataType.INT64,
            "constant",
            scalarInt64(Number(kh))
          );
          const kwConst = makeTensorConst(
            body,
            `conv_kw_${conv.id}_${cRel}_${kh}_${kw}`,
            DataType.INT64,
            "constant",
            scalarInt64(Number(kw))
          );

          // hPad = ho * strideH + kh * dilH
          const hoMulStrOp = body
            .addNode(uniq(body, `conv_hoMulStr_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [ho, strideHConst]))
            .as(OperationNode);
          const hoMulStr = body
            .addNode(uniq(body, `conv_hoMulStr_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(ho, hoMulStrOp)
            .init(new OnnxEdge.Builder(ho.literalType, ho.shape))
            .as(OnnxEdge);
          body
            .addEdge(strideHConst, hoMulStrOp)
            .init(new OnnxEdge.Builder(strideHConst.literalType, strideHConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(hoMulStrOp, hoMulStr)
            .init(new OnnxEdge.Builder(hoMulStr.literalType, hoMulStr.shape))
            .as(OnnxEdge);

          const khMulDilOp = body
            .addNode(uniq(body, `conv_khMulDil_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [khConst, dilHConst]))
            .as(OperationNode);
          const khMulDil = body
            .addNode(uniq(body, `conv_khMulDil_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(khConst, khMulDilOp)
            .init(new OnnxEdge.Builder(khConst.literalType, khConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(dilHConst, khMulDilOp)
            .init(new OnnxEdge.Builder(dilHConst.literalType, dilHConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(khMulDilOp, khMulDil)
            .init(new OnnxEdge.Builder(khMulDil.literalType, khMulDil.shape))
            .as(OnnxEdge);

          const hPadAddOp = body
            .addNode(uniq(body, `conv_hPad_add_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [hoMulStr, khMulDil]))
            .as(OperationNode);
          const hPad = body
            .addNode(uniq(body, `conv_hPad_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(hoMulStr, hPadAddOp)
            .init(new OnnxEdge.Builder(hoMulStr.literalType, hoMulStr.shape))
            .as(OnnxEdge);
          body
            .addEdge(khMulDil, hPadAddOp)
            .init(new OnnxEdge.Builder(khMulDil.literalType, khMulDil.shape))
            .as(OnnxEdge);
          body
            .addEdge(hPadAddOp, hPad)
            .init(new OnnxEdge.Builder(hPad.literalType, hPad.shape))
            .as(OnnxEdge);

          // wPad = wo * strideW + kw * dilW
          const woMulStrOp = body
            .addNode(uniq(body, `conv_woMulStr_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [wo, strideWConst]))
            .as(OperationNode);
          const woMulStr = body
            .addNode(uniq(body, `conv_woMulStr_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(wo, woMulStrOp)
            .init(new OnnxEdge.Builder(wo.literalType, wo.shape))
            .as(OnnxEdge);
          body
            .addEdge(strideWConst, woMulStrOp)
            .init(new OnnxEdge.Builder(strideWConst.literalType, strideWConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(woMulStrOp, woMulStr)
            .init(new OnnxEdge.Builder(woMulStr.literalType, woMulStr.shape))
            .as(OnnxEdge);

          const kwMulDilOp = body
            .addNode(uniq(body, `conv_kwMulDil_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [kwConst, dilWConst]))
            .as(OperationNode);
          const kwMulDil = body
            .addNode(uniq(body, `conv_kwMulDil_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(kwConst, kwMulDilOp)
            .init(new OnnxEdge.Builder(kwConst.literalType, kwConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(dilWConst, kwMulDilOp)
            .init(new OnnxEdge.Builder(dilWConst.literalType, dilWConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(kwMulDilOp, kwMulDil)
            .init(new OnnxEdge.Builder(kwMulDil.literalType, kwMulDil.shape))
            .as(OnnxEdge);

          const wPadAddOp = body
            .addNode(uniq(body, `conv_wPad_add_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [woMulStr, kwMulDil]))
            .as(OperationNode);
          const wPad = body
            .addNode(uniq(body, `conv_wPad_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(woMulStr, wPadAddOp)
            .init(new OnnxEdge.Builder(woMulStr.literalType, woMulStr.shape))
            .as(OnnxEdge);
          body
            .addEdge(kwMulDil, wPadAddOp)
            .init(new OnnxEdge.Builder(kwMulDil.literalType, kwMulDil.shape))
            .as(OnnxEdge);
          body
            .addEdge(wPadAddOp, wPad)
            .init(new OnnxEdge.Builder(wPad.literalType, wPad.shape))
            .as(OnnxEdge);

          // xIndex = ((n * C + cAbs) * H_padded + hPad) * W_padded + wPad
          const nMulCOp = body
            .addNode(uniq(body, `conv_nMulC_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [nIdx, CConst]))
            .as(OperationNode);
          const nMulC = body
            .addNode(uniq(body, `conv_nMulC_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(nIdx, nMulCOp)
            .init(new OnnxEdge.Builder(nIdx.literalType, nIdx.shape))
            .as(OnnxEdge);
          body
            .addEdge(CConst, nMulCOp)
            .init(new OnnxEdge.Builder(CConst.literalType, CConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(nMulCOp, nMulC)
            .init(new OnnxEdge.Builder(nMulC.literalType, nMulC.shape))
            .as(OnnxEdge);

          const ncPlusOp = body
            .addNode(uniq(body, `conv_ncPlus_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [nMulC, cAbs]))
            .as(OperationNode);
          const ncPlus = body
            .addNode(uniq(body, `conv_ncPlus_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(nMulC, ncPlusOp)
            .init(new OnnxEdge.Builder(nMulC.literalType, nMulC.shape))
            .as(OnnxEdge);
          body
            .addEdge(cAbs, ncPlusOp)
            .init(new OnnxEdge.Builder(cAbs.literalType, cAbs.shape))
            .as(OnnxEdge);
          body
            .addEdge(ncPlusOp, ncPlus)
            .init(new OnnxEdge.Builder(ncPlus.literalType, ncPlus.shape))
            .as(OnnxEdge);

          const mulHpadOp = body
            .addNode(uniq(body, `conv_mulHpad_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [ncPlus, HpadConst]))
            .as(OperationNode);
          const mulHpad = body
            .addNode(uniq(body, `conv_mulHpad_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(ncPlus, mulHpadOp)
            .init(new OnnxEdge.Builder(ncPlus.literalType, ncPlus.shape))
            .as(OnnxEdge);
          body
            .addEdge(HpadConst, mulHpadOp)
            .init(new OnnxEdge.Builder(HpadConst.literalType, HpadConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(mulHpadOp, mulHpad)
            .init(new OnnxEdge.Builder(mulHpad.literalType, mulHpad.shape))
            .as(OnnxEdge);

          const addHpadOp = body
            .addNode(uniq(body, `conv_addHpad_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [mulHpad, hPad]))
            .as(OperationNode);
          const addHpad = body
            .addNode(uniq(body, `conv_addHpad_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mulHpad, addHpadOp)
            .init(new OnnxEdge.Builder(mulHpad.literalType, mulHpad.shape))
            .as(OnnxEdge);
          body
            .addEdge(hPad, addHpadOp)
            .init(new OnnxEdge.Builder(hPad.literalType, hPad.shape))
            .as(OnnxEdge);
          body
            .addEdge(addHpadOp, addHpad)
            .init(new OnnxEdge.Builder(addHpad.literalType, addHpad.shape))
            .as(OnnxEdge);

          const mulWpadOp = body
            .addNode(uniq(body, `conv_mulWpad_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [addHpad, WpadConst]))
            .as(OperationNode);
          const mulWpad = body
            .addNode(uniq(body, `conv_mulWpad_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(addHpad, mulWpadOp)
            .init(new OnnxEdge.Builder(addHpad.literalType, addHpad.shape))
            .as(OnnxEdge);
          body
            .addEdge(WpadConst, mulWpadOp)
            .init(new OnnxEdge.Builder(WpadConst.literalType, WpadConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(mulWpadOp, mulWpad)
            .init(new OnnxEdge.Builder(mulWpad.literalType, mulWpad.shape))
            .as(OnnxEdge);

          const xIdxAddOp = body
            .addNode(uniq(body, `conv_xIdx_add_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [mulWpad, wPad]))
            .as(OperationNode);
          const xIdx = body
            .addNode(uniq(body, `conv_xIdx_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mulWpad, xIdxAddOp)
            .init(new OnnxEdge.Builder(mulWpad.literalType, mulWpad.shape))
            .as(OnnxEdge);
          body
            .addEdge(wPad, xIdxAddOp)
            .init(new OnnxEdge.Builder(wPad.literalType, wPad.shape))
            .as(OnnxEdge);
          body
            .addEdge(xIdxAddOp, xIdx)
            .init(new OnnxEdge.Builder(xIdx.literalType, xIdx.shape))
            .as(OnnxEdge);

          const xIdxUnsq = unsqueezeIdx(
            body,
            xIdx,
            axes0,
            `conv_xIdx_unsq_${conv.id}_${cRel}_${kh}_${kw}`
          );

          // Gather scalar X
          const gatherXOp = body
            .addNode(uniq(body, `conv_gatherX_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Gather", [X_flat, xIdxUnsq], { axis: 0 }))
            .as(OperationNode);
          const xVec = body
            .addNode(uniq(body, `conv_xVec_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(X_flat, gatherXOp)
            .init(new OnnxEdge.Builder(X_flat.literalType, X_flat.shape))
            .as(OnnxEdge);
          body
            .addEdge(xIdxUnsq, gatherXOp)
            .init(new OnnxEdge.Builder(xIdxUnsq.literalType, xIdxUnsq.shape))
            .as(OnnxEdge);
          body
            .addEdge(gatherXOp, xVec)
            .init(new OnnxEdge.Builder(xVec.literalType, xVec.shape))
            .as(OnnxEdge);

          // W index: flatK = ((mIdx * C_per_group + cRel) * kH + kh) * kW + kw
          const mMulCwOp = body
            .addNode(uniq(body, `conv_mMulCw_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [mIdx, CwConst]))
            .as(OperationNode);
          const mMulCw = body
            .addNode(uniq(body, `conv_mMulCw_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mIdx, mMulCwOp)
            .init(new OnnxEdge.Builder(mIdx.literalType, mIdx.shape))
            .as(OnnxEdge);
          body
            .addEdge(CwConst, mMulCwOp)
            .init(new OnnxEdge.Builder(CwConst.literalType, CwConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(mMulCwOp, mMulCw)
            .init(new OnnxEdge.Builder(mMulCw.literalType, mMulCw.shape))
            .as(OnnxEdge);

          const mPlusCrelOp = body
            .addNode(uniq(body, `conv_mPlusCrel_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [mMulCw, cRelConst]))
            .as(OperationNode);
          const mPlusCrel = body
            .addNode(uniq(body, `conv_mPlusCrel_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mMulCw, mPlusCrelOp)
            .init(new OnnxEdge.Builder(mMulCw.literalType, mMulCw.shape))
            .as(OnnxEdge);
          body
            .addEdge(cRelConst, mPlusCrelOp)
            .init(new OnnxEdge.Builder(cRelConst.literalType, cRelConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(mPlusCrelOp, mPlusCrel)
            .init(new OnnxEdge.Builder(mPlusCrel.literalType, mPlusCrel.shape))
            .as(OnnxEdge);

          const mulkHOp = body
            .addNode(uniq(body, `conv_mulkH_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [mPlusCrel, kHConst]))
            .as(OperationNode);
          const mulkH = body
            .addNode(uniq(body, `conv_mulkH_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mPlusCrel, mulkHOp)
            .init(new OnnxEdge.Builder(mPlusCrel.literalType, mPlusCrel.shape))
            .as(OnnxEdge);
          body
            .addEdge(kHConst, mulkHOp)
            .init(new OnnxEdge.Builder(kHConst.literalType, kHConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(mulkHOp, mulkH)
            .init(new OnnxEdge.Builder(mulkH.literalType, mulkH.shape))
            .as(OnnxEdge);

          const addKhOp = body
            .addNode(uniq(body, `conv_addKh_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [mulkH, khConst]))
            .as(OperationNode);
          const addKh = body
            .addNode(uniq(body, `conv_addKh_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mulkH, addKhOp)
            .init(new OnnxEdge.Builder(mulkH.literalType, mulkH.shape))
            .as(OnnxEdge);
          body
            .addEdge(khConst, addKhOp)
            .init(new OnnxEdge.Builder(khConst.literalType, khConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(addKhOp, addKh)
            .init(new OnnxEdge.Builder(addKh.literalType, addKh.shape))
            .as(OnnxEdge);

          const mulkWOp = body
            .addNode(uniq(body, `conv_mulkW_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [addKh, kWConst]))
            .as(OperationNode);
          const mulkW = body
            .addNode(uniq(body, `conv_mulkW_out_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(addKh, mulkWOp)
            .init(new OnnxEdge.Builder(addKh.literalType, addKh.shape))
            .as(OnnxEdge);
          body
            .addEdge(kWConst, mulkWOp)
            .init(new OnnxEdge.Builder(kWConst.literalType, kWConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(mulkWOp, mulkW)
            .init(new OnnxEdge.Builder(mulkW.literalType, mulkW.shape))
            .as(OnnxEdge);

          const wIdxAddOp = body
            .addNode(uniq(body, `conv_wIdx_add_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [mulkW, kwConst]))
            .as(OperationNode);
          const wIdx = body
            .addNode(uniq(body, `conv_wIdx_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(mulkW, wIdxAddOp)
            .init(new OnnxEdge.Builder(mulkW.literalType, mulkW.shape))
            .as(OnnxEdge);
          body
            .addEdge(kwConst, wIdxAddOp)
            .init(new OnnxEdge.Builder(kwConst.literalType, kwConst.shape))
            .as(OnnxEdge);
          body
            .addEdge(wIdxAddOp, wIdx)
            .init(new OnnxEdge.Builder(wIdx.literalType, wIdx.shape))
            .as(OnnxEdge);

          const wIdxUnsq = unsqueezeIdx(
            body,
            wIdx,
            axes0,
            `conv_wIdx_unsq_${conv.id}_${cRel}_${kh}_${kw}`
          );

          // Gather scalar W
          const gatherWOp = body
            .addNode(uniq(body, `conv_gatherW_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Gather", [W_flat, wIdxUnsq], { axis: 0 }))
            .as(OperationNode);
          const wVec = body
            .addNode(uniq(body, `conv_wVec_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(W_flat, gatherWOp)
            .init(new OnnxEdge.Builder(W_flat.literalType, W_flat.shape))
            .as(OnnxEdge);
          body
            .addEdge(wIdxUnsq, gatherWOp)
            .init(new OnnxEdge.Builder(wIdxUnsq.literalType, wIdxUnsq.shape))
            .as(OnnxEdge);
          body
            .addEdge(gatherWOp, wVec)
            .init(new OnnxEdge.Builder(wVec.literalType, wVec.shape))
            .as(OnnxEdge);

          // term = X * W
          const mulOp = body
            .addNode(uniq(body, `conv_term_mul_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Mul", [xVec, wVec]))
            .as(OperationNode);
          const term = body
            .addNode(uniq(body, `conv_term_${conv.id}_${cRel}_${kh}_${kw}`))
            .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
            .as(TensorNode);
          body
            .addEdge(xVec, mulOp)
            .init(new OnnxEdge.Builder(xVec.literalType, xVec.shape))
            .as(OnnxEdge);
          body
            .addEdge(wVec, mulOp)
            .init(new OnnxEdge.Builder(wVec.literalType, wVec.shape))
            .as(OnnxEdge);
          body
            .addEdge(mulOp, term)
            .init(new OnnxEdge.Builder(term.literalType, term.shape))
            .as(OnnxEdge);

          if (!accVec) {
            accVec = term;
          } else {
            const addOp = body
              .addNode(uniq(body, `conv_acc_add_${conv.id}_${cRel}_${kh}_${kw}`))
              .init(new OperationNode.Builder("Add", [accVec, term]))
              .as(OperationNode);
            const accOut = body
              .addNode(uniq(body, `conv_acc_${conv.id}_${cRel}_${kh}_${kw}`))
              .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
              .as(TensorNode);
            body
              .addEdge(accVec, addOp)
              .init(new OnnxEdge.Builder(accVec.literalType, accVec.shape))
              .as(OnnxEdge);
            body
              .addEdge(term, addOp)
              .init(new OnnxEdge.Builder(term.literalType, term.shape))
              .as(OnnxEdge);
            body
              .addEdge(addOp, accOut)
              .init(new OnnxEdge.Builder(accOut.literalType, accOut.shape))
              .as(OnnxEdge);
            accVec = accOut;
          }
        }
      }
    }

    if (!accVec) {
      throw new Error("ConvBuilder: internal error, accVec not built");
    }

    // Add bias if present.
    //  - "scalar": B is [1], just Add(accVec, B_in)
    //  - "perChannel1D": B is [M], gather B[mIdx]
    //  - "perChannel4D": B is [1,M,1,1], gather along axis=1 and squeeze.
    let yVec = accVec;
    if (B_in && biasKind !== "none") {
      let biasScalar: TensorNode.Class;

      if (biasKind === "scalar") {
        // B is effectively a scalar.
        biasScalar = B_in;
      } else {
        // We need to index with mIdx (output channel)
        const mIdxUnsq = unsqueezeIdx(
          body,
          mIdx,
          axes0,
          `conv_mIdx_unsq_${conv.id}`
        );

        const axisForB = biasKind === "perChannel1D" ? 0 : 1;

        const gatherBOp = body
          .addNode(uniq(body, `conv_gatherB_${conv.id}`))
          .init(
            new OperationNode.Builder("Gather", [B_in, mIdxUnsq], { axis: axisForB })
          )
          .as(OperationNode);

        // Result still has a singleton dimension; normalise to shape [1]
        const bVec = body
          .addNode(uniq(body, `conv_bVec_${conv.id}`))
          .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
          .as(TensorNode);

        body
          .addEdge(B_in, gatherBOp)
          .init(new OnnxEdge.Builder(B_in.literalType, B_in.shape))
          .as(OnnxEdge);
        body
          .addEdge(mIdxUnsq, gatherBOp)
          .init(new OnnxEdge.Builder(mIdxUnsq.literalType, mIdxUnsq.shape))
          .as(OnnxEdge);
        body
          .addEdge(gatherBOp, bVec)
          .init(new OnnxEdge.Builder(bVec.literalType, bVec.shape))
          .as(OnnxEdge);

        biasScalar = bVec;
      }

      const addBiasOp = body
        .addNode(uniq(body, `conv_add_bias_${conv.id}`))
        .init(new OperationNode.Builder("Add", [accVec, biasScalar]))
        .as(OperationNode);

      const yBias = body
        .addNode(uniq(body, `conv_y_bias_${conv.id}`))
        .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
        .as(TensorNode);

      body
        .addEdge(accVec, addBiasOp)
        .init(new OnnxEdge.Builder(accVec.literalType, accVec.shape))
        .as(OnnxEdge);
      body
        .addEdge(biasScalar, addBiasOp)
        .init(new OnnxEdge.Builder(biasScalar.literalType, biasScalar.shape))
        .as(OnnxEdge);
      body
        .addEdge(addBiasOp, yBias)
        .init(new OnnxEdge.Builder(yBias.literalType, yBias.shape))
        .as(OnnxEdge);

      yVec = yBias;
    }

    const lastOut = yVec; // [1]
    const indicesOut = iterUnsq; // [1]

    const ctx: LoopCtx = {
      opMap: new Map(),
      iter,
      unsqIdx: iterUnsq,
      carry,
      axes: axes0,
      outShape,
      coalesce: false,
    };

    // ---- Outer: trip_count, cond, v_initial ------------------------------
    inferShapes(outer); // align with other builders
    const trip = makeTensorConst(
      outer,
      `conv_trip_${conv.id}`,
      DataType.INT64,
      "constant",
      scalarInt64(Number(numOut))
    );
    const cond = makeTensorConst(
      outer,
      `conv_cond_${conv.id}`,
      DataType.BOOL,
      "constant",
      bool(true)
    );
    const v_initial = makeTensorConst(
      outer,
      `conv_init_${conv.id}`,
      elemTy,
      "constant",
      zeroTensor(elemTy, [carryLen])
    );

    inferShapes(body);

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
