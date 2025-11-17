import Graph from "@specs-feup/flow/graph/Graph";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { uniq, makeTensorConst, int64Vec, scalarInt64, bool, zeroTensor } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopBuilder, BuildResult, unsqueezeIdx, LoopCtx } from "../BuildLoop.js";

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

    const xShape = X.shape as number[];
    const wShape = W.shape as number[];

    if (xShape.length !== 4 || wShape.length !== 4) {
      throw new Error(
        `ConvBuilder: only 2D Conv with NCHW layout is supported; got X=${xShape}, W=${wShape}`
      );
    }

    const [N, C, H, Win] = xShape;
    const [M, Cw, kH, kW] = wShape;

    // ---- Attribute sanity + current restrictions -----------------------------
    const a = conv.getAttributes?.() ?? conv.attributes ?? {};

    // Strides & dilations: normalise to [2] array of numbers
    let strides: number[] = Array.isArray(a.strides) ? a.strides.map(Number) : [1, 1];
    let dilations: number[] = Array.isArray(a.dilations) ? a.dilations.map(Number) : [1, 1];
    if (strides.length === 1) strides = [strides[0], strides[0]];
    if (dilations.length === 1) dilations = [dilations[0], dilations[0]];

    // Pads: either explicit pads or VALID→[0,0,0,0]
    const auto_pad = (a.auto_pad ?? "NOTSET") as string;
    let pads: number[] =
    Array.isArray(a.pads) ? a.pads.map(Number) :
    (auto_pad === "VALID" ? [0, 0, 0, 0] : [0, 0, 0, 0]); // for conv_simple

    const group = Number(a.group ?? 1);

    if (N !== 1 || C !== 1 || Cw !== 1 || M !== 1) {
      throw new Error(
        `ConvBuilder: currently only N=C=Cw=M=1 is supported; got X=${xShape}, W=${wShape}`
      );
    }
    if (group !== 1) {
      throw new Error(`ConvBuilder: only group=1 is supported, got group=${group}`);
    }
    if (
      strides[0] !== 1 ||
      strides[1] !== 1 ||
      dilations[0] !== 1 ||
      dilations[1] !== 1
    ) {
      throw new Error(
        `ConvBuilder: only strides=[1,1] and dilations=[1,1] are supported; got strides=${strides}, dilations=${dilations}`
      );
    }
    if (pads.some((p) => p !== 0)) {
      throw new Error(
        `ConvBuilder: only pads=[0,0,0,0] is supported; got pads=${pads}`
      );
    }

    const H_out = H - kH + 1;
    const W_out = Win - kW + 1;
    if (H_out <= 0 || W_out <= 0) {
      throw new Error(
        `ConvBuilder: invalid shapes, H_out=${H_out}, W_out=${W_out} (H=${H}, W=${Win}, kH=${kH}, kW=${kW})`
      );
    }

    // 1 output channel, so total outputs = H_out * W_out
    const numOut = H_out * W_out;
    const carryLen = numOut;

    const outShape =
      outTensor.shape && outTensor.shape.length
        ? outTensor.shape
        : [1, 1, H_out, W_out];

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

    // Flatten X to 1D [-1]
    const xFlatShapeConst = makeTensorConst(
      body,
      `conv_x_flat_shape_${conv.id}`,
      DataType.INT64,
      "constant",
      int64Vec([-1])
    );
    const xReshape = body
      .addNode(uniq(body, `conv_x_reshape_${conv.id}`))
      .init(new OperationNode.Builder("Reshape", [X_in, xFlatShapeConst]))
      .as(OperationNode);
    const X_flat = body
      .addNode(uniq(body, `conv_x_flat_${conv.id}`))
      .init(new TensorNode.Builder(X.literalType, [-1], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(X_in, xReshape)
      .init(new OnnxEdge.Builder(X_in.literalType, X_in.shape))
      .as(OnnxEdge);
    body
      .addEdge(xFlatShapeConst, xReshape)
      .init(
        new OnnxEdge.Builder(
          xFlatShapeConst.literalType,
          xFlatShapeConst.shape
        )
      )
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

    // Some scalar INT64 constants: W_out, W_in, kW
    const WoutConst = makeTensorConst(
    body,
    `conv_Wout_${conv.id}`,
    DataType.INT64,
    "constant",
    scalarInt64(Number(W_out))
    );
    const WinConst = makeTensorConst(
    body,
    `conv_Win_${conv.id}`,
    DataType.INT64,
    "constant",
    // Use the input width dimension Win (from X.shape), NOT the tensor W_in.
    scalarInt64(Number(Win))
    );
    const kWConst = makeTensorConst(
    body,
    `conv_kW_${conv.id}`,
    DataType.INT64,
    "constant",
    scalarInt64(Number(kW))
    );

    // Decode iter → (ho, wo)
    // ho = iter / W_out (integer division on INT64)
    const hoDivOp = body
      .addNode(uniq(body, `conv_div_ho_${conv.id}`))
      .init(new OperationNode.Builder("Div", [iter, WoutConst]))
      .as(OperationNode);
    const ho = body
      .addNode(uniq(body, `conv_ho_${conv.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(iter, hoDivOp)
      .init(new OnnxEdge.Builder(iter.literalType, iter.shape))
      .as(OnnxEdge);
    body
      .addEdge(WoutConst, hoDivOp)
      .init(new OnnxEdge.Builder(WoutConst.literalType, WoutConst.shape))
      .as(OnnxEdge);
    body
      .addEdge(hoDivOp, ho)
      .init(new OnnxEdge.Builder(ho.literalType, ho.shape))
      .as(OnnxEdge);

    // hoTimesWout = ho * W_out
    const hoMulOp = body
      .addNode(uniq(body, `conv_mul_hoW_${conv.id}`))
      .init(new OperationNode.Builder("Mul", [ho, WoutConst]))
      .as(OperationNode);
    const hoTimesWout = body
      .addNode(uniq(body, `conv_hoW_${conv.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(ho, hoMulOp)
      .init(new OnnxEdge.Builder(ho.literalType, ho.shape))
      .as(OnnxEdge);
    body
      .addEdge(WoutConst, hoMulOp)
      .init(new OnnxEdge.Builder(WoutConst.literalType, WoutConst.shape))
      .as(OnnxEdge);
    body
      .addEdge(hoMulOp, hoTimesWout)
      .init(new OnnxEdge.Builder(hoTimesWout.literalType, hoTimesWout.shape))
      .as(OnnxEdge);

    // wo = iter - hoTimesWout
    const woSubOp = body
      .addNode(uniq(body, `conv_sub_wo_${conv.id}`))
      .init(new OperationNode.Builder("Sub", [iter, hoTimesWout]))
      .as(OperationNode);
    const wo = body
      .addNode(uniq(body, `conv_wo_${conv.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    body
      .addEdge(iter, woSubOp)
      .init(new OnnxEdge.Builder(iter.literalType, iter.shape))
      .as(OnnxEdge);
    body
      .addEdge(hoTimesWout, woSubOp)
      .init(
        new OnnxEdge.Builder(hoTimesWout.literalType, hoTimesWout.shape)
      )
      .as(OnnxEdge);
    body
      .addEdge(woSubOp, wo)
      .init(new OnnxEdge.Builder(wo.literalType, wo.shape))
      .as(OnnxEdge);

    // ---- Accumulate over kernel window ---------------------------------------
    let accVec: TensorNode.Class | null = null;

    for (let kh = 0; kh < kH; kh++) {
      for (let kw = 0; kw < kW; kw++) {
        // Constants for kh, kw
        const khConst = makeTensorConst(
          body,
          `conv_kh_${conv.id}_${kh}_${kw}`,
          DataType.INT64,
          "constant",
          scalarInt64(Number(kh))
        );
        const kwConst = makeTensorConst(
          body,
          `conv_kw_${conv.id}_${kh}_${kw}`,
          DataType.INT64,
          "constant",
          scalarInt64(Number(kw))
        );

        // h = ho + kh
        const hAddOp = body
          .addNode(uniq(body, `conv_h_add_${conv.id}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Add", [ho, khConst]))
          .as(OperationNode);
        const h = body
          .addNode(uniq(body, `conv_h_${conv.id}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body
          .addEdge(ho, hAddOp)
          .init(new OnnxEdge.Builder(ho.literalType, ho.shape))
          .as(OnnxEdge);
        body
          .addEdge(khConst, hAddOp)
          .init(new OnnxEdge.Builder(khConst.literalType, khConst.shape))
          .as(OnnxEdge);
        body
          .addEdge(hAddOp, h)
          .init(new OnnxEdge.Builder(h.literalType, h.shape))
          .as(OnnxEdge);

        // w = wo + kw
        const wAddOp = body
          .addNode(uniq(body, `conv_w_add_${conv.id}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Add", [wo, kwConst]))
          .as(OperationNode);
        const w = body
          .addNode(uniq(body, `conv_w_${conv.id}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body
          .addEdge(wo, wAddOp)
          .init(new OnnxEdge.Builder(wo.literalType, wo.shape))
          .as(OnnxEdge);
        body
          .addEdge(kwConst, wAddOp)
          .init(new OnnxEdge.Builder(kwConst.literalType, kwConst.shape))
          .as(OnnxEdge);
        body
          .addEdge(wAddOp, w)
          .init(new OnnxEdge.Builder(w.literalType, w.shape))
          .as(OnnxEdge);

        // xIndex = h * W_in + w
        const hMulWinOp = body
          .addNode(uniq(body, `conv_hMulW_${conv.id}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Mul", [h, WinConst]))
          .as(OperationNode);
        const hMulWin = body
          .addNode(uniq(body, `conv_hMulW_out_${conv.id}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body
          .addEdge(h, hMulWinOp)
          .init(new OnnxEdge.Builder(h.literalType, h.shape))
          .as(OnnxEdge);
        body
          .addEdge(WinConst, hMulWinOp)
          .init(new OnnxEdge.Builder(WinConst.literalType, WinConst.shape))
          .as(OnnxEdge);
        body
          .addEdge(hMulWinOp, hMulWin)
          .init(new OnnxEdge.Builder(hMulWin.literalType, hMulWin.shape))
          .as(OnnxEdge);

        const xIdxAddOp = body
          .addNode(uniq(body, `conv_xIdx_add_${conv.id}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Add", [hMulWin, w]))
          .as(OperationNode);
        const xIdx = body
          .addNode(uniq(body, `conv_xIdx_${conv.id}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
          .as(TensorNode);
        body
          .addEdge(hMulWin, xIdxAddOp)
          .init(new OnnxEdge.Builder(hMulWin.literalType, hMulWin.shape))
          .as(OnnxEdge);
        body
          .addEdge(w, xIdxAddOp)
          .init(new OnnxEdge.Builder(w.literalType, w.shape))
          .as(OnnxEdge);
        body
          .addEdge(xIdxAddOp, xIdx)
          .init(new OnnxEdge.Builder(xIdx.literalType, xIdx.shape))
          .as(OnnxEdge);

        // unsqueeze x-index → [1]
        const xIdxUnsq = unsqueezeIdx(
          body,
          xIdx,
          axes0,
          `conv_xIdx_unsq_${conv.id}_${kh}_${kw}`
        ); // [1]

        // Gather X_flat[xIndex]
        const gatherXOp = body
          .addNode(uniq(body, `conv_gatherX_${conv.id}_${kh}_${kw}`))
          .init(
            new OperationNode.Builder("Gather", [X_flat, xIdxUnsq], {
              axis: 0,
            })
          )
          .as(OperationNode);
        const xVec = body
          .addNode(uniq(body, `conv_xVec_${conv.id}_${kh}_${kw}`))
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

        // W index is kh * kW + kw (pure constant)
        const flatK = kh * kW + kw;
        const wIdxConst = makeTensorConst(
          body,
          `conv_wIdx_const_${conv.id}_${kh}_${kw}`,
          DataType.INT64,
          "constant",
          scalarInt64(Number(flatK))
        );
        const wIdxUnsq = unsqueezeIdx(
          body,
          wIdxConst,
          axes0,
          `conv_wIdx_unsq_${conv.id}_${kh}_${kw}`
        ); // [1]

        const gatherWOp = body
          .addNode(uniq(body, `conv_gatherW_${conv.id}_${kh}_${kw}`))
          .init(
            new OperationNode.Builder("Gather", [W_flat, wIdxUnsq], {
              axis: 0,
            })
          )
          .as(OperationNode);
        const wVec = body
          .addNode(uniq(body, `conv_wVec_${conv.id}_${kh}_${kw}`))
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

        // term = X * W  (both [1])
        const mulOp = body
          .addNode(uniq(body, `conv_mul_${conv.id}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Mul", [xVec, wVec]))
          .as(OperationNode);
        const term = body
          .addNode(uniq(body, `conv_term_${conv.id}_${kh}_${kw}`))
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
            .addNode(uniq(body, `conv_acc_add_${conv.id}_${kh}_${kw}`))
            .init(new OperationNode.Builder("Add", [accVec, term]))
            .as(OperationNode);
          const accOut = body
            .addNode(uniq(body, `conv_acc_${conv.id}_${kh}_${kw}`))
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

    if (!accVec) {
      throw new Error("ConvBuilder: internal error, accVec not built");
    }

    // Add bias if present (B is [1])
    let yVec = accVec;
    if (B_in) {
      const addBiasOp = body
        .addNode(uniq(body, `conv_add_bias_${conv.id}`))
        .init(new OperationNode.Builder("Add", [accVec, B_in]))
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
        .addEdge(B_in, addBiasOp)
        .init(new OnnxEdge.Builder(B_in.literalType, B_in.shape))
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
