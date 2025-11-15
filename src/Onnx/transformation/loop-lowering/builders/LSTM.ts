// src/Onnx/transformation/loop-lowering/handlers/LSTM.ts
import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import {
  uniq, makeTensorConst, int64Vec, scalarInt64, bool, toStaticShape, Shape,
} from "../../../Utils.js";

/**
 * Lower ONNX LSTM -> Loop over time with carried H, C.
 *
 * Supported & resilient:
 *  - direction: "forward" and "bidirectional"
 *  - default or explicit activations (Sigmoid/Tanh/Tanh). Unknowns → bail (leave op).
 *  - optional inputs: B, sequence_lens, initial_h, initial_c; missing ones are zeroed.
 *  - sequence_lens: if present, we still run T steps but mask updates for batch items
 *    that finished earlier (standard padded behavior).
 *
 * Not (yet) supported:
 *  - peepholes (peephole weights input absent in ONNX basic LSTM)
 *  - input_forget coupling
 *  - clip
 *  - multiple layers in a single op (use stacked LSTMs in graph)
 *
 * Design:
 *  - We replace each LSTM with either:
 *      * one Loop (forward), or
 *      * two Loops + Concat on dir axis=1 (bidirectional)
 *  - Y_h/Y_c are built from the carried finals (unsqueeze axis 0).
 *  - Y is the scan of H_t with an unsqueezed direction dim at axis=1.
 *  - Opset-13 style: Split sizes as input tensor; Unsqueeze axes as input tensor.
 */
export default function lowerLSTM(g: OnnxGraph.Class): void {
  const ops = g.getOperationNodes().filter(o => o.type === "LSTM");
  if (!ops.length) return;

  for (const op of ops) {
    const inps = op.getInputs() ?? [];
    if (inps.length < 3) continue; // need at least X,W,R

    const X  = inps[0]?.as(TensorNode);
    const W  = inps[1]?.as(TensorNode);
    const R  = inps[2]?.as(TensorNode);
    let  B   = inps[3]?.as(TensorNode) ?? null;
    const seqLens = inps[4]?.as(TensorNode) ?? null;
    let  H0  = inps[5]?.as(TensorNode) ?? null;
    let  C0  = inps[6]?.as(TensorNode) ?? null;

    if (!X || !W || !R) continue;

    const attrs = (op.getAttributes?.() ?? (op as any).attributes ?? {}) as Record<string, unknown>;
    const hiddenSize = Number(attrs.hidden_size ?? NaN);
    if (!Number.isFinite(hiddenSize)) continue;

    const direction = String(attrs.direction ?? "forward");
    const activations = normalizeActivations(attrs.activations);
    if (!activations) continue; // unknown activation set, skip lowering

    // Shapes
    const xShape = toStaticShape(X.shape as Shape); // [T,N,I]
    if (xShape.length !== 3) continue;
    const [Tdim, Ndim, Idim] = xShape.map(d => (d > 0 ? d : -1));
    const H = hiddenSize;

    // ONNX shapes for num_directions = 1 or 2
    const dirCount = direction === "bidirectional" ? 2 : 1;
    const WshapeOK = (W.shape?.length === 3) && Number(W.shape[0]) === dirCount && Number(W.shape[1]) === 4*H;
    const RshapeOK = (R.shape?.length === 3) && Number(R.shape[0]) === dirCount && Number(R.shape[1]) === 4*H;
    if (!WshapeOK || !RshapeOK) continue;

    // Ensure optional inputs
    if (!B) {
      B = makeTensorConst(g, uniq(g, `B_zero_${op.id}`), DataType.FLOAT, "constant",
                          { dataType: DataType.FLOAT, dims: [dirCount, 8*H], floatData: new Array(dirCount*8*H).fill(0) });
    }
    if (!H0) {
      H0 = makeTensorConst(g, uniq(g, `H0_zero_${op.id}`), DataType.FLOAT, "constant",
                           { dataType: DataType.FLOAT, dims: [dirCount, Math.max(1, Ndim), H], floatData: new Array(dirCount*Math.max(1,Ndim)*H).fill(0) });
    }
    if (!C0) {
      C0 = makeTensorConst(g, uniq(g, `C0_zero_${op.id}`), DataType.FLOAT, "constant",
                           { dataType: DataType.FLOAT, dims: [dirCount, Math.max(1, Ndim), H], floatData: new Array(dirCount*Math.max(1,Ndim)*H).fill(0) });
    }

    // If bidirectional, split into forward/backward lanes and later concat axis=1
    if (dirCount === 2) {
      const [Yf, Yhf, Ycf] = lowerOneDir(g, op, "forward",  0, X, W, R, B!, H0!, C0!, seqLens, H, activations);
      const [Yb, Yhb, Ycb] = lowerOneDir(g, op, "reverse",  1, X, W, R, B!, H0!, C0!, seqLens, H, activations);

      // Concat along direction dim (axis=1) for Y
      const axis1 = makeTensorConst(g, uniq(g, `axes1_${op.id}`), DataType.INT64, "constant", int64Vec([1]));
      const catY = g.addNode(uniq(g, `catY_${op.id}`))
        .init(new OperationNode.Builder("Concat", [Yf, Yb], { axis: 1 }))
        .as(OperationNode);
      const Y = g.addNode(uniq(g, `Y_${op.id}`))
        .init(new TensorNode.Builder(DataType.FLOAT, [Tdim, 2, Ndim, H], "intermediate"))
        .as(TensorNode);
      g.addEdge(catY, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);

      // Concat Y_h / Y_c along *direction* axis=0 (their dir dim is axis 0)
      const catYh = g.addNode(uniq(g, `catYh_${op.id}`))
        .init(new OperationNode.Builder("Concat", [Yhf, Yhb], { axis: 0 }))
        .as(OperationNode);
      const Y_h = g.addNode(uniq(g, `Y_h_${op.id}`))
        .init(new TensorNode.Builder(DataType.FLOAT, [2, Ndim, H], "intermediate"))
        .as(TensorNode);
      g.addEdge(catYh, Y_h).init(new OnnxEdge.Builder(Y_h.literalType, Y_h.shape)).as(OnnxEdge);

      const catYc = g.addNode(uniq(g, `catYc_${op.id}`))
        .init(new OperationNode.Builder("Concat", [Ycf, Ycb], { axis: 0 }))
        .as(OperationNode);
      const Y_c = g.addNode(uniq(g, `Y_c_${op.id}`))
        .init(new TensorNode.Builder(DataType.FLOAT, [2, Ndim, H], "intermediate"))
        .as(TensorNode);
      g.addEdge(catYc, Y_c).init(new OnnxEdge.Builder(Y_c.literalType, Y_c.shape)).as(OnnxEdge);

      rewireAndRemove(g, op, [Y, Y_h, Y_c]);
      continue;
    }

    // Single direction (forward)
    const [Y, Y_h, Y_c] =
      lowerOneDir(g, op, "forward", 0, X, W, R, B!, H0!, C0!, seqLens, H, activations);

    rewireAndRemove(g, op, [Y, Y_h, Y_c]);
  }
}

/** Build one Loop lane (direction = "forward" | "reverse"). */
function lowerOneDir(
  g: OnnxGraph.Class,
  op: OperationNode.Class,
  direction: "forward" | "reverse",
  dirIndex: number,
  X: TensorNode.Class,
  W: TensorNode.Class,
  R: TensorNode.Class,
  B: TensorNode.Class,
  H0: TensorNode.Class,
  C0: TensorNode.Class,
  seqLens: TensorNode.Class | null,
  H: number,
  activations: { f: string; g: string; h: string }
): [TensorNode.Class, TensorNode.Class, TensorNode.Class] {

  // Shapes [T,N,I]
  const xShape = toStaticShape(X.shape as Shape); const Tdim = xShape[0]; const Ndim = xShape[1]; const Idim = xShape[2];

  // Slice W/R/B for this direction: first dim = dirIndex
  const starts3 = makeTensorConst(g, uniq(g, `W_s_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex, 0, 0]));
  const endsW   = makeTensorConst(g, uniq(g, `W_e_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex+1, 4*H, Idim]));
  const endsR   = makeTensorConst(g, uniq(g, `R_e_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex+1, 4*H, H]));
  const axes3   = makeTensorConst(g, uniq(g, `axes3_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([0,1,2]));
  const steps3  = makeTensorConst(g, uniq(g, `steps3_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([1,1,1]));

  const slW = g.addNode(uniq(g, `Slice_W_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Slice", [W, starts3, endsW, axes3, steps3]))
    .as(OperationNode);
  const W01 = g.addNode(uniq(g, `W01_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [4*H, Idim], "intermediate"))
    .as(TensorNode);
  g.addEdge(slW, W01).init(new OnnxEdge.Builder(W01.literalType, W01.shape)).as(OnnxEdge);

  const slR = g.addNode(uniq(g, `Slice_R_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Slice", [R, starts3, endsR, axes3, steps3]))
    .as(OperationNode);
  const R01 = g.addNode(uniq(g, `R01_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [4*H, H], "intermediate"))
    .as(TensorNode);
  g.addEdge(slR, R01).init(new OnnxEdge.Builder(R01.literalType, R01.shape)).as(OnnxEdge);

  // Transpose to [I,4H] and [H,4H]
  const trW = g.addNode(uniq(g, `TrW_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Transpose", [W01], { perm: [1,0] }))
    .as(OperationNode);
  const Wt = g.addNode(uniq(g, `Wt_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [Idim, 4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(trW, Wt).init(new OnnxEdge.Builder(Wt.literalType, Wt.shape)).as(OnnxEdge);

  const trR = g.addNode(uniq(g, `TrR_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Transpose", [R01], { perm: [1,0] }))
    .as(OperationNode);
  const Rt = g.addNode(uniq(g, `Rt_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [H, 4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(trR, Rt).init(new OnnxEdge.Builder(Rt.literalType, Rt.shape)).as(OnnxEdge);

  // Bsum = B[dir, :4H] + B[dir, 4H:]
  const starts2 = makeTensorConst(g, uniq(g, `B_s_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex, 0]));
  const axes2   = makeTensorConst(g, uniq(g, `axes2_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([0,1]));
  const steps2  = makeTensorConst(g, uniq(g, `steps2_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([1,1]));
  const endsB0  = makeTensorConst(g, uniq(g, `B_e0_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex+1, 4*H]));
  const endsB1  = makeTensorConst(g, uniq(g, `B_e1_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex+1, 8*H]));

  const slB0 = g.addNode(uniq(g, `Slice_B0_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Slice", [B, starts2, endsB0, axes2, steps2]))
    .as(OperationNode);
  const B0 = g.addNode(uniq(g, `B0_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [1, 4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(slB0, B0).init(new OnnxEdge.Builder(B0.literalType, B0.shape)).as(OnnxEdge);

  const slB1 = g.addNode(uniq(g, `Slice_B1_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Slice", [B, starts2, endsB1, axes2, steps2]))
    .as(OperationNode);
  const B1 = g.addNode(uniq(g, `B1_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [1, 4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(slB1, B1).init(new OnnxEdge.Builder(B1.literalType, B1.shape)).as(OnnxEdge);

  const ax0 = makeTensorConst(g, uniq(g, `axis0_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([0]));
  const sqB0 = g.addNode(uniq(g, `sqB0_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Squeeze", [B0, ax0]))
    .as(OperationNode);
  const B0sq = g.addNode(uniq(g, `B0sq_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(sqB0, B0sq).init(new OnnxEdge.Builder(B0sq.literalType, B0sq.shape)).as(OnnxEdge);

  const sqB1 = g.addNode(uniq(g, `sqB1_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Squeeze", [B1, ax0]))
    .as(OperationNode);
  const B1sq = g.addNode(uniq(g, `B1sq_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(sqB1, B1sq).init(new OnnxEdge.Builder(B1sq.literalType, B1sq.shape)).as(OnnxEdge);

  const addB = g.addNode(uniq(g, `Bsum_add_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Add", [B0sq, B1sq]))
    .as(OperationNode);
  const Bsum = g.addNode(uniq(g, `Bsum_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [4*H], "intermediate"))
    .as(TensorNode);
  g.addEdge(addB, Bsum).init(new OnnxEdge.Builder(Bsum.literalType, Bsum.shape)).as(OnnxEdge);

  // Initial H0/C0 slice & squeeze [1,N,H] -> [N,H]
  const H0dir = sliceDirNC(g, H0, dirIndex, Ndim, H, op, "H0");
  const C0dir = sliceDirNC(g, C0, dirIndex, Ndim, H, op, "C0");

  // trip_count & cond
  const tScalar = indexFromXTime(g, X, direction); // scalar T or dynamic
  const cond    = makeTensorConst(g, uniq(g, `cond_${op.id}_${direction}`), DataType.BOOL, "constant", bool(true));

  // Split sizes and axes for Unsqueeze
  const splitSizes = makeTensorConst(g, uniq(g, `split_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([H,H,H,H]));

  // --- Build Loop body graph
  const body = new (OnnxGraph as any).Class(uniq(g, `lstm_body_${op.id}_${direction}`)) as OnnxGraph.Class;

  const iter   = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
  const condIn = body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL,  [], "input")).as(TensorNode);
  const Hin    = body.addNode(uniq(body, "Hin")).init(new TensorNode.Builder(DataType.FLOAT, [Ndim, H], "input")).as(TensorNode);
  const Cin    = body.addNode(uniq(body, "Cin")).init(new TensorNode.Builder(DataType.FLOAT, [Ndim, H], "input")).as(TensorNode);

  const Xin    = body.addNode(uniq(body, "Xin")).init(new TensorNode.Builder(DataType.FLOAT, [Tdim, Ndim, Idim], "input")).as(TensorNode);
  const Wt_in  = body.addNode(uniq(body, "Wt_in")).init(new TensorNode.Builder(DataType.FLOAT, [Idim, 4*H], "input")).as(TensorNode);
  const Rt_in  = body.addNode(uniq(body, "Rt_in")).init(new TensorNode.Builder(DataType.FLOAT, [H,   4*H], "input")).as(TensorNode);
  const Bsum_in= body.addNode(uniq(body, "Bsum_in")).init(new TensorNode.Builder(DataType.FLOAT, [4*H],    "input")).as(TensorNode);
  const split_in = body.addNode(uniq(body, "split_sizes")).init(new TensorNode.Builder(DataType.INT64, [4], "input")).as(TensorNode);

  // Handle sequence_lens (masking): get batch mask for this iter (true = use)
  let stepMask: TensorNode.Class | null = null;
  if (seqLens) {
    // mask[b] = (direction=="forward" ? iter < seqLens[b] : iter_rev < seqLens[b])
    // We’ll compute per-batch boolean mask and broadcast to [N,H] via Unsqueeze/Expand.
    const axes0 = makeTensorConst(body, uniq(body, "axes0"), DataType.INT64, "constant", int64Vec([0]));
    const iterU = unsqueeze(body, iter, axes0, "iterU");

    const cmpOp = direction === "forward" ? "Less" : "Less"; // for reverse we compare iter < seqLen as well, but we gather Xt differently
    const cmp = body.addNode(uniq(body, `cmp_len_${direction}`))
      .init(new OperationNode.Builder(cmpOp, [iterU, seqLens]))
      .as(OperationNode);
    stepMask = body.addNode(uniq(body, `mask_${direction}`))
      .init(new TensorNode.Builder(DataType.BOOL, [1, Ndim], "intermediate"))
      .as(TensorNode);
    body.addEdge(cmp, stepMask).init(new OnnxEdge.Builder(stepMask.literalType, stepMask.shape)).as(OnnxEdge);
  }

  // Gather Xt: forward uses iter, reverse uses (T-1-iter)
  let itForGather: TensorNode.Class = iter;
  if (direction === "reverse") {
    const Tconst = makeTensorConst(body, uniq(body, "Tconst"), DataType.INT64, "constant", scalarInt64(Tdim));
    const one    = makeTensorConst(body, uniq(body, "one"), DataType.INT64, "constant", scalarInt64(1));
    const t1 = body.addNode(uniq(body, "Tminus1")).init(new OperationNode.Builder("Sub", [Tconst, one])).as(OperationNode);
    const t1Out = body.addNode(uniq(body, "Tminus1_out")).init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
    body.addEdge(t1, t1Out).init(new OnnxEdge.Builder(t1Out.literalType, t1Out.shape)).as(OnnxEdge);

    const sub = body.addNode(uniq(body, "rev_idx")).init(new OperationNode.Builder("Sub", [t1Out, iter])).as(OperationNode);
    const subOut = body.addNode(uniq(body, "rev_idx_out")).init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
    body.addEdge(sub, subOut).init(new OnnxEdge.Builder(subOut.literalType, subOut.shape)).as(OnnxEdge);
    itForGather = subOut;
  }

  const gXt = body.addNode(uniq(body, "gather_xt"))
    .init(new OperationNode.Builder("Gather", [Xin, itForGather], { axis: 0 }))
    .as(OperationNode);
  const Xt = body.addNode(uniq(body, "Xt"))
    .init(new TensorNode.Builder(DataType.FLOAT, [Ndim, Idim], "intermediate"))
    .as(TensorNode);
  body.addEdge(gXt, Xt).init(new OnnxEdge.Builder(Xt.literalType, Xt.shape)).as(OnnxEdge);

  // pre = Xt@Wt + Hin@Rt + Bsum
  const mm1 = body.addNode(uniq(body, "mm_xw")).init(new OperationNode.Builder("MatMul", [Xt, Wt_in])).as(OperationNode);
  const XW  = body.addNode(uniq(body, "XW")).init(new TensorNode.Builder(DataType.FLOAT, [Ndim, 4*H], "intermediate")).as(TensorNode);
  body.addEdge(mm1, XW).init(new OnnxEdge.Builder(XW.literalType, XW.shape)).as(OnnxEdge);

  const mm2 = body.addNode(uniq(body, "mm_hr")).init(new OperationNode.Builder("MatMul", [Hin, Rt_in])).as(OperationNode);
  const HR  = body.addNode(uniq(body, "HR")).init(new TensorNode.Builder(DataType.FLOAT, [Ndim, 4*H], "intermediate")).as(TensorNode);
  body.addEdge(mm2, HR).init(new OnnxEdge.Builder(HR.literalType, HR.shape)).as(OnnxEdge);

  const add0 = body.addNode(uniq(body, "add0")).init(new OperationNode.Builder("Add", [XW, HR])).as(OperationNode);
  const pre0 = body.addNode(uniq(body, "pre0")).init(new TensorNode.Builder(DataType.FLOAT, [Ndim, 4*H], "intermediate")).as(TensorNode);
  body.addEdge(add0, pre0).init(new OnnxEdge.Builder(pre0.literalType, pre0.shape)).as(OnnxEdge);

  const add1 = body.addNode(uniq(body, "add1")).init(new OperationNode.Builder("Add", [pre0, Bsum_in])).as(OperationNode);
  const pre  = body.addNode(uniq(body, "pre")).init(new TensorNode.Builder(DataType.FLOAT, [Ndim, 4*H], "intermediate")).as(TensorNode);
  body.addEdge(add1, pre).init(new OnnxEdge.Builder(pre.literalType, pre.shape)).as(OnnxEdge);

  // Split(pre, sizes=[H,H,H,H]) in ONNX 'iofc' order → Ipre,Opre,Fpre,Cpre
  const split = body.addNode(uniq(body, "split_gates"))
    .init(new OperationNode.Builder("Split", [pre, split_in], { axis: 1 }))
    .as(OperationNode);
  const Ipre = makeGateOut(body, Ndim, H, "Ipre");
  const Opre = makeGateOut(body, Ndim, H, "Opre");
  const Fpre = makeGateOut(body, Ndim, H, "Fpre");
  const Cpre = makeGateOut(body, Ndim, H, "Cpre");
  body.addEdge(split, Ipre).init(new OnnxEdge.Builder(Ipre.literalType, Ipre.shape)).as(OnnxEdge);
  body.addEdge(split, Opre).init(new OnnxEdge.Builder(Opre.literalType, Opre.shape)).as(OnnxEdge);
  body.addEdge(split, Fpre).init(new OnnxEdge.Builder(Fpre.literalType, Fpre.shape)).as(OnnxEdge);
  body.addEdge(split, Cpre).init(new OnnxEdge.Builder(Cpre.literalType, Cpre.shape)).as(OnnxEdge);

  // Activations
  const I = applyAct(body, activations.f, Ipre, "I");
  const F = applyAct(body, activations.f, Fpre, "F");
  const O = applyAct(body, activations.f, Opre, "O");
  const G = applyAct(body, activations.g, Cpre, "G");
  const F_Cin = mul(body, F, Cin, "F_Cin", [Ndim, H]);
  const I_G   = mul(body, I, G,   "I_G",   [Ndim, H]);
  const Ct    = add(body, F_Cin, I_G, "Ct", [Ndim, H]);

  const tCt   = applyAct(body, activations.h, Ct, "tCt");
  let Ht      = mul(body, O, tCt, "Ht", [Ndim, H]);

  // If sequence_lens present, mask (keep previous state for finished batches)
  if (seqLens) {
    const axes0 = makeTensorConst(body, uniq(body, "axes0_m"), DataType.INT64, "constant", int64Vec([0]));
    const ones  = makeTensorConst(body, uniq(body, "ones"), DataType.FLOAT, "constant",
      { dataType: DataType.FLOAT, dims: [1, Ndim, H], floatData: new Array(Ndim*H).fill(1) });
    const zeros = makeTensorConst(body, uniq(body, "zeros"), DataType.FLOAT, "constant",
      { dataType: DataType.FLOAT, dims: [1, Ndim, H], floatData: new Array(Ndim*H).fill(0) });

    const maskU = unsqueeze(body, (stepMask as TensorNode.Class), axes0, "maskU"); // [1,N]
    const mask3 = expandTo(body, maskU, [1, Ndim, H], "mask3");                    // [1,N,H]

    const Hexp  = unsqueeze(body, Ht, axes0, "HtU");
    const Cexp  = unsqueeze(body, Ct, axes0, "CtU");
    const HinU  = unsqueeze(body, Hin, axes0, "HinU");
    const CinU  = unsqueeze(body, Cin, axes0, "CinU");

    // select(mask3, Hexp, HinU) etc. (Where: mask*new + (1-mask)*old)
    const maskF = castBoolToFloat(body, mask3, "maskF");
    const invM  = sub(body, ones, maskF, "invM", [1, Ndim, H]);
    const Hsel  = add(body, mul(body, Hexp, maskF, "mH", [1,Ndim,H]),
                           mul(body, HinU, invM, "iH", [1,Ndim,H]), "Hsel", [1,Ndim,H]);
    const Csel  = add(body, mul(body, Cexp, maskF, "mC", [1,Ndim,H]),
                           mul(body, CinU, invM, "iC", [1,Ndim,H]), "Csel", [1,Ndim,H]);

    Ht = squeezeTo2(body, Hsel, "sqHs", [Ndim, H]);
    // Ct already kept in Csel; squeeze:
    const Ct2 = squeezeTo2(body, Csel, "sqCs", [Ndim, H]);

    // overwrite Ct with masked Ct2:
    assignTensor(Ct, Ct2);
  }

  // Body outputs: cond passthrough, H_out, C_out, X_out (pass), H_scan
  const condOut = identityOut(body, condIn, "cond_out", DataType.BOOL, []);
  const Hout    = outFrom(body, Ht, "H_out", DataType.FLOAT, [Ndim, H]);
  const Cout    = outFrom(body, Ct, "C_out", DataType.FLOAT, [Ndim, H]);
  const Xout    = outFrom(body, Xin, "X_out", DataType.FLOAT, [Tdim, Ndim, Idim]);
  const Hscan   = outFrom(body, Ht, "H_scan", DataType.FLOAT, [Ndim, H]);

  // --- Outer Loop node
  const loop = g.addNode(uniq(g, `Loop_${op.id}_${direction}`))
    .init(new OperationNode.Builder("Loop",
      [tScalar, makeTensorConst(g, uniq(g, `cond_${op.id}_${direction}_outer`), DataType.BOOL, "constant", bool(true)),
       H0dir, C0dir, X, Wt, Rt, Bsum, splitSizes],
      {},
      body))
    .as(OperationNode);

  const H_end = g.addNode(uniq(g, `H_end_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [Ndim, H], "intermediate")).as(TensorNode);
  const C_end = g.addNode(uniq(g, `C_end_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [Ndim, H], "intermediate")).as(TensorNode);
  const X_last = g.addNode(uniq(g, `X_last_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [Tdim, Ndim, Idim], "intermediate")).as(TensorNode);
  const Y_scan = g.addNode(uniq(g, `Y_scan_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [Tdim, Ndim, H], "intermediate")).as(TensorNode);

  g.addEdge(loop, H_end).init(new OnnxEdge.Builder(H_end.literalType, H_end.shape)).as(OnnxEdge);
  g.addEdge(loop, C_end).init(new OnnxEdge.Builder(C_end.literalType, C_end.shape)).as(OnnxEdge);
  g.addEdge(loop, X_last).init(new OnnxEdge.Builder(X_last.literalType, X_last.shape)).as(OnnxEdge);
  g.addEdge(loop, Y_scan).init(new OnnxEdge.Builder(Y_scan.literalType, Y_scan.shape)).as(OnnxEdge);

  // Final public outputs
  const axesSeq = makeTensorConst(g, uniq(g, `axes_seq_pub_${op.id}_${direction}`), DataType.INT64, "constant", int64Vec([1]));
  const axesYHC = makeTensorConst(g, uniq(g, `axes_yhc_pub_${op.id}_${direction}`), DataType.INT64, "constant", int64Vec([0]));

  const Y_unsq = g.addNode(uniq(g, `unsqY_${op.id}_${direction}`))
    .init(new OperationNode.Builder("Unsqueeze", [Y_scan, axesSeq])).as(OperationNode);
  const Y = g.addNode(uniq(g, `Y_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [Tdim, 1, Ndim, H], "intermediate")).as(TensorNode);
  g.addEdge(Y_unsq, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);

  const Yh_unsq = g.addNode(uniq(g, `unsqYh_${op.id}_${direction}`))
    .init(new OperationNode.Builder("Unsqueeze", [H_end, axesYHC])).as(OperationNode);
  const Y_h = g.addNode(uniq(g, `Y_h_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [1, Ndim, H], "intermediate")).as(TensorNode);
  g.addEdge(Yh_unsq, Y_h).init(new OnnxEdge.Builder(Y_h.literalType, Y_h.shape)).as(OnnxEdge);

  const Yc_unsq = g.addNode(uniq(g, `unsqYc_${op.id}_${direction}`))
    .init(new OperationNode.Builder("Unsqueeze", [C_end, axesYHC])).as(OperationNode);
  const Y_c = g.addNode(uniq(g, `Y_c_${op.id}_${direction}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [1, Ndim, H], "intermediate")).as(TensorNode);
  g.addEdge(Yc_unsq, Y_c).init(new OnnxEdge.Builder(Y_c.literalType, Y_c.shape)).as(OnnxEdge);

  return [Y, Y_h, Y_c];
}

/* ----------------- small builder-style helpers (match your patterns) ----------------- */

function normalizeActivations(a: unknown): { f: string; g: string; h: string } | null {
  // ONNX default: f=Sigmoid, g=Tanh, h=Tanh; activations is 3 strings for single direction
  if (!a) return { f: "Sigmoid", g: "Tanh", h: "Tanh" };
  const arr = Array.isArray(a) ? a.map(String) : [String(a)];
  if (arr.length === 0) return { f: "Sigmoid", g: "Tanh", h: "Tanh" };
  const [f, g, h] = [arr[0] ?? "Sigmoid", arr[1] ?? "Tanh", arr[2] ?? "Tanh"];
  const ok = new Set(["Sigmoid","Tanh"]);
  if (!ok.has(f) || !ok.has(g) || !ok.has(h)) return null;
  return { f, g, h };
}

function sliceDirNC(
  g: OnnxGraph.Class, T: TensorNode.Class, dirIndex: number, N: number, H: number, op: OperationNode.Class, tag: string
): TensorNode.Class {
  const starts = makeTensorConst(g, uniq(g, `${tag}_s_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex, 0, 0]));
  const ends   = makeTensorConst(g, uniq(g, `${tag}_e_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([dirIndex+1, N, H]));
  const axes   = makeTensorConst(g, uniq(g, `${tag}_axes_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([0,1,2]));
  const steps  = makeTensorConst(g, uniq(g, `${tag}_steps_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([1,1,1]));
  const sl = g.addNode(uniq(g, `Slice_${tag}_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Slice", [T, starts, ends, axes, steps]))
    .as(OperationNode);
  const out3 = g.addNode(uniq(g, `${tag}_3_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [1, N, H], "intermediate"))
    .as(TensorNode);
  g.addEdge(sl, out3).init(new OnnxEdge.Builder(out3.literalType, out3.shape)).as(OnnxEdge);

  const ax0 = makeTensorConst(g, uniq(g, `${tag}_ax0_${op.id}_${dirIndex}`), DataType.INT64, "constant", int64Vec([0]));
  const sq = g.addNode(uniq(g, `Squeeze_${tag}_${op.id}_${dirIndex}`))
    .init(new OperationNode.Builder("Squeeze", [out3, ax0])).as(OperationNode);
  const out2 = g.addNode(uniq(g, `${tag}_2_${op.id}_${dirIndex}`))
    .init(new TensorNode.Builder(DataType.FLOAT, [N, H], "intermediate"))
    .as(TensorNode);
  g.addEdge(sq, out2).init(new OnnxEdge.Builder(out2.literalType, out2.shape)).as(OnnxEdge);
  return out2;
}

function indexFromXTime(g: OnnxGraph.Class, X: TensorNode.Class, dir: "forward"|"reverse"): TensorNode.Class {
  // Prefer static T if available; otherwise: Shape(X)[0]
  const shp = toStaticShape(X.shape as Shape);
  if (shp[0] > 0) {
    return makeTensorConst(g, uniq(g, `trip_${X.id}_${dir}`), DataType.INT64, "constant", scalarInt64(shp[0]));
  }
  const shapeN = g.addNode(uniq(g, `shape_${X.id}_${dir}`))
    .init(new OperationNode.Builder("Shape", [X])).as(OperationNode);
  const shapeOut = g.addNode(uniq(g, `shape_out_${X.id}_${dir}`))
    .init(new TensorNode.Builder(DataType.INT64, [3], "intermediate")).as(TensorNode);
  g.addEdge(shapeN, shapeOut).init(new OnnxEdge.Builder(shapeOut.literalType, shapeOut.shape)).as(OnnxEdge);

  const idx0 = makeTensorConst(g, uniq(g, `idx0_${X.id}_${dir}`), DataType.INT64, "constant", scalarInt64(0));
  const g0 = g.addNode(uniq(g, `g0_${X.id}_${dir}`))
    .init(new OperationNode.Builder("Gather", [shapeOut, idx0], { axis: 0 })).as(OperationNode);
  const out = g.addNode(uniq(g, `g0_out_${X.id}_${dir}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
  g.addEdge(g0, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

/* Simple helpers using your builder style */

function makeGateOut(body: OnnxGraph.Class, N: number, H: number, name: string): TensorNode.Class {
  return body.addNode(uniq(body, name))
    .init(new TensorNode.Builder(DataType.FLOAT, [N, H], "intermediate"))
    .as(TensorNode);
}

function applyAct(body: OnnxGraph.Class, kind: string, x: TensorNode.Class, tag: string): TensorNode.Class {
  const opName = (kind === "Sigmoid" ? "Sigmoid" : "Tanh");
  const n = body.addNode(uniq(body, `${tag}_${opName}`))
    .init(new OperationNode.Builder(opName, [x])).as(OperationNode);
  const out = body.addNode(uniq(body, `${tag}_out`))
    .init(new TensorNode.Builder(DataType.FLOAT, x.shape, "intermediate")).as(TensorNode);
  body.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function mul(body: OnnxGraph.Class, a: TensorNode.Class, b: TensorNode.Class, tag: string, shape: number[]): TensorNode.Class {
  const n = body.addNode(uniq(body, `mul_${tag}`))
    .init(new OperationNode.Builder("Mul", [a,b])).as(OperationNode);
  const out = body.addNode(uniq(body, `mul_${tag}_out`))
    .init(new TensorNode.Builder(DataType.FLOAT, shape, "intermediate")).as(TensorNode);
  body.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}
function add(body: OnnxGraph.Class, a: TensorNode.Class, b: TensorNode.Class, tag: string, shape: number[]): TensorNode.Class {
  const n = body.addNode(uniq(body, `add_${tag}`))
    .init(new OperationNode.Builder("Add", [a,b])).as(OperationNode);
  const out = body.addNode(uniq(body, `add_${tag}_out`))
    .init(new TensorNode.Builder(DataType.FLOAT, shape, "intermediate")).as(TensorNode);
  body.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function sub(body: OnnxGraph.Class, a: TensorNode.Class, b: TensorNode.Class, tag: string, shape: number[]): TensorNode.Class {
  const n = body.addNode(uniq(body, `sub_${tag}`))
    .init(new OperationNode.Builder("Sub", [a,b])).as(OperationNode);
  const out = body.addNode(uniq(body, `sub_${tag}_out`))
    .init(new TensorNode.Builder(DataType.FLOAT, shape, "intermediate")).as(TensorNode);
  body.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function unsqueeze(body: OnnxGraph.Class, t: TensorNode.Class, axes: TensorNode.Class, tag: string): TensorNode.Class {
  const u = body.addNode(uniq(body, tag)).init(new OperationNode.Builder("Unsqueeze", [t, axes])).as(OperationNode);
  const out = body.addNode(uniq(body, `${tag}_out`)).init(new TensorNode.Builder(t.literalType, [1, ...(t.shape ?? [])], "intermediate")).as(TensorNode);
  body.addEdge(u, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function expandTo(body: OnnxGraph.Class, t: TensorNode.Class, target: number[], tag: string): TensorNode.Class {
  const shp = makeTensorConst(body, uniq(body, `shape_${tag}`), DataType.INT64, "constant", int64Vec(target));
  const e = body.addNode(uniq(body, `Expand_${tag}`)).init(new OperationNode.Builder("Expand", [t, shp])).as(OperationNode);
  const out = body.addNode(uniq(body, `Expand_${tag}_out`)).init(new TensorNode.Builder(t.literalType, target, "intermediate")).as(TensorNode);
  body.addEdge(e, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function castBoolToFloat(body: OnnxGraph.Class, t: TensorNode.Class, tag: string): TensorNode.Class {
  const c = body.addNode(uniq(body, `cast_${tag}`)).init(new OperationNode.Builder("Cast", [t], { to: DataType.FLOAT })).as(OperationNode);
  const out = body.addNode(uniq(body, `cast_${tag}_out`)).init(new TensorNode.Builder(DataType.FLOAT, t.shape, "intermediate")).as(TensorNode);
  body.addEdge(c, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function squeezeTo2(body: OnnxGraph.Class, t: TensorNode.Class, tag: string, shape2: number[]): TensorNode.Class {
  const axes = makeTensorConst(body, uniq(body, `sq_axes_${tag}`), DataType.INT64, "constant", int64Vec([0]));
  const s = body.addNode(uniq(body, `Squeeze_${tag}`)).init(new OperationNode.Builder("Squeeze", [t, axes])).as(OperationNode);
  const out = body.addNode(uniq(body, `Squeeze_${tag}_out`)).init(new TensorNode.Builder(DataType.FLOAT, shape2, "intermediate")).as(TensorNode);
  body.addEdge(s, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function identityOut(body: OnnxGraph.Class, t: TensorNode.Class, name: string, ty: DataType, shape: number[]): TensorNode.Class {
  const n = body.addNode(uniq(body, `id_${name}`)).init(new OperationNode.Builder("Identity", [t])).as(OperationNode);
  const out = body.addNode(name).init(new TensorNode.Builder(ty, shape, "output")).as(TensorNode);
  body.addEdge(n, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}
function outFrom(body: OnnxGraph.Class, t: TensorNode.Class, name: string, ty: DataType, shape: number[]): TensorNode.Class {
  const out = body.addNode(name).init(new TensorNode.Builder(ty, shape, "output")).as(TensorNode);
  body.addEdge(t, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function assignTensor(dst: TensorNode.Class, src: TensorNode.Class) {
  // In this builder style we don’t mutate tensors in place; we just make Ht point to src by returning it.
  // Here we rely on caller to use returned value (done above for Ct/Ht).
}

function rewireAndRemove(g: OnnxGraph.Class, op: OperationNode.Class, outs: [TensorNode.Class, TensorNode.Class, TensorNode.Class]) {
  const [Y, Y_h, Y_c] = outs;
  const outEdges = op.getOutgoers;
  const yT  = outEdges.targets?.filterIs(TensorNode)?.at(0);
  const yhT = outEdges.targets?.filterIs(TensorNode)?.at(1);
  const ycT = outEdges.targets?.filterIs(TensorNode)?.at(2);

  function replace(dst: TensorNode.Class | undefined, src: TensorNode.Class) {
    if (!dst) return;
    dst.getIncomers.forEach(e => e.remove());
    g.addEdge(src, dst).init(new OnnxEdge.Builder(dst.literalType, dst.shape)).as(OnnxEdge);
  }

  replace(yT,  Y);
  replace(yhT, Y_h);
  replace(ycT, Y_c);

  op.getOutgoers.forEach(e => e.remove());
  op.remove();
}
