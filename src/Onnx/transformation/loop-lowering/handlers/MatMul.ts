import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { makeTensorConst, scalarInt64, uniq, int64Vec, toStaticShape } from "@specs-feup/onnx-flow/Onnx/Utils";
import { unsqueezeIdx, LoopCtx, resolveFusedInput, divmod, targetReshape, gatherFrom, reshapeTensor, squeezeIfLen1, broadcastShapes, decodeMixedRadix } from "../BuildLoop.js";


/* ============================== local helpers ============================== */

/** Product of an array of numbers (treat empty as 1). */
function prod(xs: number[]): number {
  return xs.length ? xs.reduce((p, v) => p * v, 1) : 1;
}

/**
 * Slice a tensor along its batch axes (before the last 2 dims) using the given batch digits.
 * If the source has size 1 on an axis, we pick index 0 (broadcast behavior).
 * Returns the sliced tensor (batch axes indexed away), reshaped to 2D.
 */
function sliceBatchThenReshape2D(
  g: OnnxGraph.Class,
  t: TensorNode.Class,
  srcBatch: number[],
  batch: number[],
  batchDigits: TensorNode.Class[],
  axesConst: TensorNode.Class,
  M: TensorNode.Class,
  K_or_N: TensorNode.Class,
  tag: string
): TensorNode.Class {
  let cur = t;

  if (batch.length > 0) {
    const zero = makeTensorConst(g, `bz_${tag}`, DataType.INT64, "constant", scalarInt64(0));

    for (let ax = 0; ax < batch.length; ax++) {
      if (ax >= srcBatch.length) continue;

      const srcDim = srcBatch[ax];
      const pickScalar = srcDim === 1 ? zero : batchDigits[ax];
      const pickU = unsqueezeIdx(g, pickScalar, axesConst, `bU_${tag}_${ax}`);

      const gather = g
        .addNode(uniq(g, `bGather_${tag}_${ax}`))
        .init(new OperationNode.Builder("Gather", [cur, pickU], { axis: ax }))
        .as(OperationNode);

      const out = g
        .addNode(uniq(g, `bGather_out_${tag}_${ax}`))
        .init(new TensorNode.Builder(cur.literalType, [], "intermediate"))
        .as(TensorNode);

      g.addEdge(gather, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
      cur = out;
    }
  }

  const shape2 = shapeVec2(g, `shape2_${tag}`, M, K_or_N);
  const resh = g.addNode(uniq(g, `reshape2D_${tag}`))
    .init(new OperationNode.Builder("Reshape", [cur, shape2])).as(OperationNode);
  const out2d = g.addNode(uniq(g, `reshape2D_out_${tag}`)).init(new TensorNode.Builder(cur.literalType, shape2.shape, "intermediate")).as(TensorNode);
  g.addEdge(resh, out2d).init(new OnnxEdge.Builder(out2d.literalType, out2d.shape)).as(OnnxEdge);

  return out2d;
}

function isKnownDim(d: number | undefined) {
  return typeof d === 'number' && Number.isFinite(d) && d > 0;
}

function scalarI64(g: OnnxGraph.Class, name: string, v: number) {
  return makeTensorConst(g, name, DataType.INT64, "constant", scalarInt64(v));
}

function gatherDim(
  g: OnnxGraph.Class,
  tag: string,
  src: TensorNode.Class,   // e.g., A2D or B2D
  negAxis: -2 | -1         // which trailing dim to read
): TensorNode.Class {
  const shape = g.addNode(uniq(g, `shape_${tag}`))
    .init(new OperationNode.Builder("Shape", [src]))
    .as(OperationNode);
  // Create tensor for Shape's output (the shape vector)
  const shapeO = g.addNode(`shape_O_${tag}`)
    .init(new TensorNode.Builder(
      DataType.INT64,         // Shape outputs int64
      [src.shape.length], 
      "intermediate"
    ))
    .as(TensorNode);

  // Wire op → tensor
  g.addEdge(shape, shapeO).init(new OnnxEdge.Builder(shapeO.literalType, shapeO.shape)).as(OnnxEdge);

  const idx = makeTensorConst(g, `idx_${tag}`, DataType.INT64, "constant", int64Vec([negAxis]));
  const gather = g.addNode(uniq(g, `g_${tag}`))
    .init(new OperationNode.Builder("Gather", [shapeO, idx], { axis: 0 }))
    .as(OperationNode);
  const out = g.addNode(uniq(g, `g_out_${tag}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(gather, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function as1D(g: OnnxGraph.Class, name: string, scalarI64T: TensorNode.Class) {
  const axes = makeTensorConst(g, `axes_${name}`, DataType.INT64, "constant", int64Vec([0]));
  return unsqueezeIdx(g, scalarI64T, axes, `${name}_u`); // 1-D [1] from scalar
}

function shapeVec2(
  g: OnnxGraph.Class,
  name: string,
  d0: TensorNode.Class,  // INT64 scalar
  d1: TensorNode.Class   // INT64 scalar
) {
  const d0v = as1D(g, `${name}_d0`, d0);
  const d1v = as1D(g, `${name}_d1`, d1);
  const cat = g.addNode(uniq(g, `concat_${name}`))
    .init(new OperationNode.Builder("Concat", [d0v, d1v], { axis: 0 }))
    .as(OperationNode);
  const out = g.addNode(uniq(g, `concat_out_${name}`))
    .init(new TensorNode.Builder(DataType.INT64, [2], "intermediate"))
    .as(TensorNode);
  g.addEdge(cat, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}


/* ============================== HANDLER ================================== */

export default function handleMatMul(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx
): TensorNode.Class {
  // Pull trusted dims from the builder
  const dims = ctx.matmulDims;
  if (!dims) {
    throw new Error(`handleMatMul: ctx.matmulDims is missing for ${op.id}`);
  }

  const lhsInput = op.getInputs()![0];
  const rhsInput = op.getInputs()![1];

  const lhsTensor = resolveFusedInput(g, lhsInput, ctx, op, false, false);
  const rhsTensor = resolveFusedInput(g, rhsInput, ctx, op, false, false);

  const aShape = lhsTensor.shape as number[];
  const bShape = rhsTensor.shape as number[];

  const { M, K, N } = dims;

  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);
  const batch = toStaticShape(dims.batchDims);
  const batchProd = dims.batchProd;

  const elemTy = lhsTensor.literalType;

  // Common constants
  const M_c = isKnownDim(M) ? scalarI64(g, `M_${op.id}`, Number(M))
                            : gatherDim(g, `M_${op.id}`, lhsTensor, -2);
  const K_c = isKnownDim(K) ? scalarI64(g, `K_${op.id}`, Number(K))
                            : gatherDim(g, `K_${op.id}`, lhsTensor, -1);
  const N_c = isKnownDim(N) ? scalarI64(g, `N_${op.id}`, Number(N))
                            : gatherDim(g, `N_${op.id}`, rhsTensor, -1);

  // KN = K*N
  const KN_node = g
    .addNode(uniq(g, `KN_${op.id}`))
    .init(new OperationNode.Builder("Mul", [K_c, N_c]))
    .as(OperationNode);
  const KN = g
    .addNode(uniq(g, `KN_out_${op.id}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(KN_node, KN).init(new OnnxEdge.Builder(KN.literalType, KN.shape)).as(OnnxEdge);

  // MN = M*N
  const MN_node = g
    .addNode(uniq(g, `MN_${op.id}`))
    .init(new OperationNode.Builder("Mul", [M_c, N_c]))
    .as(OperationNode);
  const MN = g
    .addNode(uniq(g, `MN_out_${op.id}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(MN_node, MN).init(new OnnxEdge.Builder(MN.literalType, MN.shape)).as(OnnxEdge);

  // MKN = M*(K*N)
  const MKN_node = g
    .addNode(uniq(g, `MKN_${op.id}`))
    .init(new OperationNode.Builder("Mul", [M_c, KN]))
    .as(OperationNode);
  const MKN = g
    .addNode(uniq(g, `MKN_out_${op.id}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  g.addEdge(MKN_node, MKN).init(new OnnxEdge.Builder(MKN.literalType, MKN.shape)).as(OnnxEdge);

  // ---------------- batch slicing -> A2D:[M,K], B2D:[K,N] ----------------
  let A2D = lhsTensor;
  let B2D = rhsTensor;

  let bIdx: TensorNode.Class | null = null;    // INT64 scalar
  let tIn: TensorNode.Class | null = null;     // INT64 scalar: within-batch-and-(i,j,k)

  if (batch.length > 0) {
    // b = floor(t / (M*K*N))
    bIdx = divmod(g, ctx.iter, MKN, `b_${op.id}`, "Div");
    // t_in = t % (M*K*N)
    tIn = divmod(g, ctx.iter, MKN, `tin_${op.id}`, "Mod");

    const bDigits = decodeMixedRadix(g, bIdx, batch, `batch_${op.id}`);
    const axesConst = ctx.axes;

    A2D = sliceBatchThenReshape2D(
      g,
      lhsTensor,
      aBatch,
      batch,
      bDigits,
      axesConst,
      M_c,
      K_c,
      `A_${op.id}`
    );
    B2D = sliceBatchThenReshape2D(
      g,
      rhsTensor,
      bBatch,
      batch,
      bDigits,
      axesConst,
      K_c,
      N_c,
      `B_${op.id}`
    );
  } else {
    // no batch: t_in = t
    tIn = ctx.iter;
    const shapeA2D = shapeVec2(g, `shape2_A_${op.id}`, M_c, K_c);
    A2D = reshapeTensor(g, lhsTensor, shapeA2D, `reshape2D_A_${op.id}`); 
    const shapeB2D = shapeVec2(g, `shape2_B_${op.id}`, K_c, N_c);
    B2D = reshapeTensor(g, rhsTensor, shapeB2D, `reshape2D_A_${op.id}`); 
  }

  // ---------------- decode i,j,k from t_in (NOT from full t) ----------------
  // i = floor(t_in / (K*N))
  const iIdx = divmod(g, tIn!, KN, `i_${op.id}`, "Div");
  // rem = t_in % (K*N)
  const rem = divmod(g, tIn!, KN, `rem_${op.id}`, "Mod");
  // j = floor(rem / K)
  const jIdx = divmod(g, rem, K_c, `j_${op.id}`, "Div");
  // k = rem % K
  const kIdx = divmod(g, rem, K_c, `k_${op.id}`, "Mod");

  // [1]-shaped indices for gathers
  const iU = unsqueezeIdx(g, iIdx, ctx.axes, `iU_${op.id}`);
  const jU = unsqueezeIdx(g, jIdx, ctx.axes, `jU_${op.id}`);
  const kU = unsqueezeIdx(g, kIdx, ctx.axes, `kU_${op.id}`);

  // Save in ctx for builder/chain
  ctx.iU = iU;
  ctx.jU = jU;
  ctx.kU = kU;
  ctx.kIdx = kIdx;

  if (!ctx.coalesce) {
    // ================= NON-COALESCED =================
    // Use i/j we already decoded from t_in (batch removed).
    const shape1 = makeTensorConst(
      g,
      `shape1_${op.id}`,
      DataType.INT64,
      "constant",
      int64Vec([1])
    );
    const shapeK = makeTensorConst(
      g,
      `shapeK_${op.id}`,
      DataType.INT64,
      "constant",
      int64Vec([K])
    );

    // A2D: [M,K] -> row i -> [1,K]
    const [_, rowGathered] = gatherFrom(g, A2D, `gather_${A2D.id}_${op.id}`, iU, 0);
    // B2D: [K,N] -> col j -> [K,1]
    const [__, colGathered] = gatherFrom(g, B2D, `gather_${B2D.id}_${op.id}`, jU, 1);

    const row = reshapeTensor(g, rowGathered, shapeK, `reshapeRow_${op.id}`); // [K]
    const col = reshapeTensor(g, colGathered, shapeK, `reshapeCol_${op.id}`); // [K]

    const mul = g
      .addNode(uniq(g, `mul_${op.id}`))
      .init(new OperationNode.Builder("Mul", [row, col]))
      .as(OperationNode);
    const mulOut = g
      .addNode(uniq(g, `mul_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [K], "intermediate"))
      .as(TensorNode);
    g.addEdge(mul, mulOut).init(new OnnxEdge.Builder(elemTy, [K]));

    const reduce = g
      .addNode(uniq(g, `reduce_${op.id}`))
      .init(new OperationNode.Builder("ReduceSum", [mulOut, ctx.axes]))
      .as(OperationNode);
    const reduceOut = g
      .addNode(uniq(g, `reduce_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(reduce, reduceOut).init(new OnnxEdge.Builder(elemTy, []));

    const reshape = g
      .addNode(uniq(g, `reshape_${op.id}`))
      .init(new OperationNode.Builder("Reshape", [reduceOut, shape1]))
      .as(OperationNode);
    const finalOut = g
      .addNode(uniq(g, `final_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
      .as(TensorNode);
    g.addEdge(reshape, finalOut).init(new OnnxEdge.Builder(elemTy, [1]));

    return finalOut;
  } else {
    // ================= COALESCED (scalar MAC) =================
    // flat (with batch offset): b*(M*N) + i*N + j
    let flatBase = iIdx;

    const iMulN_node = g
      .addNode(uniq(g, `iMulN_${op.id}`))
      .init(new OperationNode.Builder("Mul", [iIdx, N_c]))
      .as(OperationNode);
    const iMulN = g
      .addNode(uniq(g, `iMulN_out_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, iIdx.shape, "intermediate"))
      .as(TensorNode);
    g.addEdge(iMulN_node, iMulN).init(new OnnxEdge.Builder(iMulN.literalType, iMulN.shape)).as(OnnxEdge);

    const flat_no_batch_node = g
      .addNode(uniq(g, `flat_no_batch_${op.id}`))
      .init(new OperationNode.Builder("Add", [iMulN, jIdx]))
      .as(OperationNode);
    const flat_no_batch = g
      .addNode(uniq(g, `flat_no_batch_out_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, iIdx.shape, "intermediate"))
      .as(TensorNode);
    g.addEdge(flat_no_batch_node, flat_no_batch).init(new OnnxEdge.Builder(flat_no_batch.literalType, flat_no_batch.shape)).as(OnnxEdge);

    let flat = flat_no_batch;
    if (batch.length > 0) {
      // bIdx already computed above
      const bMulMN_node = g
        .addNode(uniq(g, `bMulMN_${op.id}`))
        .init(new OperationNode.Builder("Mul", [bIdx!, MN]))
        .as(OperationNode);
      const bMulMN = g
        .addNode(uniq(g, `bMulMN_out_${op.id}`))
        .init(new TensorNode.Builder(DataType.INT64, iIdx.shape, "intermediate"))
        .as(TensorNode);
      g.addEdge(bMulMN_node, bMulMN).init(new OnnxEdge.Builder(bMulMN.literalType, bMulMN.shape)).as(OnnxEdge);

      const flat_with_b_node = g
        .addNode(uniq(g, `flat_${op.id}`))
        .init(new OperationNode.Builder("Add", [bMulMN, flat_no_batch]))
        .as(OperationNode);
      flat = g
        .addNode(uniq(g, `flat_out_${op.id}`))
        .init(new TensorNode.Builder(DataType.INT64, iIdx.shape, "intermediate"))
        .as(TensorNode);
      g.addEdge(flat_with_b_node, flat).init(new OnnxEdge.Builder(flat.literalType, flat.shape)).as(OnnxEdge);
    }

    const flatU = unsqueezeIdx(g, flat, ctx.axes, `flatU_${op.id}`); // [1]
    ctx.flatU = flatU;
    ctx.unsqIdx = flatU;

    // A[i,k] (A2D is [M,K])
    const a_row_node = g
      .addNode(uniq(g, `a_row_${op.id}`))
      .init(new OperationNode.Builder("Gather", [A2D, iU], { axis: 0 }))
      .as(OperationNode);
    const a_row = g
      .addNode(uniq(g, `a_row_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1, K], "intermediate"))
      .as(TensorNode);
    g.addEdge(a_row_node, a_row).init(new OnnxEdge.Builder(a_row.literalType, a_row.shape)).as(OnnxEdge);

    //const Kv1 = as1D(g, `fixshape_a_vec_${op.id}`, K_c);
    const a_vec = targetReshape(g, a_row, [K], `a_vec_${op.id}`);

    const a_pick_node = g
      .addNode(uniq(g, `a_pick_${op.id}`))
      .init(new OperationNode.Builder("Gather", [a_vec, kU], { axis: 0 }))
      .as(OperationNode);
    const a_scalar = g
      .addNode(uniq(g, `a_scalar_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
      .as(TensorNode);
    g.addEdge(a_pick_node, a_scalar).init(new OnnxEdge.Builder(a_scalar.literalType, a_scalar.shape)).as(OnnxEdge);
    const sq_a = squeezeIfLen1(g, a_scalar, ctx.axes, `a_sq_${op.id}`);

    // B[k,j] (B2D is [K,N])
    const b_col_node = g
      .addNode(uniq(g, `b_col_${op.id}`))
      .init(new OperationNode.Builder("Gather", [B2D, jU], { axis: 1 }))
      .as(OperationNode);
    const b_col = g
      .addNode(uniq(g, `b_col_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1, K], "intermediate"))
      .as(TensorNode);
    g.addEdge(b_col_node, b_col).init(new OnnxEdge.Builder(b_col.literalType, b_col.shape)).as(OnnxEdge);


    //const Kv1b = as1D(g, `fixshape_b_vec_${op.id}`, K_c);
    const b_vec = targetReshape(g, b_col, [K], `b_vec_${op.id}`);

    const b_pick_node = g
      .addNode(uniq(g, `b_pick_${op.id}`))
      .init(new OperationNode.Builder("Gather", [b_vec, kU], { axis: 0 }))
      .as(OperationNode);
    const b_scalar = g
      .addNode(uniq(g, `b_scalar_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
      .as(TensorNode);
    g.addEdge(b_pick_node, b_scalar).init(new OnnxEdge.Builder(b_scalar.literalType, b_scalar.shape)).as(OnnxEdge);
    const sq_b = squeezeIfLen1(g, b_scalar, ctx.axes, `b_sq_${op.id}`);

    // prod = A[i,k] * B[k,j]
    const mul_node = g
      .addNode(uniq(g, `mul_${op.id}`))
      .init(new OperationNode.Builder("Mul", [sq_a, sq_b]))
      .as(OperationNode);
    const prodT = g
      .addNode(uniq(g, `prod_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(mul_node, prodT).init(new OnnxEdge.Builder(prodT.literalType, prodT.shape)).as(OnnxEdge);

    // prev = carry[flat]
    const prev_node = g
      .addNode(uniq(g, `prev_${op.id}`))
      .init(new OperationNode.Builder("GatherElements", [ctx.carry, flatU], { axis: 0 }))
      .as(OperationNode);
    const prev = g
      .addNode(uniq(g, `prev_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, flatU.shape, "intermediate"))
      .as(TensorNode);
    g.addEdge(prev_node, prev).init(new OnnxEdge.Builder(prev.literalType, prev.shape)).as(OnnxEdge);
    const prev_sq = squeezeIfLen1(g, prev, ctx.axes, `prev_sq_${op.id}`);

    // acc = prev + prod
    const add_node = g
      .addNode(uniq(g, `acc_${op.id}`))
      .init(new OperationNode.Builder("Add", [prev_sq, prodT]))
      .as(OperationNode);
    const acc = g
      .addNode(uniq(g, `acc_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(add_node, acc).init(new OnnxEdge.Builder(acc.literalType, acc.shape)).as(OnnxEdge);

    if (ctx.gateByK) {
      const one = scalarI64(g, `one_${op.id}`, 1);
      const Km1  = g.addNode(uniq(g, `Km1_${op.id}`))
        .init(new OperationNode.Builder("Sub", [K_c, one]))
        .as(OperationNode);
      const Km1_out = g.addNode(uniq(g, `Km1_out_${op.id}`))
        .init(new TensorNode.Builder(DataType.INT64, [undefined], "intermediate")).as(TensorNode);
      g.addEdge(Km1, Km1_out).init(new OnnxEdge.Builder(Km1_out.literalType, Km1_out.shape)).as(OnnxEdge);
      ctx.kM1 = Km1_out;
    }

    ctx.running = acc;
    return acc; // scalar; outer builder scatters/writes
  }
}
