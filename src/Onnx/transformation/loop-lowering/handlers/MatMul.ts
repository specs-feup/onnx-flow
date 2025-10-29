import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { makeTensorConst, int64Vec, scalarInt64, uniq } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopCtx, resolveFusedInput, divmod, unsqueezeIdx, gatherFrom, reshapeTensor, targetReshape, squeezeIfLen1 } from "../BuildLoop.js";

/* ============================== HANDLER ================================== */

export default function handleMatMul(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx
): TensorNode.Class {
  const lhsInput = op.getInputs()![0];
  const rhsInput = op.getInputs()![1];

  const lhsTensor = resolveFusedInput(g, lhsInput, ctx, op, false, false);
  const rhsTensor = resolveFusedInput(g, rhsInput, ctx, op, false, false);

  const K = (lhsTensor.shape as number[]).at(-1)!;
  const N = (rhsTensor.shape as number[]).at(-1)!;

  const elemTy = lhsTensor.literalType;

  if (!ctx.coalesce) {
    // === OLD BEHAVIOR ===
    const shape1 = makeTensorConst(g, `shape1_${op.id}`, DataType.INT64, "constant", int64Vec([1]));

    const Nconst = makeTensorConst(g, `N_${op.id}`, DataType.INT64, "constant", scalarInt64(N));
    const shapeK = makeTensorConst(g, `shapeK_${op.id}`, DataType.INT64, "constant", int64Vec([K]));

    const rowIdx = divmod(g, ctx.iter, Nconst, "rowIdx", "Div");
    const colIdx = divmod(g, ctx.iter, Nconst, "colIdx", "Mod");

    const rowU = unsqueezeIdx(g, rowIdx, ctx.axes, `rowU_${op.id}`);
    const colU = unsqueezeIdx(g, colIdx, ctx.axes, `colU_${op.id}`);

    const [_, rowGathered] = gatherFrom(g, lhsTensor, `gather_${lhsTensor.id}_${op.id}`, rowU, 0);
    const [__, colGathered] = gatherFrom(g, rhsTensor, `gather_${rhsTensor.id}_${op.id}`, colU, 1);

    const row = reshapeTensor(g, rowGathered, shapeK, `reshapeRow_${op.id}`);
    const col = reshapeTensor(g, colGathered, shapeK, `reshapeCol_${op.id}`);

    const mul = g.addNode(uniq(g, `mul_${op.id}`))
                 .init(new OperationNode.Builder("Mul", [row, col]))
                 .as(OperationNode);
    const mulOut = g.addNode(uniq(g, `mul_out_${op.id}`))
                    .init(new TensorNode.Builder(elemTy, [K], "intermediate"))
                    .as(TensorNode);
    g.addEdge(mul, mulOut).init(new OnnxEdge.Builder(elemTy, [K]));

    const reduce = g.addNode(uniq(g, `reduce_${op.id}`))
                    .init(new OperationNode.Builder("ReduceSum", [mulOut, ctx.axes]))
                    .as(OperationNode);
    const reduceOut = g.addNode(uniq(g, `reduce_out_${op.id}`))
                       .init(new TensorNode.Builder(elemTy, [], "intermediate"))
                       .as(TensorNode);
    g.addEdge(reduce, reduceOut).init(new OnnxEdge.Builder(elemTy, []));

    const reshape = g.addNode(uniq(g, `reshape_${op.id}`))
                     .init(new OperationNode.Builder("Reshape", [reduceOut, shape1]))
                     .as(OperationNode);
    const finalOut = g.addNode(uniq(g, `final_out_${op.id}`))
                      .init(new TensorNode.Builder(elemTy, [1], "intermediate"))
                      .as(TensorNode);
    g.addEdge(reshape, finalOut).init(new OnnxEdge.Builder(elemTy, [1]));

    return finalOut;
  } else {
    // ================= COALESCED MATMUL (scalar MAC) =================

    // ---- constants (int64) ----
    const KN_const = makeTensorConst(g, `KN_${op.id}`, DataType.INT64, "constant", scalarInt64(K * N));
    const K_const  = makeTensorConst(g,  `K_${op.id}`, DataType.INT64, "constant", scalarInt64(K));
    const N_const  = makeTensorConst(g,  `N_${op.id}`, DataType.INT64, "constant", scalarInt64(N));

    // ---- decode (i,j,k) from loop counter ctx.iter ----
    const iIdx = divmod(g, ctx.iter, KN_const, `i_${op.id}`, "Div");   // t // (N*K)
    const rem  = divmod(g, ctx.iter, KN_const, `rem_${op.id}`, "Mod"); // t %  (N*K)
    const jIdx = divmod(g, rem,      K_const,  `j_${op.id}`, "Div");   // rem // K
    const kIdx = divmod(g, rem,      K_const,  `k_${op.id}`, "Mod");   // rem %  K

    // ---- make [1]-shaped indices for Gather/GatherElements ----
    const iU = unsqueezeIdx(g, iIdx, ctx.axes, `iU_${op.id}`);
    const jU = unsqueezeIdx(g, jIdx, ctx.axes, `jU_${op.id}`);
    const kU = unsqueezeIdx(g, kIdx, ctx.axes, `kU_${op.id}`);


    // ---- flat = i*N + j  → flatU = Unsqueeze(flat) ----
    const iMulN_node = g.addNode(uniq(g, `iMulN_${op.id}`))
      .init(new OperationNode.Builder("Mul", [iIdx, N_const])).as(OperationNode);
    const iMulN = g.addNode(uniq(g, `iMulN_out_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, kIdx.shape, "intermediate")).as(TensorNode);
    g.addEdge(iMulN_node, iMulN).init(new OnnxEdge.Builder(iMulN.literalType, iMulN.shape)).as(OnnxEdge);

    const flat_node = g.addNode(uniq(g, `flat_${op.id}`))
      .init(new OperationNode.Builder("Add", [iMulN, jIdx])).as(OperationNode);
    const flat = g.addNode(uniq(g, `flat_out_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, iIdx.shape, "intermediate")).as(TensorNode);
    g.addEdge(flat_node, flat).init(new OnnxEdge.Builder(flat.literalType, flat.shape)).as(OnnxEdge);

    const flatU = unsqueezeIdx(g, flat, ctx.axes, `flatU_${op.id}`); // [1]

    ctx.iU = iU;
    ctx.jU = jU;
    ctx.kU = kU;
    ctx.kIdx = kIdx;
    if (ctx.gateByK) {
      const KM1_const = makeTensorConst(g, `Km1_${op.id}`, DataType.INT64, "constant", scalarInt64(K - 1));
      ctx.kM1 = KM1_const;
    }
    ctx.flatU = flatU;
    ctx.unsqIdx = flatU;


    // ---- A[i,k] as [1] ----
    const a_row_node = g.addNode(uniq(g, `a_row_${op.id}`))
      .init(new OperationNode.Builder("Gather", [lhsTensor, iU], { axis: 0 })).as(OperationNode);
    const a_row = g.addNode(uniq(g, `a_row_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1, K], "intermediate")).as(TensorNode);
    g.addEdge(a_row_node, a_row).init(new OnnxEdge.Builder(a_row.literalType, a_row.shape)).as(OnnxEdge);

    // squeeze [1,K] -> [K]
    const a_vec = targetReshape(g, a_row, [K], `a_vec_${op.id}`); // [K]

    const a_pick_node = g.addNode(uniq(g, `a_pick_${op.id}`))
      .init(new OperationNode.Builder("Gather", [a_vec, kU], { axis: 0 })).as(OperationNode);
    const a_scalar = g.addNode(uniq(g, `a_scalar_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(a_pick_node, a_scalar).init(new OnnxEdge.Builder(a_scalar.literalType, a_scalar.shape)).as(OnnxEdge);
    const sq_a = squeezeIfLen1(g, a_scalar, ctx.axes, `a_sq_${op.id}`);

    // ---- B[k,j] as [1] ----
    const b_col_node = g.addNode(uniq(g, `b_col_${op.id}`))
      .init(new OperationNode.Builder("Gather", [rhsTensor, jU], { axis: 1 })).as(OperationNode);
    const b_col = g.addNode(uniq(g, `b_col_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [K, 1], "intermediate")).as(TensorNode);
    g.addEdge(b_col_node, b_col).init(new OnnxEdge.Builder(b_col.literalType, b_col.shape)).as(OnnxEdge);

    // squeeze [K,1] -> [K]
    const b_vec = targetReshape(g, b_col, [K], `b_vec_${op.id}`); // [K]


    const b_pick_node = g.addNode(uniq(g, `b_pick_${op.id}`))
      .init(new OperationNode.Builder("Gather", [b_vec, kU], { axis: 0 })).as(OperationNode);
    const b_scalar = g.addNode(uniq(g, `b_scalar_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(b_pick_node, b_scalar).init(new OnnxEdge.Builder(b_scalar.literalType, b_scalar.shape)).as(OnnxEdge);
    const sq_b = squeezeIfLen1(g, b_scalar, ctx.axes, `b_sq_${op.id}`);

    // ---- prod = A[i,k] * B[k,j]  (scalar) ----
    const mul_node = g.addNode(uniq(g, `mul_${op.id}`))
      .init(new OperationNode.Builder("Mul", [sq_a, sq_b])).as(OperationNode);
    const prod = g.addNode(uniq(g, `prod_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate")).as(TensorNode);
    g.addEdge(mul_node, prod).init(new OnnxEdge.Builder(prod.literalType, prod.shape)).as(OnnxEdge);

    // ---- prev = carry[flat]  (scalar) ----
    const prev_node = g.addNode(uniq(g, `prev_${op.id}`))
      .init(new OperationNode.Builder("GatherElements", [ctx.carry, flatU], { axis: 0 })).as(OperationNode);
    const prev = g.addNode(uniq(g, `prev_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, flatU.shape, "intermediate")).as(TensorNode);
    g.addEdge(prev_node, prev).init(new OnnxEdge.Builder(prev.literalType, prev.shape)).as(OnnxEdge);
    const prev_sq = squeezeIfLen1(g, prev, ctx.axes, `prev_sq_${op.id}`);


    // ---- acc = prev + prod  (scalar) ----
    const add_node = g.addNode(uniq(g, `acc_${op.id}`))
      .init(new OperationNode.Builder("Add", [prev_sq, prod])).as(OperationNode);
    const acc = g.addNode(uniq(g, `acc_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate")).as(TensorNode);
    g.addEdge(add_node, acc).init(new OnnxEdge.Builder(acc.literalType, acc.shape)).as(OnnxEdge);

    ctx.running = acc
    return acc; // [1], to be scattered by the outer builder
  }
}