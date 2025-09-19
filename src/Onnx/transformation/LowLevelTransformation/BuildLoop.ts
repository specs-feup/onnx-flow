/**********************************************************************
 * Build a Loop node (outer-graph) + body graph for a linear chain
 *********************************************************************/
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { DataType, TensorProto } from "../../OnnxTypes.js";
import { bool, int64Vec, scalarInt64, zeroTensor } from "../Utilities.js";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import TransformChain from "./TransformChain.js";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";

const GRAPHS : OnnxGraph.Class[] = [];

type LoopCtx = {
  opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
  iter: TensorNode.Class,
  unsqIdx: TensorNode.Class | null,
  carry: TensorNode.Class,
  axes: TensorNode.Class,
  outShape: number[],
  coalesce: boolean,
  // NEW: indices derived by coalesced MatMul
  iU?: TensorNode.Class | null,
  jU?: TensorNode.Class | null,
  kU?: TensorNode.Class | null,
  flatU?: TensorNode.Class | null,
  kIdx?: TensorNode.Class | null,
  kM1?: TensorNode.Class | null,
  gateByK?: boolean,
  running?: TensorNode.Class | null,
};

/* ------------------------------ Helpers ------------------------------ */

export function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  // ensure uniqueness within the CURRENT graph and across others we’re tracking
  const exists = () => g.hasNode(id) || GRAPHS.some(gr => gr.hasNode(id));
  while (exists()) id = `${base}_${++i}`;
  return id;
}

function makeTensorConst(
  g: OnnxGraph.Class, id: string, dataType: DataType,
  tensorKind: TensorNode.TensorKind, proto: TensorProto
) {
  const builder = tensorKind === "constant" ? new TensorNode.Builder(dataType, proto.dims!, tensorKind, proto) : new TensorNode.Builder(dataType, proto.dims!, tensorKind, undefined, proto);
  return g.addNode(uniq(g, id)).init(builder).as(TensorNode);
}

function gatherFrom(
  g: OnnxGraph.Class, data: TensorNode.Class, tag: string,
  indexNode: OperationNode.Class | TensorNode.Class, axis: number
): [OperationNode.Class, TensorNode.Class] {
  const gather = g.addNode(uniq(g, tag))
    .init(new OperationNode.Builder("Gather", [data, indexNode], { axis }))
    .as(OperationNode);

  const dataShape = data.shape;
  const indexShape = indexNode.is(TensorNode) ? indexNode.shape : []; // fallback if not static

  // Compute: data[:axis] + index + data[axis+1:]
  const outShape = [
    ...dataShape.slice(0, axis),
    ...indexShape,
    ...dataShape.slice(axis + 1),
  ];

  const out = g.addNode(uniq(g, `${tag}_out`))
    .init(new TensorNode.Builder(data.literalType, outShape, 'intermediate'))
    .as(TensorNode);
  g.addEdge(gather, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

  return [gather, out];
}

function gatherFrom2D(
  g: OnnxGraph.Class,
  input: TensorNode.Class,
  rowIdx: TensorNode.Class,
  colIdx: TensorNode.Class,
  tag: string
): [OperationNode.Class, TensorNode.Class] {
  const gather0 = g.addNode(uniq(g, `${tag}_g0`)).init(new OperationNode.Builder("Gather", [input, rowIdx], { axis: 0 })).as(OperationNode);
  const g0Out = g.addNode(uniq(g, `${tag}_g0_out`)).init(new TensorNode.Builder(input.literalType, input.shape.slice(1), "intermediate")).as(TensorNode);
  g.addEdge(gather0, g0Out).init(new OnnxEdge.Builder(g0Out.literalType, g0Out.shape)).as(OnnxEdge);

  const gather1 = g.addNode(uniq(g, `${tag}_g1`)).init(new OperationNode.Builder("Gather", [g0Out, colIdx], { axis: 0 })).as(OperationNode);
  const g1Out = g.addNode(uniq(g, `${tag}_g1_out`)).init(new TensorNode.Builder(input.literalType, [], "intermediate")).as(TensorNode);
  g.addEdge(gather1, g1Out).init(new OnnxEdge.Builder(g1Out.literalType, [])).as(OnnxEdge);

  return [gather1, g1Out];
}

function gatherAt2DPoint(
  g: OnnxGraph.Class,
  input: TensorNode.Class,   // [M,N]
  rowIdx: TensorNode.Class,  // [1]
  colIdx: TensorNode.Class,  // [1]
  tag: string
): [OperationNode.Class, TensorNode.Class] {
  // Gather rows -> [1,N]
  const g0 = g.addNode(uniq(g, `${tag}_g0`))
              .init(new OperationNode.Builder("Gather", [input, rowIdx], { axis: 0 }))
              .as(OperationNode);
  const g0Out = g.addNode(uniq(g, `${tag}_g0_out`))
                 .init(new TensorNode.Builder(input.literalType, [1, input.shape[1]], "intermediate"))
                 .as(TensorNode);
  g.addEdge(g0, g0Out).init(new OnnxEdge.Builder(g0Out.literalType, g0Out.shape)).as(OnnxEdge);

  // Gather cols -> [1, N] gathered along axis 1 with [1] index -> still rank 2
  const g1 = g.addNode(uniq(g, `${tag}_g1`))
              .init(new OperationNode.Builder("Gather", [g0Out, colIdx], { axis: 1 }))
              .as(OperationNode);
  const g1Out = g.addNode(uniq(g, `${tag}_g1_out`))
                 .init(new TensorNode.Builder(input.literalType, [1, input.shape[1]], "intermediate"))
                 .as(TensorNode);
  g.addEdge(g1, g1Out).init(new OnnxEdge.Builder(g1Out.literalType, g1Out.shape)).as(OnnxEdge);

  // Reshape to [1] so updates rank matches indices rank for ScatterElements
  const shape1 = makeTensorConst(g, `${tag}_shape1`, DataType.INT64, "constant", int64Vec([1]));
  const rs = g.addNode(uniq(g, `${tag}_reshape`))
              .init(new OperationNode.Builder("Reshape", [g1Out, shape1]))
              .as(OperationNode);
  const out = g.addNode(uniq(g, `${tag}_out1`))
               .init(new TensorNode.Builder(input.literalType, [1], "intermediate"))
               .as(TensorNode);
  g.addEdge(rs, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

  return [g1, out];
}

function ensureFlatInput(
  g: OnnxGraph.Class, t: TensorNode.Class
): TensorNode.Class {
  const shape = t.shape;
  if (shape.length <= 1) return t;
  const total = shape.reduce((a, d) => a * d, 1);
  const shapeConst = makeTensorConst(g, `flat_shape_${t.id}`, DataType.INT64, "constant", int64Vec([total]));
  const rs = g.addNode(uniq(g, `flat_rs_${t.id}`))
              .init(new OperationNode.Builder("Reshape", [t, shapeConst]))
              .as(OperationNode);
  const flat = g.addNode(uniq(g, `${t.id}_flat`))
                .init(new TensorNode.Builder(t.literalType, [total], "intermediate"))
                .as(TensorNode);
  g.addEdge(rs, flat).init(new OnnxEdge.Builder(t.literalType, [total])).as(OnnxEdge);
  return flat;
}

function divmod(
  g: OnnxGraph.Class, lhs: TensorNode.Class, rhs: TensorNode.Class,
  tag: string, op: "Div" | "Mod"
): TensorNode.Class {
  const node = g.addNode(uniq(g, `${op}_${tag}`))
                .init(new OperationNode.Builder(op, [lhs, rhs]))
                .as(OperationNode);
  const out = g.addNode(uniq(g, `${op}_${tag}_out`))
               .init(new TensorNode.Builder(lhs.literalType, lhs.shape, "intermediate"))
               .as(TensorNode);
  g.addEdge(node, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function unsqueezeIdx(
  g: OnnxGraph.Class, idx: TensorNode.Class, axes: TensorNode.Class, tag: string
): TensorNode.Class {
  const unsq = g.addNode(uniq(g, tag))
                .init(new OperationNode.Builder("Unsqueeze", [idx, axes]))
                .as(OperationNode);
  const out = g.addNode(uniq(g, `${tag}_out`))
               .init(new TensorNode.Builder(idx.literalType, [1], "intermediate"))
               .as(TensorNode);
  g.addEdge(unsq, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function getSmallestRankShape(tensors: TensorNode.Class[]): number[] {
  if (tensors.length === 0) return [];

  let smallest = tensors[0].shape;
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].shape.length < smallest.length) {
      smallest = tensors[i].shape;
    }
  }
  return smallest;
}

function getLargestRankShape(tensors: TensorNode.Class[]): number[] {
  if (tensors.length === 0) return [];
  let largest = tensors[0].shape;
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].shape.length > largest.length) {
      largest = tensors[i].shape;
    }
  }
  return largest;
}

function gatherAndReshape(
  g: OnnxGraph.Class, t: TensorNode.Class, idx: TensorNode.Class,
  axis: number, shape: TensorNode.Class, tag: string
): TensorNode.Class {
  const [_, gathered] = gatherFrom(g, t, tag, idx, axis);
  const reshape = g.addNode(uniq(g, `${tag}_reshape`))
                   .init(new OperationNode.Builder("Reshape", [gathered, shape]))
                   .as(OperationNode);
  const out = g.addNode(uniq(g, `${tag}_reshaped_out`))
               .init(new TensorNode.Builder(t.literalType, [shape.shape[0]], "intermediate"))
               .as(TensorNode);
  g.addEdge(reshape, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function targetReshape(
  g: OnnxGraph.Class,
  tensor: TensorNode.Class,
  targetShape: number[],
  tag: string
): TensorNode.Class {
  const actualShape = tensor.shape;

  // Check if shape is already correct
  const isSame = actualShape.length === targetShape.length &&
                 actualShape.every((d, i) => d === targetShape[i]);
  //console.log("SHAPES:", actualShape, targetShape, isSame);
  if (isSame || targetShape.length == 0) return tensor;

  // Create shape constant
  const shapeConst = makeTensorConst(g, `fixshape_${tag}`, DataType.INT64, "constant", int64Vec(targetShape));

  // Create reshape op
  const reshape = g.addNode(uniq(g, `reshape_${tag}`))
                   .init(new OperationNode.Builder("Reshape", [tensor, shapeConst]))
                   .as(OperationNode);

  const out = g.addNode(uniq(g, `reshaped_${tag}`))
               .init(new TensorNode.Builder(tensor.literalType, targetShape, "intermediate"))
               .as(TensorNode);

  g.addEdge(reshape, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

  return out;
}

function reshapeTensor(
  g: OnnxGraph.Class,
  input: TensorNode.Class,
  shape: TensorNode.Class,
  tag: string
): TensorNode.Class {
  const reshape = g.addNode(uniq(g, `reshape_${tag}`))
                   .init(new OperationNode.Builder("Reshape", [input, shape]))
                   .as(OperationNode);
  const out = g.addNode(uniq(g, `reshaped_${tag}`))
               .init(new TensorNode.Builder(input.literalType, shape.shape, "intermediate"))
               .as(TensorNode);
  g.addEdge(reshape, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

function resolveFusedInput(
  g: OnnxGraph.Class,
  input: BaseNode.Class,
  ctx: LoopCtx,
  op: OperationNode.Class,
  flatten: boolean = true,
  returnGather: boolean = true
): TensorNode.Class {
  // 1) If producer is fused, reuse its fused output
  if (input.is(TensorNode)) {
    const t = input.as(TensorNode);

    if (t.type === "intermediate" && t.getIncomers.length > 0) {
      const producer = t.getIncomers[0].source;
      if (producer.is(OperationNode)) {
        const fused = Array.from(ctx.opMap.entries()).find(([key]) => key.id === producer.id);
        if (fused) return fused[1][1];
      }
    }

    // 2) Otherwise, we gather by *the correct* index for t.shape
    //    Prefer (iU, jU) / (flatU) if coalesced MatMul provided them
    let idxToUse: TensorNode.Class | null = ctx.unsqIdx; // default

    const [M, N] = ctx.outShape.length === 2 ? ctx.outShape : [undefined, undefined];

    if (ctx.coalesce && (ctx.iU || ctx.jU || ctx.flatU)) {
      const s = t.shape;

      if (s.length === 0) {
        // scalar: no gather needed
        return t;
      }

      if (s.length === 1) {
        // length-1D: choose j for [N], i for [M]
        const len = s[0];
        if (N !== undefined && len === N && ctx.jU) idxToUse = ctx.jU;
        else if (M !== undefined && len === M && ctx.iU) idxToUse = ctx.iU;
        else if (ctx.flatU) idxToUse = ctx.flatU; // fallback (may be risky if len != M*N)
        // no flatten needed; already 1D
        if (!returnGather) return t;
        const [_, gathered] = gatherFrom(g, t, `gather_${t.id}_${op.id}`, idxToUse!, 0);
        return gathered;
      }

      if (s.length === 2) {
        // [M,N]: best to address with (i,j)
        if (ctx.iU && ctx.jU) {
          const [_, picked] = gatherAt2DPoint(g, t, ctx.iU!, ctx.jU!, `g2d_${t.id}_${op.id}`);
          return picked;
        }
        // fallback: flatten to its own length and use flat index
        const flatT = ensureFlatInput(g, t);
        const idx = ctx.flatU ?? idxToUse!;
        if (!returnGather) return flatT;
        const [__, gathered] = gatherFrom(g, flatT, `gather_${t.id}_${op.id}`, idx, 0);
        return gathered;
      }
    }

    // Non-coalesced path (or no special indices available)
    if (!ctx.unsqIdx) {
      const unsq = g.addNode(uniq(g, "unsq_idx"))
        .init(new OperationNode.Builder("Unsqueeze", [ctx.iter, ctx.axes]))
        .as(OperationNode);
      const unsqOut = g.addNode(uniq(g, "unsq_idx_out"))
        .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
        .as(TensorNode);
      g.addEdge(unsq, unsqOut).init(new OnnxEdge.Builder(unsqOut.literalType, unsqOut.shape)).as(OnnxEdge);
      ctx.unsqIdx = unsqOut;
    }

    const gatherInput = flatten ? ensureFlatInput(g, t) : t;
    if (!returnGather) return gatherInput;
    const [__, gathered] = gatherFrom(g, gatherInput, `gather_${t.id}_${op.id}`, ctx.unsqIdx, 0);
    g.addEdge(ctx.unsqIdx, __ as any).init(new OnnxEdge.Builder(ctx.unsqIdx.literalType, ctx.unsqIdx.shape)).as(OnnxEdge);
    return gathered;
  }

  // 3) If the input is an op and it's fused, return fused out
  if (input.is(OperationNode)) {
    const fused = Array.from(ctx.opMap.entries()).find(([key]) => key.id === input.id);
    if (fused) return fused[1][1];
  }

  throw new Error(`Unhandled input case in resolveFusedInput for ${input.id}`);
}




/* ------------------- Handlers for Operation Types ------------------- */

function handleSimpleArithmeticOperation(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx
): TensorNode.Class {
  const inputs = op.getInputs()!.map(inp => resolveFusedInput(g, inp, ctx, op));

  const node = g.addNode(uniq(g, `${op.type}_${op.id}`))
                .init(new OperationNode.Builder(op.type, inputs))
                .as(OperationNode);

  const outShape = getLargestRankShape(inputs);
  const out = g.addNode(uniq(g, `${op.id}_out`))
               .init(new TensorNode.Builder(inputs[0].literalType, outShape, "intermediate"))
               .as(TensorNode);

  g.addEdge(node, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

  // Gate ONLY when we’re in a coalesced + fused chain
  if (ctx.coalesce && ctx.gateByK && ctx.kIdx && ctx.kM1) {
    const eqNode = g.addNode(uniq(g, `eq_k_last_${op.id}`))
                    .init(new OperationNode.Builder("Equal", [ctx.kIdx, ctx.kM1]))
                    .as(OperationNode);
    const eqOut = g.addNode(uniq(g, `eq_k_last_${op.id}_out`))
                   .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
                   .as(TensorNode);
    g.addEdge(eqNode, eqOut).init(new OnnxEdge.Builder(eqOut.literalType, eqOut.shape)).as(OnnxEdge);

    const passthrough = ctx.running ?? inputs[0];

    // Where(eq, applied_out, passthrough_left_input)
    const whereNode = g.addNode(uniq(g, `gate_${op.type}_${op.id}`))
                      .init(new OperationNode.Builder("Where", [eqOut, out, passthrough]))
                      .as(OperationNode);
    const gated = g.addNode(uniq(g, `gated_${op.id}`))
                   .init(new TensorNode.Builder(passthrough.literalType, outShape, "intermediate"))
                   .as(TensorNode);
    g.addEdge(whereNode, gated).init(new OnnxEdge.Builder(gated.literalType, gated.shape)).as(OnnxEdge);

    return gated;
  }

  return out;
}


function handleMatMul(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx
): TensorNode.Class {
  const lhsInput = op.getInputs()![0];
  const rhsInput = op.getInputs()![1];

  const lhsTensor = resolveFusedInput(g, lhsInput, ctx, op, false, false);
  const rhsTensor = resolveFusedInput(g, rhsInput, ctx, op, false, false);

  const K = lhsTensor.shape.at(-1)!;
  const N = rhsTensor.shape.at(-1)!;

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
      .init(new TensorNode.Builder(elemTy, [], "intermediate")).as(TensorNode);
    g.addEdge(a_pick_node, a_scalar).init(new OnnxEdge.Builder(a_scalar.literalType, a_scalar.shape)).as(OnnxEdge);

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
      .init(new TensorNode.Builder(elemTy, [], "intermediate")).as(TensorNode);
    g.addEdge(b_pick_node, b_scalar).init(new OnnxEdge.Builder(b_scalar.literalType, b_scalar.shape)).as(OnnxEdge);

    // ---- prod = A[i,k] * B[k,j]  (shape [1]) ----
    const mul_node = g.addNode(uniq(g, `mul_${op.id}`))
      .init(new OperationNode.Builder("Mul", [a_scalar, b_scalar])).as(OperationNode);
    const prod = g.addNode(uniq(g, `prod_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [], "intermediate")).as(TensorNode);
    g.addEdge(mul_node, prod).init(new OnnxEdge.Builder(prod.literalType, prod.shape)).as(OnnxEdge);

    // ---- prev = carry[flat]  (shape [1]) ----
    const prev_node = g.addNode(uniq(g, `prev_${op.id}`))
      .init(new OperationNode.Builder("GatherElements", [ctx.carry, flatU], { axis: 0 })).as(OperationNode);
    const prev = g.addNode(uniq(g, `prev_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, flatU.shape, "intermediate")).as(TensorNode);
    g.addEdge(prev_node, prev).init(new OnnxEdge.Builder(prev.literalType, prev.shape)).as(OnnxEdge);

    // ---- acc = prev + prod  (shape [1]) ----
    const add_node = g.addNode(uniq(g, `acc_${op.id}`))
      .init(new OperationNode.Builder("Add", [prev, prod])).as(OperationNode);
    const acc = g.addNode(uniq(g, `acc_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(add_node, acc).init(new OnnxEdge.Builder(acc.literalType, acc.shape)).as(OnnxEdge);

    ctx.running = acc;
    return acc; // [1], to be scattered by the outer builder
  }
}



export {
  handleSimpleArithmeticOperation,
  handleMatMul
};


/* Main Function */

export function buildLoopForChain(
  chain: OperationNode.Class[],
  graph: OnnxGraph.Class,
  fuse: boolean = true,
  recurse: boolean = true,
  coalesce: boolean = true
): void {
  GRAPHS.push(graph);

  const matmulOp = chain.find(op => op.type === "MatMul");
  const includesCoalescedMatMul = coalesce && matmulOp;
  const matmulIndex = chain.findIndex(op => op.type === "MatMul");
  const needsGating =
    coalesce &&
    matmulIndex !== -1 &&
    chain.slice(matmulIndex + 1).some(op =>
      op.type === "Add" || op.type === "Sub" || op.type === "Mul" || op.type === "Div"
    );
  const lastOp = chain.at(-1)!;
  const outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();

  const elemTy = outTensor.literalType === DataType.UNDEFINED
    ? lastOp.getOutgoers.first().literalType
    : outTensor.literalType;
  let outShape = outTensor.shape.length === 0
    ? lastOp.getOutgoers.first().shape
    : outTensor.shape;

  let totalIters: number;
  let carryLen: number;
  if (includesCoalescedMatMul) {
    const lhs = matmulOp.getInputs()![0].as(TensorNode);
    const rhs = matmulOp.getInputs()![1].as(TensorNode);
    const M = lhs.shape.at(0)!;
    const K = lhs.shape.at(1)!;
    const N = rhs.shape.at(1)!;

    totalIters = M * K * N;
    outShape = [M, N];  // final carry shape
    carryLen = outShape[0] * outShape[1];
  } else {
    totalIters = outShape.length <= 1 ? outShape[0] ?? 1 : outShape.reduce((a, b) => a * b, 1);
    carryLen = totalIters;
  }

  const inputs = new Map<string, TensorNode.Class>();
  chain.forEach(op =>
    op.getInputs()?.filter(n => n.is(TensorNode)).forEach(t => inputs.set(t.id, t.as(TensorNode)))
  );

  const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
  GRAPHS.push(body);
  const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
  const condIn = body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode);
  const carryInit = zeroTensor(elemTy, [carryLen]);
  const carry = body.addNode(uniq(body, "carry")).init(new TensorNode.Builder(elemTy, [carryLen], "input", carryInit)).as(TensorNode);

  const axes = makeTensorConst(body, "axes", DataType.INT64, "constant", int64Vec([0]));
  let unsqOut = null;

  if (!includesCoalescedMatMul) {
    const unsq = body.addNode(uniq(body, "unsq"))
    .init(new OperationNode.Builder("Unsqueeze", [iter, axes]))
    .as(OperationNode);

    unsqOut = body.addNode(uniq(body, "unsq_out"))
      .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
      .as(TensorNode);
    body.addEdge(unsq, unsqOut).init(new OnnxEdge.Builder(unsqOut.literalType, unsqOut.shape)).as(OnnxEdge);
  }

  let indicesOut = unsqOut;

  const opMap = new Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>();

  const handlers: Record<string, typeof handleSimpleArithmeticOperation> = {
    Add: handleSimpleArithmeticOperation,
    Sub: handleSimpleArithmeticOperation,
    Mul: handleSimpleArithmeticOperation,
    Div: handleSimpleArithmeticOperation,
    MatMul: handleMatMul,
  };

  const ctx : LoopCtx = {
    opMap,
    iter,
    unsqIdx: unsqOut,
    carry,
    axes,
    outShape,
    coalesce,
    iU: null, jU: null, kU: null, flatU: null, kIdx: null, kM1: null,
    gateByK: needsGating,
    running: null,
  };

  for (const op of chain) {
    const handler = handlers[op.type];
    if (!handler) throw new Error(`Unsupported op: ${op.type}`);
    
    const output = handler(op, body, ctx);
    if (coalesce && op.type === "MatMul") {
      indicesOut = ctx.flatU ?? ctx.unsqIdx!;
    }
    opMap.set(op, [op, output]);
  }

  const lastOut = opMap.get(lastOp)![1];
  

  const scatter = body.addNode(uniq(body, "scatter"))
    .init(new OperationNode.Builder("ScatterElements", [carry, indicesOut, lastOut], { axis: 0 }))
    .as(OperationNode);

  body.addEdge(carry, scatter).init(new OnnxEdge.Builder(carry.literalType, carry.shape));
  body.addEdge(indicesOut, scatter).init(new OnnxEdge.Builder(indicesOut.literalType, indicesOut.shape));
  body.addEdge(lastOut, scatter).init(new OnnxEdge.Builder(lastOut.literalType, lastOut.shape));

  /* cond passthrough */
  const idCond = body.addNode(uniq(body, "id_cond"))
    .init(new OperationNode.Builder("Identity", [condIn]))
    .as(OperationNode);
  const condOut = body.addNode(uniq(body, "cond_out"))
    .init(new TensorNode.Builder(DataType.BOOL, [], "output"))
    .as(TensorNode);
  body.addEdge(condIn, idCond).init(new OnnxEdge.Builder(condIn.literalType, condIn.shape));
  body.addEdge(idCond, condOut).init(new OnnxEdge.Builder(condOut.literalType, condOut.shape));

  const carryOut = body.addNode(uniq(body, "carry_out"))
    .init(new TensorNode.Builder(elemTy, carry.shape, "output"))
    .as(TensorNode);
  body.addEdge(scatter, carryOut).init(new OnnxEdge.Builder(carryOut.literalType, carryOut.shape));

  inferShapes(graph);
  inferShapes(body);

  if (recurse) {
    const recursiveDecomposer = new TransformChain(fuse, recurse);
    recursiveDecomposer.apply(body);
  }
  /* ---------- outer Loop node + wiring -------------------------------- */

  /* ensure global trip_count / cond exist                                */  
  const trip = makeTensorConst(graph, `trip_count_${chain[0].id}`, DataType.INT64, "constant", scalarInt64(totalIters));
  const cond = makeTensorConst(graph, `cond_${chain[0].id}`, DataType.BOOL, "constant", bool(true));


  const v_initial = makeTensorConst(graph, "init_carry", DataType.FLOAT, "initializer", carryInit);
  const loop = graph.addNode(uniq(graph, `Loop_${chain[0].id}`))
                    .init(new OperationNode.Builder("Loop", [trip, cond, v_initial], {}, body))
                    .as(OperationNode);

  graph.addEdge(trip, loop).init(new OnnxEdge.Builder(trip.literalType, trip.shape)).as(OnnxEdge);
  graph.addEdge(cond, loop).init(new OnnxEdge.Builder(cond.literalType, cond.shape)).as(OnnxEdge);

  /* wire original model inputs as scan inputs                            */
  inputs.forEach(t => {
    graph.addEdge(t, loop).init(new OnnxEdge.Builder(t.literalType, t.shape)).as(OnnxEdge);
  });

  /* replace outgoing connections                                         */
  chain[chain.length - 1].getOutgoers.forEach(e => e.remove());

  const isGlobalOutput = graph.getOutputTensorNodes().contains(outTensor);
  graph.getNodeById(outTensor.id).remove();
  graph.addNode(outTensor.id).init(new TensorNode.Builder(elemTy, outShape, isGlobalOutput ? "output" : "intermediate")).as(TensorNode);

  if (outShape.length > 1) {
    const loop_out = graph.addNode(uniq(graph, "loop_out")).init(new TensorNode.Builder(elemTy, carry.shape, 'intermediate')).as(TensorNode);
    graph.addEdge(loop, loop_out).init(new OnnxEdge.Builder(loop_out.literalType, loop_out.shape)).as(OnnxEdge);
    
    const shapeProto = int64Vec(outShape);
    const shapeNode  = graph.addNode(uniq(graph, `reshape_shape_${chain[0].id}`))
                            .init(new TensorNode.Builder(
                                  DataType.INT64, [outShape.length],
                                  "constant", shapeProto))
                            .as(TensorNode);
                
    const reshape = graph.addNode(uniq(graph, `reshape_${chain[0].id}`))
                         .init(new OperationNode.Builder("Reshape",[loop_out, shapeNode]))
                         .as(OperationNode);
    
    graph.addEdge(loop_out, reshape).init(new OnnxEdge.Builder(loop_out.literalType, loop_out.shape)).as(OnnxEdge);

    graph.addEdge(shapeNode, reshape).init(new OnnxEdge.Builder(shapeNode.literalType, shapeNode.shape)).as(OnnxEdge);
    graph.addEdge(reshape, outTensor)
         .init(new OnnxEdge.Builder(elemTy, outShape)).as(OnnxEdge);
  } else {
    graph.addEdge(loop, outTensor)
         .init(new OnnxEdge.Builder(elemTy, outShape)).as(OnnxEdge);
  }

  /* finally, remove the original ops & dangling tensors                  */
  chain.forEach(op => op.remove());
}
