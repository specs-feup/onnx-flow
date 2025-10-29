/**********************************************************************
 * Build a Loop node (outer-graph) + body graph for a linear chain
 *********************************************************************/
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { DataType, TensorProto } from "../../OnnxTypes.js";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import TransformChain from "./TransformChain.js";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import { scalarInt64, uniq, int64Vec, zeroTensor, bool, getLargestRankShape, Shape, asStaticDims, makeTensorConst, computeStrides, isNum, toStaticShape } from "../../Utils.js";
import handleElementWiseOperation from "./handlers/ElementWiseOperations.js";
import handleMatMul from "./handlers/MatMul.js";
import handleTranspose from "./handlers/Transpose.js";
import handleRange from "./handlers/Range.js";

const GRAPHS: OnnxGraph.Class[] = [];

export type LoopCtx = {
  opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
  iter: TensorNode.Class,
  unsqIdx: TensorNode.Class | null,
  carry: TensorNode.Class,
  axes: TensorNode.Class,
  outShape: (number | String)[],
  coalesce: boolean,
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

// Mixed-radix decode of ctx.iter into digits along 'dims' (rightmost fastest)
export function decodeMixedRadix(
  g: OnnxGraph.Class,
  iter: TensorNode.Class,
  dims: number[],
  tag: string
): TensorNode.Class[] {
  // We allow unknown dims (-1) by replacing them with 1 for decoding
  const dd = dims.map(d => (d > 0 ? d : 1));
  const out: TensorNode.Class[] = [];
  let rem = iter;

  for (let k = dd.length - 1; k >= 0; k--) {
    const dConst = makeTensorConst(g, `mr_dim_${tag}_${k}`, DataType.INT64, "constant", scalarInt64(dd[k]));
    const modN = g.addNode(uniq(g, `mr_mod_${tag}_${k}`))
                  .init(new OperationNode.Builder("Mod", [rem, dConst]))
                  .as(OperationNode);
    const modOut = g.addNode(uniq(g, `mr_mod_out_${tag}_${k}`))
                    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
                    .as(TensorNode);
    g.addEdge(modN, modOut).init(new OnnxEdge.Builder(modOut.literalType, modOut.shape)).as(OnnxEdge);
    out.unshift(modOut);

    const divN = g.addNode(uniq(g, `mr_div_${tag}_${k}`))
                  .init(new OperationNode.Builder("Div", [rem, dConst]))
                  .as(OperationNode);
    const divOut = g.addNode(uniq(g, `mr_div_out_${tag}_${k}`))
                    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
                    .as(TensorNode);
    g.addEdge(divN, divOut).init(new OnnxEdge.Builder(divOut.literalType, divOut.shape)).as(OnnxEdge);
    rem = divOut;
  }
  return out; // one INT64 scalar per axis
}

// sum_i idx[i] * stride[i]
export function buildLinearIndex(
  g: OnnxGraph.Class,
  idx: TensorNode.Class[],
  strides: number[],
  tag: string
): TensorNode.Class {
  let acc = makeTensorConst(g, `lin_zero_${tag}`, DataType.INT64, "constant", scalarInt64(0));
  for (let i = 0; i < idx.length; i++) {
    const sConst = makeTensorConst(g, `lin_stride_${tag}_${i}`, DataType.INT64, "constant", scalarInt64(strides[i]));
    const mulN = g.addNode(uniq(g, `lin_mul_${tag}_${i}`))
                  .init(new OperationNode.Builder("Mul", [idx[i], sConst]))
                  .as(OperationNode);
    const mulOut = g.addNode(uniq(g, `lin_mul_out_${tag}_${i}`))
                    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
                    .as(TensorNode);
    g.addEdge(mulN, mulOut).init(new OnnxEdge.Builder(mulOut.literalType, mulOut.shape)).as(OnnxEdge);

    const addN = g.addNode(uniq(g, `lin_add_${tag}_${i}`))
                  .init(new OperationNode.Builder("Add", [acc, mulOut]))
                  .as(OperationNode);
    const addOut = g.addNode(uniq(g, `lin_add_out_${tag}_${i}`))
                    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
                    .as(TensorNode);
    g.addEdge(addN, addOut).init(new OnnxEdge.Builder(addOut.literalType, addOut.shape)).as(OnnxEdge);
    acc = addOut;
  }
  return acc;
}

// True broadcast result for element-wise ops (numeric dims only; -1 passes through)
export function broadcastShapes(shapes: number[][]): number[] {
  if (shapes.length === 0) return [];
  const maxR = Math.max(...shapes.map(s => s.length));
  const out = Array(maxR).fill(1);
  for (let i = 0; i < maxR; i++) {
    let dim = 1;
    for (const s of shapes) {
      const d = s[s.length - 1 - i] ?? 1;
      if (d === -1) { dim = dim === 1 ? -1 : dim; continue; }
      if (dim === 1) dim = d;
      else if (d === 1) continue;
      else if (dim === -1) dim = d;
      else if (dim !== d) throw new Error(`Broadcast mismatch on axis -${i+1}: got ${dim} vs ${d}`);
    }
    out[maxR - 1 - i] = dim;
  }
  return out;
}

export function getMatDims(aShape: (number|string)[], bShape: (number|string)[]) {
  // Accept [M,K]·[K,N], also rank-1 vector cases promoted by ONNX
  const a = asStaticDims(aShape);
  const b = asStaticDims(bShape);
  // Promote vectors to 2D: [K] -> [K,1] if RHS; [K] -> [1,K] if LHS
  let aR = a.length, bR = b.length;
  let A = a.slice(), B = b.slice();
  if (aR === 1) { A = [1, a[0]]; aR = 2; }
  if (bR === 1) { B = [b[0], 1]; bR = 2; }
  const M = A[A.length - 2];
  const K = A[A.length - 1];
  const KN = B[B.length - 2]; // must equal K
  const N = B[B.length - 1];
  return { M, K, KN, N, A2: A, B2: B, aWasVec: a.length === 1, bWasVec: b.length === 1 };
}

export function gatherFrom(
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

export function gatherFrom2D(
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

export function gatherAt2DPoint(
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

// Gather one element from 't' following ONNX broadcast rules against ctx.outShape
export function gatherWithBroadcast(
  g: OnnxGraph.Class,
  t: TensorNode.Class,
  ctx: LoopCtx,
  tag: string
): TensorNode.Class {
  // Scalars don’t need gathering (broadcast naturally)
  if (t.shape.length === 0) return t;

  const outDimsStatic = toStaticShape(ctx.outShape as Shape);   // may contain -1
  const inDims        = toStaticShape(t.shape as Shape).map(d => (d > 0 ? d : 1));

  // --- 1-D unknown-length fast path (Range + Add, etc.) ---
  if (inDims.length === 1 && outDimsStatic.length === 1 && outDimsStatic[0] <= 0) {
    // Just use iter as the index; Range's trip_count guarantees in-bounds
    const idx = ctx.unsqIdx ?? unsqueezeIdx(g, ctx.iter, ctx.axes, `unsq_idx_${tag}`);
    const src = /* no need to flatten a 1-D input */ t;
    const [__, gathered] = gatherFrom(g, src, `gb_1d_iter_${tag}`, idx, 0);
    return gathered; // shape [1]
  }

  // If input rank > output rank, fallback to flat gather
  if (inDims.length > outDimsStatic.length) {
    const src = ensureFlatInput(g, t);
    const idx = ctx.unsqIdx ?? unsqueezeIdx(g, ctx.iter, ctx.axes, `unsq_idx_${tag}`);
    const [__, gathered] = gatherFrom(g, src, `gb_fallback_${tag}`, idx, 0);
    return gathered; // [1]
  }

  // Build the decode radix per axis:
  // Use output dim if known; if unknown (-1), substitute this input’s non-1 dim for that aligned axis (else 1).
  const rO = outDimsStatic.length, rI = inDims.length;
  const decodeDims: number[] = outDimsStatic.map(d => (d > 0 ? d : 1));
  for (let k = 0; k < rI; k++) {
    const outPos = rO - rI + k;
    if (outPos >= 0 && outDimsStatic[outPos] <= 0) {
      // unknown out dim → pick the concrete size from this input axis if >1
      if (inDims[k] > 1) decodeDims[outPos] = inDims[k];
    }
  }

  // Decode output digits once
  const oDigits = decodeMixedRadix(g, ctx.iter, decodeDims, `gb_${tag}`);

  // Align from the right for broadcast: input axis uses 0 if its dim is 1; otherwise reuse the output digit
  const iDigits: TensorNode.Class[] = [];
  for (let k = 0; k < rI; k++) {
    const outPos = rO - rI + k;
    const inDim  = inDims[k];
    if (inDim === 1) {
      const z = makeTensorConst(g, `gb_zero_${tag}_${k}`, DataType.INT64, "constant", scalarInt64(0));
      iDigits.push(z);
    } else {
      iDigits.push(oDigits[outPos]);
    }
  }

  // Build linear index and gather from a flattened view
  const strides = computeStrides(inDims);
  const lin = buildLinearIndex(g, iDigits, strides, `gb_lin_${tag}`);
  const linU = unsqueezeIdx(g, lin, ctx.axes, `gb_linU_${tag}`);

  const flat = ensureFlatInput(g, t);
  const [_, gathered] = gatherFrom(g, flat, `gb_g_${tag}`, linU, 0); // [1]
  return gathered;
}

export function squeezeIfLen1(g: OnnxGraph.Class, t: TensorNode.Class, axes: TensorNode.Class, tag: string) {
  if (t.shape.length === 1 && t.shape[0] === 1) {
    const sq = g.addNode(uniq(g, `sq_${tag}`))
      .init(new OperationNode.Builder("Squeeze", [t, axes]))
      .as(OperationNode);
    const out = g.addNode(uniq(g, `sq_${tag}_out`))
      .init(new TensorNode.Builder(t.literalType, [], "intermediate"))
      .as(TensorNode);
    g.addEdge(sq, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
  }
  return t;
}

export function ensureFlatInput(
  g: OnnxGraph.Class, t: TensorNode.Class
): TensorNode.Class {
  const shape: Shape = t.shape;
  if (shape.length <= 1) return t;

  const allNum = shape.every(isNum);
  const total = allNum
    ? (shape as number[]).reduce((a, d) => a * d, 1)
    : -1; // unknown size → use -1

  // Reshape to [total] ([-1] when dynamic)
  const shpVec = int64Vec([total]);
  const shapeConst = makeTensorConst(g, `flat_shape_${t.id}`, DataType.INT64, "constant", shpVec);

  const rs = g.addNode(uniq(g, `flat_rs_${t.id}`))
    .init(new OperationNode.Builder("Reshape", [t, shapeConst]))
    .as(OperationNode);

  const outStatic = [total]; // number[]
  const flat = g.addNode(uniq(g, `${t.id}_flat`))
    .init(new TensorNode.Builder(t.literalType, outStatic, "intermediate"))
    .as(TensorNode);

  g.addEdge(rs, flat).init(new OnnxEdge.Builder(t.literalType, outStatic)).as(OnnxEdge);
  return flat;
}

export function divmod(
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

export function unsqueezeIdx(
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

export function gatherAndReshape(
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

export function targetReshape(
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

export function reshapeTensor(
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

export function resolveFusedInput(
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

    if (!returnGather) {
      return flatten ? ensureFlatInput(g, t) : t;
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

    // Non-coalesced path (or no special indices available) — broadcast-aware
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

    // If someone asked for the raw tensor (e.g., to reshape), honor it.
    if (!returnGather) return flatten ? ensureFlatInput(g, t) : t;

    // Broadcast-correct element gather for this input
    return gatherWithBroadcast(g, t, ctx, `${t.id}_${op.id}`);
  }

  // 3) If the input is an op and it's fused, return fused out
  if (input.is(OperationNode)) {
    const fused = Array.from(ctx.opMap.entries()).find(([key]) => key.id === input.id);
    if (fused) return fused[1][1];
  }

  throw new Error(`Unhandled input case in resolveFusedInput for ${input.id}`);
}


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
  const rangeOp = chain.find(op => op.type === "Range");

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
  
  if (rangeOp) {
    // mark unknown 1-D length
    outShape = [undefined];  
  }

  let totalIters: number = null;
  let carryLen: number = null;
  if (includesCoalescedMatMul) {
    const lhs = matmulOp.getInputs()![0].as(TensorNode);
    const rhs = matmulOp.getInputs()![1].as(TensorNode);
    const M = Number(lhs.shape.at(0)!);
    const K = Number(lhs.shape.at(1)!);
    const N = Number(rhs.shape.at(1)!);

    totalIters = M * K * N;
    outShape = [M, N];  // final carry shape
    carryLen = M * N;
  } else {
    const staticOut = toStaticShape(outShape as Shape);
    totalIters = staticOut.length <= 1 ? (staticOut[0] ?? 1) : staticOut.reduce((a, b) => a * b, 1);
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

  let carryInit = null;
  let carry = null;

  // Determine if the first dim is dynamic (string or -1)
  const firstDim = outShape[0];
  const isDynamicLen =
    !!rangeOp ||
    firstDim === undefined ||
    typeof firstDim === 'string' ||
    (typeof firstDim === 'number' && firstDim < 0);

  if (isDynamicLen) {
    // Body carry: unknown length → declare as input of shape [-1], no zero initializer
    carry = body.addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [-1], "input" /* no initializer */))
      .as(TensorNode);
  } else {
    // Known length → we can safely allocate a zero initializer
    carryInit = zeroTensor(elemTy, [carryLen]);
    carry = body.addNode(uniq(body, "carry"))
      .init(new TensorNode.Builder(elemTy, [carryLen], "input", carryInit))
      .as(TensorNode);
  }

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

  const handlers: Record<string, typeof handleElementWiseOperation> = {
    Add: handleElementWiseOperation,
    Sub: handleElementWiseOperation,
    Mul: handleElementWiseOperation,
    Div: handleElementWiseOperation,
    Relu: handleElementWiseOperation,
    Sum: handleElementWiseOperation,
    Min: handleElementWiseOperation,
    Max: handleElementWiseOperation,
    Sigmoid: handleElementWiseOperation, 
    Tanh: handleElementWiseOperation, 
    Exp: handleElementWiseOperation,
    MatMul: handleMatMul,
    Transpose: handleTranspose,
    Range: handleRange
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

  let lastOut = opMap.get(lastOp)![1];
  if(lastOut.shape.length === 0){
    lastOut = unsqueezeIdx(body, lastOut, ctx.axes, "updateUnsq");
  }

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


  // ---- Build Loop inputs (trip_count, cond, v_initial) in the OUTER graph ----
  let trip: TensorNode.Class;        // INT64 scalar [] → Loop input 0
  let cond: TensorNode.Class;        // BOOL  scalar [] → Loop input 1
  let v_initial: TensorNode.Class;   // elemTy [1]     → Loop input 2


  if (rangeOp) {
    const [startT, limitT, deltaT] = rangeOp.getInputs()!.map(n => n.as(TensorNode));

    // 1) sub = (limit - start) : same dtype as inputs, scalar []
    const subN = graph.addNode(uniq(graph, `range_sub_${chain[0].id}`))
                      .init(new OperationNode.Builder("Sub", [limitT, startT]))
                      .as(OperationNode);
    const subOut = graph.addNode(uniq(graph, `range_sub_out_${chain[0].id}`))
                        .init(new TensorNode.Builder(startT.literalType, [], "intermediate"))
                        .as(TensorNode);
    graph.addEdge(subN, subOut).init(new OnnxEdge.Builder(subOut.literalType, subOut.shape)).as(OnnxEdge);

    // 2) Cast sub, delta → FLOAT
    const subCastN = graph.addNode(uniq(graph, `range_subF_${chain[0].id}`))
                          .init(new OperationNode.Builder("Cast", [subOut], { to: DataType.FLOAT }))
                          .as(OperationNode);
    const subF = graph.addNode(uniq(graph, `range_subF_out_${chain[0].id}`))
                      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
                      .as(TensorNode);
    graph.addEdge(subCastN, subF).init(new OnnxEdge.Builder(subF.literalType, subF.shape)).as(OnnxEdge);

    const deltaCastN = graph.addNode(uniq(graph, `range_deltaF_${chain[0].id}`))
                            .init(new OperationNode.Builder("Cast", [deltaT], { to: DataType.FLOAT }))
                            .as(OperationNode);
    const deltaF = graph.addNode(uniq(graph, `range_deltaF_out_${chain[0].id}`))
                        .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
                        .as(TensorNode);
    graph.addEdge(deltaCastN, deltaF).init(new OnnxEdge.Builder(deltaF.literalType, deltaF.shape)).as(OnnxEdge);

    // 3) divF = subF / deltaF → FLOAT []
    const divN = graph.addNode(uniq(graph, `range_divF_${chain[0].id}`))
                      .init(new OperationNode.Builder("Div", [subF, deltaF]))
                      .as(OperationNode);
    const divF = graph.addNode(uniq(graph, `range_divF_out_${chain[0].id}`))
                      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
                      .as(TensorNode);
    graph.addEdge(divN, divF).init(new OnnxEdge.Builder(divF.literalType, divF.shape)).as(OnnxEdge);

    // 4) ceilF = Ceil(divF) ; maxF = Max(ceilF, 0.0)
    const ceilN = graph.addNode(uniq(graph, `range_ceilF_${chain[0].id}`))
                      .init(new OperationNode.Builder("Ceil", [divF]))
                      .as(OperationNode);
    const ceilF = graph.addNode(uniq(graph, `range_ceilF_out_${chain[0].id}`))
                      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
                      .as(TensorNode);
    graph.addEdge(ceilN, ceilF).init(new OnnxEdge.Builder(ceilF.literalType, ceilF.shape)).as(OnnxEdge);

    const zeroF = makeTensorConst(
      graph, `range_zeroF_${chain[0].id}`, DataType.FLOAT, "constant",
      { dataType: DataType.FLOAT, dims: [], floatData: [0] } as TensorProto
    );

    const maxN = graph.addNode(uniq(graph, `range_maxF_${chain[0].id}`))
                      .init(new OperationNode.Builder("Max", [ceilF, zeroF]))
                      .as(OperationNode);
    const maxF = graph.addNode(uniq(graph, `range_maxF_out_${chain[0].id}`))
                      .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
                      .as(TensorNode);
    graph.addEdge(maxN, maxF).init(new OnnxEdge.Builder(maxF.literalType, maxF.shape)).as(OnnxEdge);

    // 5) tripScalar = Cast(maxF) → INT64 scalar []; tripVec = Unsqueeze([0]) → INT64[1]
    const tripCastN = graph.addNode(uniq(graph, `range_trip_${chain[0].id}`))
                          .init(new OperationNode.Builder("Cast", [maxF], { to: DataType.INT64 }))
                          .as(OperationNode);
    const tripScalar = graph.addNode(uniq(graph, `range_trip_out_${chain[0].id}`))
                            .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
                            .as(TensorNode);
    graph.addEdge(tripCastN, tripScalar).init(new OnnxEdge.Builder(tripScalar.literalType, tripScalar.shape)).as(OnnxEdge);

    const axes0 = makeTensorConst(graph, `axes0_${chain[0].id}`, DataType.INT64, "constant", int64Vec([0]));
    const tripUnsq = graph.addNode(uniq(graph, `range_trip_unsq_${chain[0].id}`))
                          .init(new OperationNode.Builder("Unsqueeze", [tripScalar, axes0]))
                          .as(OperationNode);
    const tripVec = graph.addNode(uniq(graph, `range_trip_vec_${chain[0].id}`))
                        .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
                        .as(TensorNode);
    graph.addEdge(tripUnsq, tripVec).init(new OnnxEdge.Builder(tripVec.literalType, tripVec.shape)).as(OnnxEdge);

    // 6) v_initial = ConstantOfShape(tripVec) with zeros(elemTy)
    const init = graph.addNode(uniq(graph, `range_init_${chain[0].id}`))
                        .init(new OperationNode.Builder("ConstantOfShape", [tripVec], {
                          value: { type: "TENSOR", ...zeroTensor(elemTy, [1]) }
                        }))
                        .as(OperationNode);
    const vInitDims = isDynamicLen ? [undefined] : [carryLen]; // or [-1]
    v_initial = graph.addNode(uniq(graph, `range_init_out_${chain[0].id}`))
      .init(new TensorNode.Builder(elemTy, vInitDims, "intermediate"))
      .as(TensorNode);
    graph.addEdge(init, v_initial).init(new OnnxEdge.Builder(v_initial.literalType, v_initial.shape)).as(OnnxEdge);

    // Loop inputs
    trip = tripScalar; // INT64 scalar
    cond = makeTensorConst(graph, `cond_${chain[0].id}`, DataType.BOOL, "constant", bool(true));

  } else {
    // --- STATIC BRANCH: use known trip_count and a zero initializer ---
    trip = makeTensorConst(graph, `trip_count_${chain[0].id}`, DataType.INT64, "constant", scalarInt64(totalIters));
    cond = makeTensorConst(graph, `cond_${chain[0].id}`, DataType.BOOL, "constant", bool(true));
    v_initial = makeTensorConst(graph, "init_carry", elemTy, "initializer", carryInit ? carryInit : zeroTensor(elemTy, [carryLen]));
  }


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

    const shapeProto = int64Vec((outShape as number[]));
    const shapeNode = graph.addNode(uniq(graph, `reshape_shape_${chain[0].id}`))
      .init(new TensorNode.Builder(
        DataType.INT64, [outShape.length],
        "constant", shapeProto))
      .as(TensorNode);

    const reshape = graph.addNode(uniq(graph, `reshape_${chain[0].id}`))
      .init(new OperationNode.Builder("Reshape", [loop_out, shapeNode]))
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
