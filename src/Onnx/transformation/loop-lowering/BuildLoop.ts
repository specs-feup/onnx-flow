/**********************************************************************
 * Orchestrator: exposes helpers & context and delegates to builders
 *********************************************************************/
import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { DataType } from "../../OnnxTypes.js";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import TransformChain from "./TransformChain.js";
import { inferShapes } from "@specs-feup/onnx-flow/initGraph";
import {
  scalarInt64, uniq, int64Vec, zeroTensor, bool, Shape,
  makeTensorConst, computeStrides, isNum, toStaticShape, asStaticDims
} from "../../Utils.js";

/* ------------------------------------------------------------------ */
/* Public context given to handlers/builders                           */
/* ------------------------------------------------------------------ */
export type LoopCtx = {
  opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
  iter: TensorNode.Class,
  unsqIdx: TensorNode.Class | null,
  carry: TensorNode.Class,
  axes: TensorNode.Class,
  outShape: (number | String)[],
  coalesce: boolean,

  // Optional indices for coalesced MatMul
  iU?: TensorNode.Class | null,
  jU?: TensorNode.Class | null,
  kU?: TensorNode.Class | null,
  flatU?: TensorNode.Class | null,
  kIdx?: TensorNode.Class | null,
  kM1?: TensorNode.Class | null,
  gateByK?: boolean,
  running?: TensorNode.Class | null,

  // Optional for Reduce
  meanScale?: TensorNode.Class,
};

/* ------------------------------------------------------------------ */
/* Shared helpers (unchanged from your working version)                */
/* ------------------------------------------------------------------ */

export function decodeMixedRadix(
  g: OnnxGraph.Class, iter: TensorNode.Class, dims: number[], tag: string
): TensorNode.Class[] {
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
  return out;
}

export function buildLinearIndex(
  g: OnnxGraph.Class, idx: TensorNode.Class[], strides: number[], tag: string
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
      else if (dim !== d) throw new Error(`Broadcast mismatch on axis -${i + 1}: got ${dim} vs ${d}`);
    }
    out[maxR - 1 - i] = dim;
  }
  return out;
}

export function getMatDims(aShape: (number | string)[], bShape: (number | string)[]) {
  const a = asStaticDims(aShape);
  const b = asStaticDims(bShape);
  let aR = a.length, bR = b.length;
  let A = a.slice(), B = b.slice();
  if (aR === 1) { A = [1, a[0]]; aR = 2; }
  if (bR === 1) { B = [b[0], 1]; bR = 2; }
  const M = A[A.length - 2];
  const K = A[A.length - 1];
  const KN = B[B.length - 2];
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
  const indexShape = indexNode.is(TensorNode) ? indexNode.shape : [];

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

export function gatherAt2DPoint(
  g: OnnxGraph.Class, input: TensorNode.Class, rowIdx: TensorNode.Class, colIdx: TensorNode.Class, tag: string
): [OperationNode.Class, TensorNode.Class] {
  const g0 = g.addNode(uniq(g, `${tag}_g0`))
    .init(new OperationNode.Builder("Gather", [input, rowIdx], { axis: 0 }))
    .as(OperationNode);
  const g0Out = g.addNode(uniq(g, `${tag}_g0_out`))
    .init(new TensorNode.Builder(input.literalType, [1, input.shape[1]], "intermediate"))
    .as(TensorNode);
  g.addEdge(g0, g0Out).init(new OnnxEdge.Builder(g0Out.literalType, g0Out.shape)).as(OnnxEdge);

  const g1 = g.addNode(uniq(g, `${tag}_g1`))
    .init(new OperationNode.Builder("Gather", [g0Out, colIdx], { axis: 1 }))
    .as(OperationNode);
  const g1Out = g.addNode(uniq(g, `${tag}_g1_out`))
    .init(new TensorNode.Builder(input.literalType, [1, input.shape[1]], "intermediate"))
    .as(TensorNode);
  g.addEdge(g1, g1Out).init(new OnnxEdge.Builder(g1Out.literalType, g1Out.shape)).as(OnnxEdge);

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

export function gatherWithBroadcast(
  g: OnnxGraph.Class, t: TensorNode.Class, ctx: LoopCtx, tag: string
): TensorNode.Class {
  if (t.shape.length === 0) return t;

  const outDimsStatic = toStaticShape(ctx.outShape as Shape);
  const inDims = toStaticShape(t.shape as Shape).map(d => (d > 0 ? d : 1));

  if (inDims.length === 1 && outDimsStatic.length === 1 && outDimsStatic[0] <= 0) {
    const idx = ctx.unsqIdx ?? unsqueezeIdx(g, ctx.iter, ctx.axes, `unsq_idx_${tag}`);
    const [__, gathered] = gatherFrom(g, t, `gb_1d_iter_${tag}`, idx, 0);
    return gathered;
  }

  if (inDims.length > outDimsStatic.length) {
    const src = ensureFlatInput(g, t);
    const idx = ctx.unsqIdx ?? unsqueezeIdx(g, ctx.iter, ctx.axes, `unsq_idx_${tag}`);
    const [__, gathered] = gatherFrom(g, src, `gb_fallback_${tag}`, idx, 0);
    return gathered;
  }

  const rO = outDimsStatic.length, rI = inDims.length;
  const decodeDims: number[] = outDimsStatic.map(d => (d > 0 ? d : 1));
  for (let k = 0; k < rI; k++) {
    const outPos = rO - rI + k;
    if (outPos >= 0 && outDimsStatic[outPos] <= 0) {
      if (inDims[k] > 1) decodeDims[outPos] = inDims[k];
    }
  }

  const oDigits = decodeMixedRadix(g, ctx.iter, decodeDims, `gb_${tag}`);

  const iDigits: TensorNode.Class[] = [];
  for (let k = 0; k < rI; k++) {
    const outPos = rO - rI + k;
    const inDim = inDims[k];
    if (inDim === 1) {
      const z = makeTensorConst(g, `gb_zero_${tag}_${k}`, DataType.INT64, "constant", scalarInt64(0));
      iDigits.push(z);
    } else {
      iDigits.push(oDigits[outPos]);
    }
  }

  const strides = computeStrides(inDims);
  const lin = buildLinearIndex(g, iDigits, strides, `gb_lin_${tag}`);
  const linU = unsqueezeIdx(g, lin, ctx.axes, `gb_linU_${tag}`);

  const flat = ensureFlatInput(g, t);
  const [_, gathered] = gatherFrom(g, flat, `gb_g_${tag}`, linU, 0);
  return gathered;
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

export function ensureFlatInput(g: OnnxGraph.Class, t: TensorNode.Class): TensorNode.Class {
  const shape: Shape = t.shape;
  if (shape.length <= 1) return t;

  const allNum = shape.every(isNum);
  const total = allNum ? (shape as number[]).reduce((a, d) => a * d, 1) : -1;

  const shpVec = int64Vec([total]);
  const shapeConst = makeTensorConst(g, `flat_shape_${t.id}`, DataType.INT64, "constant", shpVec);

  const rs = g.addNode(uniq(g, `flat_rs_${t.id}`))
    .init(new OperationNode.Builder("Reshape", [t, shapeConst]))
    .as(OperationNode);

  const outStatic = [total];
  const flat = g.addNode(uniq(g, `${t.id}_flat`))
    .init(new TensorNode.Builder(t.literalType, outStatic, "intermediate"))
    .as(TensorNode);

  g.addEdge(rs, flat).init(new OnnxEdge.Builder(t.literalType, outStatic)).as(OnnxEdge);
  return flat;
}

export function unsqueezeIdx(g: OnnxGraph.Class, idx: TensorNode.Class, axes: TensorNode.Class, tag: string): TensorNode.Class {
  const unsq = g.addNode(uniq(g, tag))
    .init(new OperationNode.Builder("Unsqueeze", [idx, axes]))
    .as(OperationNode);
  const out = g.addNode(uniq(g, `${tag}_out`))
    .init(new TensorNode.Builder(idx.literalType, [1], "intermediate"))
    .as(TensorNode);
  g.addEdge(unsq, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
  return out;
}

export function resolveFusedInput(
  g: OnnxGraph.Class, input: BaseNode.Class, ctx: LoopCtx,
  op: OperationNode.Class, flatten = true, returnGather = true
): TensorNode.Class {
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

    let idxToUse: TensorNode.Class | null = ctx.unsqIdx;

    const [M, N] = ctx.outShape.length === 2 ? ctx.outShape : [undefined, undefined];

    if (ctx.coalesce && (ctx.iU || ctx.jU || ctx.flatU)) {
      const s = t.shape;

      if (s.length === 0) return t;

      if (s.length === 1) {
        const len = s[0];
        if (N !== undefined && len === N && ctx.jU) idxToUse = ctx.jU;
        else if (M !== undefined && len === M && ctx.iU) idxToUse = ctx.iU;
        else if (ctx.flatU) idxToUse = ctx.flatU;

        if (!returnGather) return t;
        const [_, gathered] = gatherFrom(g, t, `gather_${t.id}_${op.id}`, idxToUse!, 0);
        return gathered;
      }

      if (s.length === 2) {
        if (ctx.iU && ctx.jU) {
          const [_, picked] = gatherAt2DPoint(g, t, ctx.iU!, ctx.jU!, `g2d_${t.id}_${op.id}`);
          return picked;
        }
        const flatT = ensureFlatInput(g, t);
        const idx = ctx.flatU ?? idxToUse!;
        if (!returnGather) return flatT;
        const [__, gathered] = gatherFrom(g, flatT, `gather_${t.id}_${op.id}`, idx, 0);
        return gathered;
      }
    }

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

    return gatherWithBroadcast(g, t, ctx, `${t.id}_${op.id}`);
  }

  if (input.is(OperationNode)) {
    const fused = Array.from(ctx.opMap.entries()).find(([key]) => key.id === input.id);
    if (fused) return fused[1][1];
  }

  throw new Error(`Unhandled input case in resolveFusedInput for ${input.id}`);
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

/* ------------------------------------------------------------------ */
/* Builder wiring                                                      */
/* ------------------------------------------------------------------ */
export type BuildResult = {
  body: OnnxGraph.Class,
  ctx: LoopCtx,
  lastOut: TensorNode.Class,
  indicesOut: TensorNode.Class,
  elemTy: DataType,
  outShape: (number | String)[],
  inputs: Map<string, TensorNode.Class>,
  outTensor: TensorNode.Class,
  trip: TensorNode.Class,
  cond: TensorNode.Class,
  v_initial: TensorNode.Class
};

export interface LoopBuilder {
  canHandle(chain: OperationNode.Class[]): boolean;
  build(
    chain: OperationNode.Class[],
    outer: OnnxGraph.Class,
    opts: { fuse: boolean; recurse: boolean; coalesce: boolean }
  ): BuildResult;
}

/* ------------------------------------------------------------------ */
/* Builder selection                                                   */
/* ------------------------------------------------------------------ */
import DefaultBuilder from "./builders/Default.js";
import GenerativeBuilder from "./builders/Generative.js";
import MatMulBuilder from "./builders/MatMul.js";
import ReducesBuilder from "./builders/Reduces.js";

const BUILDERS: LoopBuilder[] = [
  new ReducesBuilder(),
  new MatMulBuilder(),      // must come before Default (it also handles trailing elemwise)
  new GenerativeBuilder(),  // Range (may also have trailing elemwise)
  new DefaultBuilder(),     // pure elemwise/transpose/etc.
];

/* ------------------------------------------------------------------ */
/* Main entry                                                         */
/* ------------------------------------------------------------------ */
export function buildLoopForChain(
  chain: OperationNode.Class[],
  graph: OnnxGraph.Class,
  fuse = true,
  recurse = true,
  coalesce = true
): void {
  const builder = BUILDERS.find(b => b.canHandle(chain));
  if (!builder) throw new Error(`No builder can handle chain starting at ${chain[0].type}`);

  const {
    body, ctx, lastOut, indicesOut, elemTy,
    outShape, inputs, outTensor, trip, cond, v_initial
  } = builder.build(chain, graph, { fuse, recurse, coalesce });

  // cond passthrough
  const condIn = body.getInputTensorNodes().filter(t => t.id.includes("cond_in")).first();
  const idCond = body.addNode(uniq(body, "id_cond"))
    .init(new OperationNode.Builder("Identity", [condIn]))
    .as(OperationNode);
  const condOut = body.addNode(uniq(body, "cond_out"))
    .init(new TensorNode.Builder(DataType.BOOL, [], "output"))
    .as(TensorNode);
  body.addEdge(condIn, idCond).init(new OnnxEdge.Builder(condIn.literalType, condIn.shape));
  body.addEdge(idCond, condOut).init(new OnnxEdge.Builder(condOut.literalType, condOut.shape));

  // Scatter update
  const scatter = body.addNode(uniq(body, "scatter"))
    .init(new OperationNode.Builder("ScatterElements", [ctx.carry, indicesOut, lastOut], { axis: 0 }))
    .as(OperationNode);
  body.addEdge(ctx.carry, scatter).init(new OnnxEdge.Builder(ctx.carry.literalType, ctx.carry.shape));
  body.addEdge(indicesOut, scatter).init(new OnnxEdge.Builder(indicesOut.literalType, indicesOut.shape));
  body.addEdge(lastOut, scatter).init(new OnnxEdge.Builder(lastOut.literalType, lastOut.shape));

  const carryOut = body.addNode(uniq(body, "carry_out"))
    .init(new TensorNode.Builder(elemTy, ctx.carry.shape, "output"))
    .as(TensorNode);
  body.addEdge(scatter, carryOut).init(new OnnxEdge.Builder(carryOut.literalType, carryOut.shape));

  inferShapes(graph);
  inferShapes(body);

  if (recurse) {
    const recursiveDecomposer = new TransformChain(fuse, recurse);
    recursiveDecomposer.apply(body);
  }

  /* ---------- Outer Loop node + wiring -------------------------------- */
  const loop = graph.addNode(uniq(graph, `Loop_${chain[0].id}`))
    .init(new OperationNode.Builder("Loop", [trip, cond, v_initial], {}, body))
    .as(OperationNode);

  graph.addEdge(trip, loop).init(new OnnxEdge.Builder(trip.literalType, trip.shape)).as(OnnxEdge);
  graph.addEdge(cond, loop).init(new OnnxEdge.Builder(cond.literalType, cond.shape)).as(OnnxEdge);

  inputs.forEach(t => {
    graph.addEdge(t, loop).init(new OnnxEdge.Builder(t.literalType, t.shape)).as(OnnxEdge);
  });

  chain[chain.length - 1].getOutgoers.forEach(e => e.remove());

  const isGlobalOutput = graph.getOutputTensorNodes().contains(outTensor);
  graph.getNodeById(outTensor.id).remove();
  graph.addNode(outTensor.id)
    .init(new TensorNode.Builder(elemTy, outShape, isGlobalOutput ? "output" : "intermediate"))
    .as(TensorNode);

  if (outShape.length > 1) {
    const loop_out = graph.addNode(uniq(graph, "loop_out"))
      .init(new TensorNode.Builder(elemTy, ctx.carry.shape, 'intermediate')).as(TensorNode);
    graph.addEdge(loop, loop_out).init(new OnnxEdge.Builder(loop_out.literalType, loop_out.shape)).as(OnnxEdge);

    const shapeVec = int64Vec(outShape as number[]);
    const shapeNode = graph.addNode(uniq(graph, `reshape_shape_${chain[0].id}`))
      .init(new TensorNode.Builder(DataType.INT64, [outShape.length], "constant", shapeVec))
      .as(TensorNode);

    const reshape = graph.addNode(uniq(graph, `reshape_${chain[0].id}`))
      .init(new OperationNode.Builder("Reshape", [loop_out, shapeNode]))
      .as(OperationNode);

    graph.addEdge(loop_out, reshape).init(new OnnxEdge.Builder(loop_out.literalType, loop_out.shape)).as(OnnxEdge);
    graph.addEdge(shapeNode, reshape).init(new OnnxEdge.Builder(shapeNode.literalType, shapeNode.shape)).as(OnnxEdge);
    graph.addEdge(reshape, outTensor).init(new OnnxEdge.Builder(elemTy, outShape)).as(OnnxEdge);
  } else {
    graph.addEdge(loop, outTensor).init(new OnnxEdge.Builder(elemTy, outShape)).as(OnnxEdge);
  }

  chain.forEach(op => op.remove());
}
