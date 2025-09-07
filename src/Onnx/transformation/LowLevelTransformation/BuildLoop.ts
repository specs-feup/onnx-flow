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

/* ------------------------------ Helpers ------------------------------ */

export function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (GRAPHS.some(gr => gr.hasNode(id))) id = `${base}_${++i}`;
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

function ensureFlatInput(
  g: OnnxGraph.Class, shape: number[], t: TensorNode.Class
): TensorNode.Class {
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
               .init(new TensorNode.Builder(lhs.literalType, [], "intermediate"))
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
  console.log("SHAPES:", actualShape, targetShape, isSame);
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
  ctx: {
    opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
    unsqIdx: TensorNode.Class,
    outShape: number[]
  },
  op: OperationNode.Class,
  flatten : boolean = true,
  returnGather : boolean = true
): TensorNode.Class {
  // Case 1: Input is a TensorNode (could be intermediate, input, etc.)
  if (input.is(TensorNode)) {
    const t = input.as(TensorNode);

    // Check if it's intermediate and has a producer that is fused
    if (t.type === "intermediate" && t.getIncomers.length > 0) {
      const producer = t.getIncomers[0].source;
      if (producer.is(OperationNode)) {
        const fused = Array.from(ctx.opMap.entries()).find(([key]) => key.id === producer.id);
        if (fused) {
          return fused[1][1]; // second element is the fused output tensor
        }
      }
    }

    // Otherwise fall back to Gather
    let gatherInput = t;
    if(flatten) gatherInput = ensureFlatInput(g, ctx.outShape, t);
    if(returnGather){
      const [gather, gathered] = gatherFrom(g, gatherInput, `gather_${t.id}_${op.id}`, ctx.unsqIdx, 0);
      g.addEdge(ctx.unsqIdx, gather).init(new OnnxEdge.Builder(ctx.unsqIdx.literalType, ctx.unsqIdx.shape)).as(OnnxEdge);
      return gathered;
    }
    else{
      return gatherInput;
    }
  }

  // Case 2: Direct operation input
  if (input.is(OperationNode)) {
    const fused = Array.from(ctx.opMap.entries()).find(([key]) => key.id === input.id);
    if (fused) {
      return fused[1][1]; // second element is the fused output tensor
    }
  }

  throw new Error(`Unhandled input case in resolveFusedInput for ${input.id}`);
}



/* ------------------- Handlers for Operation Types ------------------- */

function handleSimpleArithmeticOperation(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: {
    opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
    iter: TensorNode.Class,
    unsqIdx: TensorNode.Class,
    carry: TensorNode.Class,
    axes: TensorNode.Class,
    outShape: number[],
    coalesce: boolean,
  }
): TensorNode.Class {
  const inputs = op.getInputs()!.map(inp => resolveFusedInput(g, inp, ctx, op));

  const node = g.addNode(uniq(g, `${op.type}_${op.id}`))
                .init(new OperationNode.Builder(op.type, inputs))
                .as(OperationNode);

  const out = g.addNode(uniq(g, `${op.id}_out`))
               .init(new TensorNode.Builder(inputs[0].literalType, inputs[0].shape, "intermediate"))
               .as(TensorNode);

  g.addEdge(node, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

  return out;
}


function handleMatMul(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: {
    opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
    iter: TensorNode.Class,
    unsqIdx: TensorNode.Class,
    carry: TensorNode.Class,
    axes: TensorNode.Class,
    outShape: number[],
    coalesce: boolean,
  }
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
    const N_const  = makeTensorConst(g,  `N_${op.id}`, DataType.INT64, "constant", scalarInt64(N));

    // ---- decode (i,j,k) from loop counter ctx.iter ----
    const kIdx = divmod(g, ctx.iter, KN_const, `ij_${op.id}`, "Div"); // t // (N*K)
    const ijIdx  = divmod(g, ctx.iter, KN_const,  `k_${op.id}`, "Mod"); // t %  (N*K)

    const iIdx  = divmod(g, ijIdx, N_const,   `i_${op.id}`, "Div");    // (t//NK)//N
    const jIdx  = divmod(g, ijIdx, N_const,   `j_${op.id}`, "Mod");    // (t//NK)%N

    // ---- make [1]-shaped indices for Gather/GatherElements ----
    const iU = unsqueezeIdx(g, iIdx, ctx.axes, `iU_${op.id}`);
    const jU = unsqueezeIdx(g, jIdx, ctx.axes, `jU_${op.id}`);
    const kU = unsqueezeIdx(g, kIdx, ctx.axes, `kU_${op.id}`);


    // ---- flat = i*N + j  → flatU = Unsqueeze(flat) ----
    const iMulN_node = g.addNode(uniq(g, `iMulN_${op.id}`))
      .init(new OperationNode.Builder("Mul", [kIdx, N_const])).as(OperationNode);
    const iMulN = g.addNode(uniq(g, `iMulN_out_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
    g.addEdge(iMulN_node, iMulN).init(new OnnxEdge.Builder(iMulN.literalType, iMulN.shape)).as(OnnxEdge);

    const flat_node = g.addNode(uniq(g, `flat_${op.id}`))
      .init(new OperationNode.Builder("Add", [iMulN, iIdx])).as(OperationNode);
    const flat = g.addNode(uniq(g, `flat_out_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
    g.addEdge(flat_node, flat).init(new OnnxEdge.Builder(flat.literalType, flat.shape)).as(OnnxEdge);

    const flatU = unsqueezeIdx(g, flat, ctx.axes, `flatU_${op.id}`); // [1]
    ctx.unsqIdx = flatU;


    // ---- A[i,k] as [1] ----
    const a_row_node = g.addNode(uniq(g, `a_row_${op.id}`))
      .init(new OperationNode.Builder("Gather", [lhsTensor, kU], { axis: 0 })).as(OperationNode);
    const a_row = g.addNode(uniq(g, `a_row_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1, K], "intermediate")).as(TensorNode);
    g.addEdge(a_row_node, a_row).init(new OnnxEdge.Builder(a_row.literalType, a_row.shape)).as(OnnxEdge);

    // squeeze [1,K] -> [K]
    const a_vec = targetReshape(g, a_row, [K], `a_vec_${op.id}`); // [K]

    const a_pick_node = g.addNode(uniq(g, `a_pick_${op.id}`))
      .init(new OperationNode.Builder("Gather", [a_vec, jU], { axis: 0 })).as(OperationNode);
    const a_scalar = g.addNode(uniq(g, `a_scalar_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(a_pick_node, a_scalar).init(new OnnxEdge.Builder(a_scalar.literalType, a_scalar.shape)).as(OnnxEdge);

    // ---- B[k,j] as [1] ----
    const b_col_node = g.addNode(uniq(g, `b_col_${op.id}`))
      .init(new OperationNode.Builder("Gather", [rhsTensor, iU], { axis: 1 })).as(OperationNode);
    const b_col = g.addNode(uniq(g, `b_col_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [K, 1], "intermediate")).as(TensorNode);
    g.addEdge(b_col_node, b_col).init(new OnnxEdge.Builder(b_col.literalType, b_col.shape)).as(OnnxEdge);

    // squeeze [K,1] -> [K]
    const b_vec = targetReshape(g, b_col, [K], `b_vec_${op.id}`); // [K]


    const b_pick_node = g.addNode(uniq(g, `b_pick_${op.id}`))
      .init(new OperationNode.Builder("Gather", [b_vec, jU], { axis: 0 })).as(OperationNode);
    const b_scalar = g.addNode(uniq(g, `b_scalar_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(b_pick_node, b_scalar).init(new OnnxEdge.Builder(b_scalar.literalType, b_scalar.shape)).as(OnnxEdge);

    // ---- prod = A[i,k] * B[k,j]  (shape [1]) ----
    const mul_node = g.addNode(uniq(g, `mul_${op.id}`))
      .init(new OperationNode.Builder("Mul", [a_scalar, b_scalar])).as(OperationNode);
    const prod = g.addNode(uniq(g, `prod_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(mul_node, prod).init(new OnnxEdge.Builder(prod.literalType, prod.shape)).as(OnnxEdge);

    // ---- prev = carry[flat]  (shape [1]) ----
    const prev_node = g.addNode(uniq(g, `prev_${op.id}`))
      .init(new OperationNode.Builder("GatherElements", [ctx.carry, flatU], { axis: 0 })).as(OperationNode);
    const prev = g.addNode(uniq(g, `prev_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(prev_node, prev).init(new OnnxEdge.Builder(prev.literalType, prev.shape)).as(OnnxEdge);

    // ---- acc = prev + prod  (shape [1]) ----
    const add_node = g.addNode(uniq(g, `acc_${op.id}`))
      .init(new OperationNode.Builder("Add", [prev, prod])).as(OperationNode);
    const acc = g.addNode(uniq(g, `acc_out_${op.id}`))
      .init(new TensorNode.Builder(elemTy, [1], "intermediate")).as(TensorNode);
    g.addEdge(add_node, acc).init(new OnnxEdge.Builder(acc.literalType, acc.shape)).as(OnnxEdge);

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

  const withoutCoalescing = !coalesce && !chain.some(op => op.type == "MatMul");
  const matmulOp = chain.find(op => op.type === "MatMul");
  const lastOp = chain.at(-1)!;
  const outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();

  const elemTy = outTensor.literalType === DataType.UNDEFINED
    ? lastOp.getOutgoers.first().literalType
    : outTensor.literalType;
  let outShape = outTensor.shape.length === 0
    ? lastOp.getOutgoers.first().shape
    : outTensor.shape;

  let totalIters: number;
  if (coalesce && matmulOp) {
    const lhs = matmulOp.getInputs()![0].as(TensorNode);
    const rhs = matmulOp.getInputs()![1].as(TensorNode);
    const M = lhs.shape.at(0)!;
    const K = lhs.shape.at(1)!;
    const N = rhs.shape.at(1)!;

    totalIters = M * K * N;
    outShape = [M, N];  // final carry shape
  } else {
    totalIters = outShape.length <= 1 ? outShape[0] ?? 1 : outShape.reduce((a, b) => a * b, 1);
  }

  const carryLen = (coalesce && matmulOp)
    ? (outShape[0] * outShape[1])  // M * N
    : (outShape.length <= 1 ? (outShape[0] ?? 1) : outShape.reduce((a, b) => a * b, 1));

  const inputs = new Map<string, TensorNode.Class>();
  chain.forEach(op =>
    op.getInputs()?.filter(n => n.is(TensorNode)).forEach(t => inputs.set(t.id, t.as(TensorNode)))
  );

  const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
  const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
  const condIn = body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode);
  const carryInit = zeroTensor(elemTy, [carryLen]);
  const carry = body.addNode(uniq(body, "carry")).init(new TensorNode.Builder(elemTy, [carryLen], "input", carryInit)).as(TensorNode);

  const axes = makeTensorConst(body, "axes", DataType.INT64, "constant", int64Vec([0]));
  let unsqOut = null;

  if (!(coalesce && matmulOp)) {
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

  for (const op of chain) {
    const handler = handlers[op.type];
    if (!handler) throw new Error(`Unsupported op: ${op.type}`);
    const ctx = {
      opMap,
      iter,
      unsqIdx: unsqOut,
      carry,
      axes,
      outShape,
      coalesce,
    };
    const output = handler(op, body, ctx);
    if (coalesce && op.type === "MatMul") {
      indicesOut = ctx.unsqIdx;
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
    const loop_out = graph.addNode(uniq(graph, "loop_out")).init(new TensorNode.Builder(elemTy, [carryLen], 'intermediate')).as(TensorNode);
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
