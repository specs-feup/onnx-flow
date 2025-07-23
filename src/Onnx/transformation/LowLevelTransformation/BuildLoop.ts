/**********************************************************************
 * Build a Loop node (outer-graph) + body graph for a linear chain
 *********************************************************************/
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { DataType, TensorProto } from "../../OnnxTypes.js";
import { int64Vec, scalarInt64, zeroTensor } from "../Utilities.js";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import BaseNode from "@specs-feup/flow/graph/BaseNode";

/* ------------------------------ Helpers ------------------------------ */

export function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
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
  const out = g.addNode(uniq(g, `${tag}_out`))
    .init(new TensorNode.Builder(data.literalType, data.shape, 'intermediate'))
    .as(TensorNode);
  g.addEdge(gather, out).init(new OnnxEdge.Builder()).as(OnnxEdge);
  return [gather, out];
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
               .init(new TensorNode.Builder(DataType.UNDEFINED, [], "intermediate"))
               .as(TensorNode);
  g.addEdge(node, out).init(new OnnxEdge.Builder()).as(OnnxEdge);
  return out;
}

function unsqueezeIdx(
  g: OnnxGraph.Class, idx: TensorNode.Class, axes: TensorNode.Class, tag: string
): TensorNode.Class {
  const unsq = g.addNode(uniq(g, tag))
                .init(new OperationNode.Builder("Unsqueeze", [idx, axes]))
                .as(OperationNode);
  const out = g.addNode(uniq(g, `${tag}_out`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [1], "intermediate"))
               .as(TensorNode);
  g.addEdge(unsq, out).init(new OnnxEdge.Builder()).as(OnnxEdge);
  return out;
}

function gatherAndReshape(
  g: OnnxGraph.Class, t: TensorNode.Class, idx: TensorNode.Class,
  axis: number, shape: TensorNode.Class, tag: string
): TensorNode.Class {
  const [_, gathered] = gatherFrom(g, t, tag, idx, axis);
  const reshape = g.addNode(uniq(g, `${tag}_reshape`))
                   .init(new OperationNode.Builder("Reshape", [gathered, shape]))
                   .as(OperationNode);
  const out = g.addNode(uniq(g, `${tag}_out`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [shape.shape[0]], "intermediate"))
               .as(TensorNode);
  g.addEdge(reshape, out).init(new OnnxEdge.Builder()).as(OnnxEdge);
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
  g.addEdge(reshape, out).init(new OnnxEdge.Builder()).as(OnnxEdge);
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
      const [gather, gathered] = gatherFrom(g, gatherInput, `g_${t.id}_${op.id}`, ctx.unsqIdx, 0);
      g.addEdge(ctx.unsqIdx, gather).init(new OnnxEdge.Builder()).as(OnnxEdge);
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
    axes: TensorNode.Class,
    outShape: number[]
  }
): TensorNode.Class {
  const inputs = op.getInputs()!.map(inp => resolveFusedInput(g, inp, ctx, op));

  const node = g.addNode(uniq(g, `${op.type}_${op.id}`))
                .init(new OperationNode.Builder(op.type, inputs))
                .as(OperationNode);

  const out = g.addNode(uniq(g, `${op.id}_out`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [], "intermediate"))
               .as(TensorNode);

  g.addEdge(node, out).init(new OnnxEdge.Builder()).as(OnnxEdge);

  return out;
}


function handleMatMul(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: {
    opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
    iter: TensorNode.Class,
    unsqIdx: TensorNode.Class,
    axes: TensorNode.Class,
    outShape: number[]
  }
): TensorNode.Class {
  const lhsInput = op.getInputs()![0];
  const rhsInput = op.getInputs()![1];

  const lhsTensor = resolveFusedInput(g, lhsInput, ctx, op, false, false);
  const rhsTensor = resolveFusedInput(g, rhsInput, ctx, op, false, false);

  const K = lhsTensor.shape.at(-1)!;
  const N = rhsTensor.shape.at(-1)!;

  const Nconst = makeTensorConst(g, `N_${op.id}`, DataType.INT64, "constant", scalarInt64(N));
  const shape1 = makeTensorConst(g, `shape1_${op.id}`, DataType.INT64, "constant", int64Vec([1]));
  const shapeK = makeTensorConst(g, `shapeK_${op.id}`, DataType.INT64, "constant", int64Vec([K]));

  const rowIdx = divmod(g, ctx.iter, Nconst, op.id, "Div");
  const colIdx = divmod(g, ctx.iter, Nconst, op.id, "Mod");

  const rowU = unsqueezeIdx(g, rowIdx, ctx.axes, `rowU_${op.id}`);
  const colU = unsqueezeIdx(g, colIdx, ctx.axes, `colU_${op.id}`);

  const [_, rowGathered] = gatherFrom(g, lhsTensor, `g_${lhsTensor.id}_${op.id}`, rowU, 0);
  const [__, colGathered] = gatherFrom(g, rhsTensor, `g_${rhsTensor.id}_${op.id}`, colU, 1);

  const row = reshapeTensor(g, rowGathered, shapeK, `reshapeRow_${op.id}`);
  const col = reshapeTensor(g, colGathered, shapeK, `reshapeCol_${op.id}`);

  const mul = g.addNode(uniq(g, `mul_${op.id}`))
               .init(new OperationNode.Builder("Mul", [row, col]))
               .as(OperationNode);
  const mulOut = g.addNode(uniq(g, `mul_out_${op.id}`))
                  .init(new TensorNode.Builder(DataType.UNDEFINED, [K], "intermediate"))
                  .as(TensorNode);
  g.addEdge(mul, mulOut).init(new OnnxEdge.Builder()).as(OnnxEdge);

  const reduce = g.addNode(uniq(g, `reduce_${op.id}`))
                  .init(new OperationNode.Builder("ReduceSum", [mulOut, ctx.axes]))
                  .as(OperationNode);
  const reduceOut = g.addNode(uniq(g, `reduce_out_${op.id}`))
                     .init(new TensorNode.Builder(DataType.UNDEFINED, [], "intermediate"))
                     .as(TensorNode);
  g.addEdge(reduce, reduceOut).init(new OnnxEdge.Builder()).as(OnnxEdge);

  const reshape = g.addNode(uniq(g, `reshape_${op.id}`))
                   .init(new OperationNode.Builder("Reshape", [reduceOut, shape1]))
                   .as(OperationNode);
  const finalOut = g.addNode(uniq(g, `final_out_${op.id}`))
                    .init(new TensorNode.Builder(DataType.UNDEFINED, [1], "intermediate"))
                    .as(TensorNode);
  g.addEdge(reshape, finalOut).init(new OnnxEdge.Builder()).as(OnnxEdge);

  return finalOut;
}

function handleTranspose(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: {
    opMap: Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>,
    iter: TensorNode.Class,
    unsqIdx: TensorNode.Class,
    axes: TensorNode.Class,
    outShape: number[]
  }
): TensorNode.Class {
  const firstInput = op.getInputs()![0];

  const matTensor = resolveFusedInput(g, firstInput, ctx, op, false, false);

  const rowSize = matTensor.shape.at(0)!; // N
  const colSize = matTensor.shape.at(1)!; // M

  const Nconst = makeTensorConst(g, `N_${op.id}`, DataType.INT64, "constant", scalarInt64(rowSize));
  const flattenedShape = makeTensorConst(g, `shape1_${op.id}`, DataType.INT64, "constant", int64Vec([colSize]));

  // Row index
  const scalarRowIdx = g.addNode(uniq(g, `Mod_${op.id}`))
                   .init(new OperationNode.Builder("Mod", [ctx.iter, Nconst]))
                   .as(OperationNode);
  
  g.addEdge(ctx.iter, scalarRowIdx).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the iteration with the "Mod" operation.
  g.addEdge(Nconst, scalarRowIdx).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the Nconst with the "Mod" operation.

  const scalarRowIdxOut = g.addNode(uniq(g, `scalarRowIndex_out_${op.id}`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [], "intermediate"))
               .as(TensorNode);

  g.addEdge(scalarRowIdx, scalarRowIdxOut).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the Div operation with its output.


  // Column index
  const scalarColIdx = g.addNode(uniq(g, `Div_${op.id}`))
                   .init(new OperationNode.Builder("Div", [ctx.iter, Nconst]))
                   .as(OperationNode);
  
  g.addEdge(ctx.iter, scalarColIdx).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the "iteration number" with the Div operation.
  g.addEdge(Nconst, scalarColIdx).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the "Nconst" with the Div operation.

  
  const scalarColIdxOut = g.addNode(uniq(g, `scalarColumnIndex_out_${op.id}`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [], "intermediate"))
               .as(TensorNode);
  
  g.addEdge(scalarColIdx, scalarColIdxOut).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the Mod operation with its output.


  // Column index unsqueeze
  const columnIndexNode = g.addNode(uniq(g, `columnIndex_${op.id}`))
                .init(new OperationNode.Builder("Unsqueeze", [scalarColIdxOut, ctx.axes]))
                .as(OperationNode);
  g.addEdge(scalarColIdxOut, columnIndexNode).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the "scalarColIdxOut" to the unsqueeze operator.
  g.addEdge(ctx.axes, columnIndexNode).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the "axes" to the unsqueeze operator.

  const colU = g.addNode(uniq(g, `columnIndex_out_${op.id}`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [1], "intermediate"))
               .as(TensorNode);

  g.addEdge(columnIndexNode, colU).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the Unsqueeze operation with its output.


  // Row index unsqueeze
  const rowIndexNode = g.addNode(uniq(g, `rowIndex_${op.id}`))
                .init(new OperationNode.Builder("Unsqueeze", [scalarRowIdxOut, ctx.axes]))
                .as(OperationNode);
  g.addEdge(scalarRowIdxOut, rowIndexNode).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the "scalarRowIdxOut" to the unsqueeze operator.
  g.addEdge(ctx.axes, rowIndexNode).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the "axes" to the unsqueeze operator.

  const rowU = g.addNode(uniq(g, `rowIndex_out_${op.id}`))
               .init(new TensorNode.Builder(DataType.UNDEFINED, [1], "intermediate"))
               .as(TensorNode);

  g.addEdge(rowIndexNode, rowU).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects the Unsqueeze operation with its output.

  // Gather entire row
  const [_, rowMatrix] = gatherFrom(g, matTensor, `rowMatrix_${op.id}`, rowU, 0);

  g.addEdge(rowU, _).init(new OnnxEdge.Builder()).as(OnnxEdge) // Edge that connects the unsqueeze for "rowIndex" to the gather for "rowMatrix".
  
  // Reshape row to 1D vector
  const reshape = g.addNode(uniq(g, `reshape_row_${op.id}`))
                   .init(new OperationNode.Builder("Reshape", [rowMatrix, flattenedShape]))
                   .as(OperationNode);

  g.addEdge(flattenedShape, reshape).init(new OnnxEdge.Builder()).as(OnnxEdge) // Edge that connects the "iteration number" to reshape.
  g.addEdge(rowMatrix, reshape).init(new OnnxEdge.Builder()).as(OnnxEdge) // Edge that connects the gather for "rowMatrix" to reshape.
  
  const row = g.addNode(uniq(g, `row_${op.id}`))
                   .init(new TensorNode.Builder(DataType.UNDEFINED, [1], "intermediate"))
                   .as(TensorNode);
  
  g.addEdge(reshape, row).init(new OnnxEdge.Builder()).as(OnnxEdge);


  const [__, matrixElement] = gatherFrom(g, row, `matrixElement_${op.id}`, colU, 0)

  g.addEdge(colU, __).init(new OnnxEdge.Builder()).as(OnnxEdge);
  g.addEdge(row, __).init(new OnnxEdge.Builder()).as(OnnxEdge); // Edge that connects "row" to the last gather.


  return matrixElement;
}


export {
  handleSimpleArithmeticOperation,
  handleMatMul,
  handleTranspose
};


/* Main Function */

export function buildLoopForChain(
  chain: OperationNode.Class[],
  graph: OnnxGraph.Class
): void {
  const lastOp = chain.at(-1)!;
  const outTensor = lastOp.getOutgoers.targets.filterIs(TensorNode).first();
  const elemTy = outTensor.literalType === DataType.UNDEFINED
    ? lastOp.getOutgoers.first().literalType
    : outTensor.literalType;
  const outShape = outTensor.shape.length === 0
    ? lastOp.getOutgoers.first().shape
    : outTensor.shape;
  const totalIters = outShape.length <= 1 ? outShape[0] ?? 1 : outShape.reduce((a, b) => a * b, 1);

  const inputs = new Map<string, TensorNode.Class>();
  chain.forEach(op =>
    op.getInputs()?.filter(n => n.is(TensorNode)).forEach(t => inputs.set(t.id, t.as(TensorNode)))
  );

  const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
  const iter = body.addNode(uniq(body, "iter")).init(new TensorNode.Builder(DataType.INT64, [], "input")).as(TensorNode);
  const condIn = body.addNode(uniq(body, "cond_in")).init(new TensorNode.Builder(DataType.BOOL, [], "input")).as(TensorNode);
  const carryInit = zeroTensor(elemTy, [totalIters]);
  const carry = body.addNode(uniq(body, "carry")).init(new TensorNode.Builder(elemTy, [totalIters], "input", carryInit)).as(TensorNode);

  const axes = makeTensorConst(body, "axes", DataType.INT64, "constant", int64Vec([0]));

  const unsq = body.addNode(uniq(body, "unsq"))
    .init(new OperationNode.Builder("Unsqueeze", [iter, axes]))
    .as(OperationNode);
  const unsqOut = body.addNode(uniq(body, "unsq_out"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
    .as(TensorNode);
  body.addEdge(unsq, unsqOut).init(new OnnxEdge.Builder()).as(OnnxEdge);

  const opMap = new Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>();

  const handlers: Record<string, typeof handleSimpleArithmeticOperation> = {
    Add: handleSimpleArithmeticOperation,
    Sub: handleSimpleArithmeticOperation,
    Mul: handleSimpleArithmeticOperation,
    Div: handleSimpleArithmeticOperation,
    MatMul: handleMatMul,
    Transpose: handleTranspose,
  };

  for (const op of chain) {
    const handler = handlers[op.type];
    if (!handler) throw new Error(`Unsupported op: ${op.type}`);
    const output = handler(op, body, {
      opMap,
      iter,
      unsqIdx: unsqOut,
      axes,
      outShape,
    });
    opMap.set(op, [undefined as any, output]); // Op node unused outside; only value matters.
  }

  const lastOut = opMap.get(lastOp)![1];

  const scatter = body.addNode(uniq(body, "scatter"))
                      .init(new OperationNode.Builder("ScatterElements",
                            [carry, unsqOut, lastOut], { axis: 0 }))
                      .as(OperationNode);
  body.addEdge(carry, scatter).init(new OnnxEdge.Builder()).as(OnnxEdge);
  body.addEdge(unsqOut,  scatter).init(new OnnxEdge.Builder()).as(OnnxEdge);
  body.addEdge(lastOut, scatter).init(new OnnxEdge.Builder()).as(OnnxEdge);

  /* cond passthrough */
  const idCond = body.addNode(uniq(body, "id_cond"))
                     .init(new OperationNode.Builder("Identity", [condIn]))
                     .as(OperationNode);
  const condOut = body.addNode(uniq(body, "cond_out"))
                      .init(new TensorNode.Builder(DataType.BOOL, [], "output"))
                      .as(TensorNode);
  body.addEdge(condIn,idCond).init(new OnnxEdge.Builder()).as(OnnxEdge);
  body.addEdge(idCond,condOut).init(new OnnxEdge.Builder()).as(OnnxEdge);

  const carryOut = body.addNode(uniq(body, "carry_out"))
                       .init(new TensorNode.Builder(elemTy, carry.shape, "output"))
                       .as(TensorNode);
  body.addEdge(scatter, carryOut)
      .init(new OnnxEdge.Builder(carryOut.literalType, carryOut.shape)).as(OnnxEdge);

  /* ---------- outer Loop node + wiring -------------------------------- */

  /* ensure global trip_count / cond exist                                */  
  const trip = graph.addNode(uniq(graph, `trip_count_${chain[0].id}`))
                    .init(new TensorNode.Builder(DataType.INT64, [], "input"))
                    .as(TensorNode);
  
  const cond = graph.addNode(uniq(graph, `cond_${chain[0].id}`)) 
                    .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
                    .as(TensorNode);
  

  const v_initial = makeTensorConst(graph, "init_carry", DataType.FLOAT, "initializer", carryInit);
  const loop = graph.addNode(uniq(graph, `Loop_${chain[0].id}`))
                    .init(new OperationNode.Builder("Loop", [trip, cond, v_initial], {}, body))
                    .as(OperationNode);

  graph.addEdge(trip, loop).init(new OnnxEdge.Builder()).as(OnnxEdge);
  graph.addEdge(cond, loop).init(new OnnxEdge.Builder()).as(OnnxEdge);

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
    const loop_out = graph.addNode(uniq(graph, "loop_out")).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
    graph.addEdge(loop, loop_out).init(new OnnxEdge.Builder()).as(OnnxEdge);
    
    const shapeProto = int64Vec(outShape);
    const shapeNode  = graph.addNode(uniq(graph, `reshape_shape_${chain[0].id}`))
                            .init(new TensorNode.Builder(
                                  DataType.INT64, [outShape.length],
                                  "constant", shapeProto))
                            .as(TensorNode);
                
    const reshape = graph.addNode(uniq(graph, `reshape_${chain[0].id}`))
                         .init(new OperationNode.Builder("Reshape",[loop_out, shapeNode]))
                         .as(OperationNode);
    
    graph.addEdge(loop_out, reshape).init(new OnnxEdge.Builder()).as(OnnxEdge);

    graph.addEdge(shapeNode, reshape).init(new OnnxEdge.Builder()).as(OnnxEdge);
    graph.addEdge(reshape, outTensor)
         .init(new OnnxEdge.Builder(elemTy, outShape)).as(OnnxEdge);
  } else {
    graph.addEdge(loop, outTensor)
         .init(new OnnxEdge.Builder(elemTy, outShape)).as(OnnxEdge);
  }

  /* finally, remove the original ops & dangling tensors                  */
  chain.forEach(op => op.remove());
}
