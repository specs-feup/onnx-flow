/**********************************************************************
 * Build a Loop node (outer-graph) + body graph for a linear chain
 *********************************************************************/
import Graph           from "@specs-feup/flow/graph/Graph";
import OnnxGraph       from "../../OnnxGraph.js";
import TensorNode      from "../../TensorNode.js";
import OperationNode   from "../../OperationNode.js";
import OnnxEdge        from "../../OnnxEdge.js";
import { DataType, TensorProto }    from "../../OnnxTypes.js";
import { int64Vec, scalarInt64, zeroTensor } from "../Utilities.js"

/* util to ensure unique node IDs                                            */
export function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0;
  let id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}

/* util to create TensorNode const                                           */
function makeTensorConst(
  g: OnnxGraph.Class, id: string, dataType: DataType, tensorKind: TensorNode.TensorKind, proto: TensorProto
) {
  return tensorKind == "initializer" ? g.addNode(uniq(g, id))
          .init(new TensorNode.Builder(dataType, proto.dims!, tensorKind, undefined, proto))
          .as(TensorNode) 
          : g.addNode(uniq(g, id))
          .init(new TensorNode.Builder(dataType, proto.dims!, tensorKind, proto))
          .as(TensorNode);
}

/* util to obtain scalar inputs inside body by Gather                  */
  function gatherFrom(g: OnnxGraph.Class, data: TensorNode.Class, tag: string,
                      indexNode: OperationNode.Class | TensorNode.Class,
                      axis: number) {
    const gather_node = g.addNode(uniq(g, tag))
                  .init(new OperationNode.Builder("Gather",
                        [data, indexNode], { axis }))
                  .as(OperationNode);
    const gather_out = g.addNode(uniq(g, `${tag}_out`)).init(new TensorNode.Builder(data.literalType, data.shape, 'intermediate')).as(TensorNode);
    g.addEdge(gather_node, gather_out).init(new OnnxEdge.Builder()).as(OnnxEdge);
    return [gather_node, gather_out] as [OperationNode.Class, TensorNode.Class]
}

/**  Flatten a tensor to 1-D if its rank > 1 so it can be used
  *  element-by-element inside a vector loop. */
function ensureFlatInput(
  g      : OnnxGraph.Class,
  shape  : number[], 
  t      : TensorNode.Class           // original tensor
): TensorNode.Class {
  if (shape.length <= 1) return t;          // already 1-D (or scalar)

  const total = shape.reduce((a, d) => a * d, 1);
  const shapeConst = makeTensorConst(
        g, `flat_shape_${t.id}`, DataType.INT64, "constant",
        int64Vec([total]));

  /* Reshape(t, [total]) ------------------------------------------------ */
  const rs = g.addNode(uniq(g, `flat_rs_${t.id}`))
              .init(new OperationNode.Builder("Reshape", [t, shapeConst]))
              .as(OperationNode);

  const flat = g.addNode(uniq(g, `${t.id}_flat`))
                .init(new TensorNode.Builder(
                      t.literalType, [total], "intermediate"))
                .as(TensorNode);

  g.addEdge(rs, flat)      .init(new OnnxEdge.Builder(
                                t.literalType, [total])).as(OnnxEdge);

  return flat;
}

export function buildLoopForChain(
  chain: OperationNode.Class[],
  graph: OnnxGraph.Class
): void {

  /* ---------- collect basic facts -------------------------------- */
  const outTensor = chain[chain.length - 1].getOutgoers.targets
                       .filterIs(TensorNode).first();
  const elemTy    = outTensor.literalType == DataType.UNDEFINED ? chain[chain.length - 1].getOutgoers.first().literalType : outTensor.literalType;
  const outShape  = outTensor.shape.length == 0 ? chain[chain.length - 1].getOutgoers.first().shape : outTensor.shape;

  const inputs = new Map<string, TensorNode.Class>();
  chain.forEach(op => {
    op.getInputs()?.filter(n => n.is(TensorNode))
      .forEach(t => inputs.set(t.id, t.as(TensorNode)));
  });

  const totalIters = outShape.length == 0 ? 1 : (outShape.length > 1 ? outShape.reduce((a, d) => a * d, 1) : outShape[0]);

  /* ---------- create body graph ---------------------------------- */
  const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

  /* mandatory inputs */
  const iter   = body.addNode(uniq(body, "iter_idx"))
                     .init(new TensorNode.Builder(DataType.INT64, [],"input"))
                     .as(TensorNode);
  const condIn = body.addNode(uniq(body, "cond_in"))
                     .init(new TensorNode.Builder(DataType.BOOL, [],"input"))
                     .as(TensorNode);


  const init_carry : TensorProto = zeroTensor(elemTy, [totalIters]);

  const carry = body.addNode(uniq(body, "carry"))
                     .init(new TensorNode.Builder(elemTy ,[totalIters],"input", init_carry))
                     .as(TensorNode);


  /* constants inside body */
  const axes = makeTensorConst(body, "axes",DataType.INT64, "constant", int64Vec([0]));

  /* idx → Unsqueeze                                                     */
  const unsq = body.addNode(uniq(body, "idx_unsq"))
                   .init(new OperationNode.Builder("Unsqueeze",[iter, axes]))
                   .as(OperationNode);
  body.addEdge(iter, unsq).init(new OnnxEdge.Builder()).as(OnnxEdge);
  body.addEdge(axes, unsq).init(new OnnxEdge.Builder()).as(OnnxEdge);

  const unsq_out = body.addNode(uniq(body, "idx_unsq_out")).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
  body.addEdge(unsq, unsq_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

  const opMap = new Map<OperationNode.Class, [OperationNode.Class, TensorNode.Class]>(); // original → clone
  let lastScalar: TensorNode.Class | OperationNode.Class;

  const idxScalar = unsq_out; // reuse
  const gatherings = new Map<string, [OperationNode.Class, TensorNode.Class]>();

  // Determine if the current chain needs flattening (based on ops)
  const needsFlattening = new Set(["Add", "Sub", "Mul", "Div"]);

  chain.forEach(orig => {
    let opType = orig.type;

    let clone: OperationNode.Class;
    let clone_out: TensorNode.Class;

    switch (opType) {
      case "Add":
      case "Sub":
      case "Mul":
      case "Div": {
        const inClones = orig.getInputs()!.map(inp => {
          if (inp.is(TensorNode)) {
            const t = inp.as(TensorNode);
            const flat = ensureFlatInput(body, outShape, t);
            const [gatherNode, gatherOut] = gatherFrom(body, flat, `g_${t.id}_${orig.id}`, idxScalar, 0);
            body.addEdge(unsq_out, gatherNode).init(new OnnxEdge.Builder()).as(OnnxEdge);
            return [gatherNode, gatherOut];
          }
          return opMap.get(inp as OperationNode.Class)!;
        });

        clone = body.addNode(uniq(body, `${opType}_${orig.id}`))
                    .init(new OperationNode.Builder(opType, inClones.map(i => i[1])))
                    .as(OperationNode);
        clone_out = body.addNode(uniq(body, `${opType}_${orig.id}_out`))
                        .init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate'))
                        .as(TensorNode);
        break;
      }

      case "MatMul": {
        // General MatMul decomposition (without assuming just one in the chain)
        const A = orig.getInputs()![0].as(TensorNode); // left input
        const B = orig.getInputs()![1].as(TensorNode); // right input

        // Assume 2D, get shape from A and B
        const K = (A.shape.length === 2) ? A.shape[1] : A.shape.at(-1);
        const N = (B.shape.length === 2) ? B.shape[1] : B.shape.at(-1);

        const Nconst = makeTensorConst(body, `N_${orig.id}`, DataType.INT64, "constant", scalarInt64(N));
        const shape1 = makeTensorConst(body, `shape1_${orig.id}`, DataType.INT64, "constant", int64Vec([1]));

        const row = body.addNode(uniq(body, `row_${orig.id}`))
                        .init(new OperationNode.Builder("Div", [iter, Nconst]))
                        .as(OperationNode);
        const row_out = body.addNode(uniq(body, `row_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(iter, row).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(Nconst, row).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(row, row_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const col = body.addNode(uniq(body, `col_${orig.id}`))
                        .init(new OperationNode.Builder("Mod", [iter, Nconst]))
                        .as(OperationNode);
        const col_out = body.addNode(uniq(body, `col_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(iter, col).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(Nconst, col).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(col, col_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const rowU = body.addNode(uniq(body, `rowU_${orig.id}`))
                         .init(new OperationNode.Builder("Unsqueeze", [row_out, axes]))
                         .as(OperationNode);
        const rowU_out = body.addNode(uniq(body, `rowU_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(row_out, rowU).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(axes, rowU).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(rowU, rowU_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const colU = body.addNode(uniq(body, `colU_${orig.id}`))
                         .init(new OperationNode.Builder("Unsqueeze", [col_out, axes]))
                         .as(OperationNode);
        const colU_out = body.addNode(uniq(body, `colU_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(col_out, colU).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(axes, colU).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(colU, colU_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const aRow = gatherFrom(body, A, `gA_${orig.id}`, rowU_out, 0);
        const bCol = gatherFrom(body, B, `gB_${orig.id}`, colU_out, 1);
        const shapeK = makeTensorConst(body, `shapeK_${orig.id}`, DataType.INT64, "constant", int64Vec([K]));

        const aVec = body.addNode(uniq(body, `aVec_${orig.id}`))
                         .init(new OperationNode.Builder("Reshape", [aRow[1], shapeK]))
                         .as(OperationNode);
        const aVec_out = body.addNode(uniq(body, `aVec_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(aRow[1], aVec).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(shapeK, aVec).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(aVec, aVec_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const bVec = body.addNode(uniq(body, `bVec_${orig.id}`))
                         .init(new OperationNode.Builder("Reshape", [bCol[1], shapeK]))
                         .as(OperationNode);
        const bVec_out = body.addNode(uniq(body, `bVec_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(bCol[1], bVec).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(shapeK, bVec).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(bVec, bVec_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const prod = body.addNode(uniq(body, `prod_${orig.id}`))
                         .init(new OperationNode.Builder("Mul", [aVec_out, bVec_out]))
                         .as(OperationNode);
        const prod_out = body.addNode(uniq(body, `prod_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(aVec_out, prod).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(bVec_out, prod).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(prod, prod_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const dot = body.addNode(uniq(body, `dot_${orig.id}`))
                        .init(new OperationNode.Builder("ReduceSum", [prod_out, axes]))
                        .as(OperationNode);
        const dot_out = body.addNode(uniq(body, `dot_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(prod_out, dot).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(axes, dot).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(dot, dot_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        const dotS = body.addNode(uniq(body, `dotS_${orig.id}`))
                         .init(new OperationNode.Builder("Reshape", [dot_out, shape1]))
                         .as(OperationNode);
        clone_out = body.addNode(uniq(body, `dotS_out_${orig.id}`)).init(new TensorNode.Builder(DataType.UNDEFINED, [], 'intermediate')).as(TensorNode);
        body.addEdge(dot_out, dotS).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(shape1, dotS).init(new OnnxEdge.Builder()).as(OnnxEdge);
        body.addEdge(dotS, clone_out).init(new OnnxEdge.Builder()).as(OnnxEdge);

        clone = dotS; // final operation for bookkeeping
        break;
      }

      default:
        throw new Error(`Unsupported operation: ${opType}`);
    }

    body.addEdge(clone, clone_out).init(new OnnxEdge.Builder()).as(OnnxEdge);
    opMap.set(orig, [clone, clone_out]);
  });

  lastScalar = opMap.get(chain[chain.length - 1])![1];

  const updates = lastScalar;

  /* Scatter carry[idx] = lastScalar                                      */
  const scatter = body.addNode(uniq(body, "scatter"))
                      .init(new OperationNode.Builder("ScatterElements",
                            [carry, unsq_out, updates], { axis: 0 }))
                      .as(OperationNode);
  body.addEdge(carry, scatter).init(new OnnxEdge.Builder()).as(OnnxEdge);
  body.addEdge(unsq_out,  scatter).init(new OnnxEdge.Builder()).as(OnnxEdge);
  body.addEdge(updates, scatter).init(new OnnxEdge.Builder()).as(OnnxEdge);

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
  

  const v_initial = makeTensorConst(graph, "init_carry", DataType.FLOAT, "initializer", init_carry);
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
