import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js";
import ConstantNode from "../../ConstantNode.js";
import VariableNode from "../../VariableNode.js";
import { typeSizeMap, formatId} from "../Utilities.js";
import OnnxInnerEdge from "../../OnnxInnerEdge.js";

export default function transformLoop(node: OperationNode.Class, graph: OnnxGraph.Class): void {
  const nodeId = node.id;
  const edges = node.incomers.filterIs(OnnxEdge);

  const [edge0, edge1] = edges;
  const type = edge0.literalType;
  if (type === undefined) return;

  const shape = edge0.shape;
  const elemCount = shape.reduce((acc, dim) => acc * dim, 1);
  let order = 0;

  const loopNode = graph.addNode(formatId("Loop", nodeId))
    .init(new OperationNode.Builder("Loop")).as(OperationNode);
  let tripCount = graph.getNodeById("trip_count");
  let loopCond = graph.getNodeById("cond");
  let initialSum = graph.getNodeById("initial_sum");
  
  if (tripCount) {
    if (!tripCount.is(ConstantNode)) {
      tripCount.remove();
      tripCount = null;
    }
  }
  
  if (!tripCount) {
    tripCount = graph.addNode(formatId("trip_count", nodeId))
      .init(new ConstantNode.Builder(elemCount)).as(ConstantNode);
  }
  
  if (loopCond) {
    if (!loopCond.is(ConstantNode)) {
      loopCond.remove();
      loopCond = null;
    }
  }
  
  if (!loopCond) {
    loopCond = graph.addNode(formatId("cond", nodeId))
      .init(new ConstantNode.Builder(1)).as(ConstantNode);
  }
  
  if (!initialSum) {
    initialSum = edge0.source;
  }

  graph.addEdge(tripCount, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(loopCond, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(initialSum, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  
  const iterCount = graph.addNode(formatId("iter_count", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["int64"], "iter_count", "index")).as(VariableNode);
  
  const sumIn = graph.addNode(formatId("sum_in", nodeId), loopNode)
    .init(new VariableNode.Builder(type, "sum_in", "variable")).as(VariableNode);
  
  const condIn = graph.addNode(formatId("cond_in", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["bool"], "cond_in", "variable")).as(VariableNode);
  const addOp = graph.addNode(formatId("add", nodeId), loopNode)
    .init(new OperationNode.Builder("Add")).as(OperationNode);
  
  const identityOp = graph.addNode(formatId("identity", nodeId), loopNode)
    .init(new OperationNode.Builder("Identity")).as(OperationNode);
  
  graph.addEdge(iterCount, addOp).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(sumIn, addOp).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(condIn, identityOp).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  
  const condOut = graph.addNode(formatId("cond_out", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["bool"], "cond_out", "output")).as(VariableNode);
  
  const sumOut = graph.addNode(formatId("sum_out", nodeId), loopNode)
    .init(new VariableNode.Builder(type, "sum_out", "output")).as(VariableNode);

    graph.addEdge(identityOp, condOut).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(addOp, sumOut).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  
  let finalSum = graph.getNodeById("final_sum");
  
  if (finalSum) {
    if (!finalSum.is(TensorNode)) {
      finalSum.remove();
      finalSum = null;
    }
  }
  
  if (!finalSum) {
    finalSum = graph.addNode(formatId("final_sum", nodeId))
      .init(new TensorNode.Builder(type, shape, "output")).as(TensorNode);
  }

  node.outgoers.forEach(edge => {
    const target = edge.target;
    graph.addEdge(loopNode, target).init(
      new OnnxEdge.Builder(type, shape)
    ).as(OnnxEdge);

    edge.remove();
  });

  node.remove();
}