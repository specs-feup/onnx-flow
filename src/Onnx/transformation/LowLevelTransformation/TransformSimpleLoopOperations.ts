import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js";
import ConstantNode from "../../ConstantNode.js";
import VariableNode from "../../VariableNode.js";
import { typeSizeMap, formatId } from "../Utilities.js";
import OnnxInnerEdge from "../../OnnxInnerEdge.js";

export default function transformSimpleLoopOperations(node: OperationNode.Class, graph: OnnxGraph.Class): void {
  const supportedOps = ["Add", "Sub", "Mul", "Div"];
  if (!supportedOps.includes(node.type)) return;

  const nodeId = node.id;
  const edges = node.incomers.filterIs(OnnxEdge);
  //if (edges.length !== 2) return;

  const [edge0, edge1] = edges;
  const type = edge0.literalType;
  if (type === undefined) return;

  const shape = edge0.shape;
  const elemCount = shape.reduce((acc, dim) => acc * dim, 1);
  const displacement = typeSizeMap[type];
  const typeId = typeSizeMap[type];

  let order = 0;

  // Trip count and initial condition for Loop
  const tripCount = graph.addNode(formatId("trip_count", nodeId))
    .init(new ConstantNode.Builder(elemCount)).as(ConstantNode);
  const loopCond = graph.addNode(formatId("cond", nodeId))
    .init(new ConstantNode.Builder(1)).as(ConstantNode);

  // Initial state tensor (zeros)
  const initialState = graph.addNode(formatId("initialState", nodeId))
    .init(new TensorNode.Builder(order++, new Array(elemCount).fill(0), "input")).as(TensorNode);

  // Loop node
  const loopNode = graph.addNode(formatId("Loop", nodeId))
    .init(new OperationNode.Builder("Loop")).as(OperationNode);

  graph.addEdge(tripCount, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(loopCond, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(initialState, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Loop body (inlined inside same graph for now)
  const iterIdx = graph.addNode(formatId("iter", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["int64"], "i", "index")).as(VariableNode);

  const unsqueeze = graph.addNode(formatId("unsqueeze", nodeId), loopNode)
    .init(new OperationNode.Builder("Unsqueeze")).as(OperationNode);
  graph.addEdge(iterIdx, unsqueeze).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Gather input 0
  const gather0 = graph.addNode(formatId("gather0", nodeId), loopNode)
    .init(new OperationNode.Builder("Gather")).as(OperationNode);
  graph.addEdge(edge0.source, gather0).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, gather0).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Gather input 1
  const gather1 = graph.addNode(formatId("gather1", nodeId), loopNode)
    .init(new OperationNode.Builder("Gather")).as(OperationNode);
  graph.addEdge(edge1.source, gather1).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, gather1).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Operation node (Add/Sub/Mul/Div)
  const op = graph.addNode(formatId("compute", nodeId), loopNode)
    .init(new OperationNode.Builder(node.type)).as(OperationNode);
  graph.addEdge(gather0, op).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(gather1, op).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Scatter result into output
  const scatter = graph.addNode(formatId("scatter", nodeId), loopNode)
    .init(new OperationNode.Builder("ScatterElements")).as(OperationNode);

  graph.addEdge(initialState, scatter).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, scatter).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(op, scatter).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  node.outgoers.forEach(edge => {
    const target = edge.target;
    graph.addEdge(loopNode, target).init(
        new OnnxEdge.Builder(type, shape)
    ).as(OnnxEdge);

    edge.remove(); // remove old edge from original op to target
  });

  node.remove();
}