import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js";
import ConstantNode from "../../ConstantNode.js";
import VariableNode from "../../VariableNode.js";
import { typeSizeMap, formatId } from "../Utilities.js";
import OnnxInnerEdge from "../../OnnxInnerEdge.js";

/**
 * Transforms a chain of three connected Add operations into a single Loop
 * @param addNodes Array of three connected Add operation nodes
 * @param graph The ONNX graph
 */
export default function transformChain(addNodes: OperationNode.Class[], graph: OnnxGraph.Class): void {
  const firstAddNode = addNodes[0];
  const secondAddNode = addNodes[1];
  const thirdAddNode = addNodes[2];
  
  const nodeId = firstAddNode.id; // Use first node's ID as base
  
  // Find the 4 inputs for our chain
  // The first two inputs come from the first Add node
  const firstAddInputs = firstAddNode.incomers.filterIs(OnnxEdge);
  
  // The second Add node has two inputs from the second Add node
  const secondAddInputs = secondAddNode.incomers.filterIs(OnnxEdge);

  // Use first input edge for type and shape information
  const type = firstAddInputs[0].literalType;
  if (type === undefined) return;

  const shape = firstAddInputs[0].shape;
  const elemCount = shape.reduce((acc, dim) => acc * dim, 1);
  const typeId = typeSizeMap[type];

  let order = 0;

  // Create the loop node structure
  const tripCount = graph.addNode(formatId("trip_count", nodeId))
    .init(new ConstantNode.Builder(elemCount)).as(ConstantNode);
  const loopCond = graph.addNode(formatId("cond", nodeId))
    .init(new ConstantNode.Builder(1)).as(ConstantNode);
    
  // Loop node
  const loopNode = graph.addNode(formatId("Loop", nodeId))
    .init(new OperationNode.Builder("Loop")).as(OperationNode);
  
  // Loop condition handling
  const condIn = graph.addNode(formatId("cond_in", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["bool"], "cond_in", "output")).as(VariableNode);
  const identityOp = graph.addNode(formatId("identity", nodeId), loopNode)
    .init(new OperationNode.Builder("Identity")).as(OperationNode);
  const condOut = graph.addNode(formatId("cond_out", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["bool"], "cond_out", "output")).as(VariableNode);
  
  graph.addEdge(condIn, identityOp).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(identityOp, condOut).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  
  graph.addEdge(tripCount, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(loopCond, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Loop iteration index
  const iterIdx = graph.addNode(formatId("iter", nodeId), loopNode)
    .init(new VariableNode.Builder(typeSizeMap["int64"], "iter", "index")).as(VariableNode);

  const unsqueeze = graph.addNode(formatId("unsqueeze", nodeId), loopNode)
    .init(new OperationNode.Builder("Unsqueeze")).as(OperationNode);
  graph.addEdge(iterIdx, unsqueeze).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Gather all four inputs
  // Inputs from first Add node
  const gather0 = graph.addNode(formatId("gather0", nodeId), loopNode)
    .init(new OperationNode.Builder("Gather")).as(OperationNode);
  graph.addEdge(firstAddInputs[0].source, gather0).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, gather0).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  const gather1 = graph.addNode(formatId("gather1", nodeId), loopNode)
    .init(new OperationNode.Builder("Gather")).as(OperationNode);
  graph.addEdge(firstAddInputs[1].source, gather1).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, gather1).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Input from second Add node
  const gather2 = graph.addNode(formatId("gather2", nodeId), loopNode)
    .init(new OperationNode.Builder("Gather")).as(OperationNode);
  graph.addEdge(secondAddInputs[0].source, gather2).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, gather2).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Input from third Add node
  const gather3 = graph.addNode(formatId("gather3", nodeId), loopNode)
    .init(new OperationNode.Builder("Gather")).as(OperationNode);
  graph.addEdge(secondAddInputs[1].source, gather3).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(unsqueeze, gather3).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Create the chain of Add operations within the loop
  // First Add: A + B
  const op1 = graph.addNode(formatId("op1", nodeId), loopNode)
    .init(new OperationNode.Builder(firstAddNode.type)).as(OperationNode);
  graph.addEdge(gather0, op1).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(gather1, op1).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Second Add: C + D
  const op2 = graph.addNode(formatId("op2", nodeId), loopNode)
    .init(new OperationNode.Builder(secondAddNode.type)).as(OperationNode);
  graph.addEdge(gather2, op2).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(gather3, op2).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Third Add: (A+B) + (C+D)
  const op3 = graph.addNode(formatId("op3", nodeId), loopNode)
    .init(new OperationNode.Builder(thirdAddNode.type)).as(OperationNode);
  graph.addEdge(op1, op3).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(op2, op3).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Scatter the result
  const scatter = graph.addNode(formatId("scatter", nodeId), loopNode)
    .init(new OperationNode.Builder("ScatterElements")).as(OperationNode);

  const carry = graph.addNode(formatId("carry", nodeId), loopNode)
    .init(new VariableNode.Builder(typeId, "carry", "output")).as(VariableNode);
  const carryOut = graph.addNode(formatId("carry_out", nodeId), loopNode)
    .init(new VariableNode.Builder(typeId, "carry_out", "output")).as(VariableNode);

  graph.addEdge(unsqueeze, scatter).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(op3, scatter).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(carry, scatter).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
  graph.addEdge(scatter, carryOut).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

  // Connect the loop output to all targets of the last Add node
  thirdAddNode.outgoers.forEach(edge => {
    const target = edge.target;
    graph.addEdge(loopNode, target).init(
      new OnnxEdge.Builder(type, shape)
    ).as(OnnxEdge);
    
    edge.remove(); // Remove the original edge
  });

  // Remove all Add nodes since they're now represented inside the loop
  addNodes.forEach(node => {
    if (graph.hasNode(node.id)) {
      node.remove();
    }
  });
  
  graph.nodes.forEach(node => {
    const inputNodes = graph.getInputTensorNodes();
    const outputNodes = graph.getOutputTensorNodes();

    if (inputNodes.some(input => input.id === node.id)) return;
    if (outputNodes.some(output => output.id === node.id)) return;
    if (graph.nodes.filterIs(OperationNode).some(opNode => opNode.id === node.id)) return;
    if (graph.nodes.filterIs(VariableNode).some(opNode => opNode.id === node.id)) return;
    if (graph.nodes.filterIs(ConstantNode).some(opNode => opNode.id === node.id)) return;

    node.remove();
  });
}
