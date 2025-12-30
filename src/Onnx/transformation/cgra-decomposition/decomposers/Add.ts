import OnnxEdge from "../../../OnnxEdge.js";
import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { mergeOutputs, shapesEqual, splitInput } from "../utils.js";

export default function decomposeAdd(node: OperationNode.Class, g: OnnxGraph.Class): boolean {
  const [input1, input2] = node.getInputs().map((inp) => inp.as(TensorNode));
  const edgeBuilder = new OnnxEdge.Builder();

  if (!shapesEqual(input1, input2)) {
    throw new Error("Add decomposition is only supported for inputs with the same shape.");
  }

  const canDivide = input1.shape.length === 2;
  if (!canDivide) {
    return false;
  }

  const newInputs1: TensorNode.Class[] = splitInput(input1, g, true);
  const newInputs2: TensorNode.Class[] = splitInput(input2, g, true);

  const newOutputs: TensorNode.Class[] = [];
  const output = node.outgoers.at(0).target.as(TensorNode);
  const outputBuilder = new TensorNode.Builder(
    output.literalType,
    [],
    output.type,
  );

  for (let i = 0; i < newInputs1.length; i++) {
    // Create Add node
    const addBuilder = new OperationNode.Builder("Add", [
      newInputs1[i],
      newInputs2[i],
    ]);

    const addNode = g
      .addNode(`${node.id}_add${i}`, node.parent)
      .init(addBuilder)
      .as(OperationNode);

    // Create output tensor node
    const newOutput = g
      .addNode(`${output.id}${i}`)
      .init(outputBuilder)
      .as(TensorNode);
    newOutputs.push(newOutput);

    // Connect operation nodes to results
    g.addEdge(addNode, newOutput).init(edgeBuilder);
  }

  // Merge outputs
  if (output.type !== "output") {
    mergeOutputs(newOutputs, output, g);
  } else {
    // Otherwise, just replace the output
    output.remove();
  }

  // Remove original Add node and its edges
  for (const edge of [...node.incomers, ...node.outgoers]) {
    edge.remove();
  }
  node.remove();

  return true;
}