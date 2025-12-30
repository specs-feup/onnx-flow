import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { mergeOutputs, splitInput } from "../utils.js";

export function decomposeMatMul(node: OperationNode.Class, g: OnnxGraph.Class): boolean {
  const [input1, input2] = node.getInputs().map((inp) => inp.as(TensorNode));
  const literalType = input1.literalType;
  const edgeBuilder = new OnnxEdge.Builder();

  if (input1.shape.length > 2 || input2.shape.length > 2) {
    throw new Error("MatMul decomposition for tensors with more than 2 dimensions is not supported.");
  }

  const canDivideFirst = input1.shape.length === 2;
  const canDivideSecond = input2.shape.length === 2;
  if (!canDivideFirst && !canDivideSecond) {
    return false;
  }

  // Create new input1 nodes
  const newInputs1: TensorNode.Class[] = splitInput(input1, g, true);
  const numRows = newInputs1.length;

  // Create new input2 nodes
  const newInputs2: TensorNode.Class[] = splitInput(input2, g, false);
  const numCols = newInputs2.length;

  const newOutputs: TensorNode.Class[] = [];
  const output = node.outgoers.at(0).target.as(TensorNode);
  const outputBuilder = new TensorNode.Builder(
    output.literalType,
    [],
    output.type,
  );

  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col < numCols; col++) {
      // Create Mul node
      const mulBuilder = new OperationNode.Builder("Mul", [
        newInputs1[row],
        newInputs2[col],
      ]);

      const mulNode = g
        .addNode(`${node.id}_mul${row}${col}`, node.parent)
        .init(mulBuilder)
        .as(OperationNode);

      // Create intermediate tensor node
      const intermediateBuilder = new TensorNode.Builder(
        literalType,
        [input1.shape[1]],
        "intermediate",
      );
      const intermediateNode = g
        .addNode(`${node.id}_intermediate${row}${col}`, node.parent)
        .init(intermediateBuilder)
        .as(TensorNode);

      // Create ReduceSum node
      const reduceSumBuilder = new OperationNode.Builder(
        "ReduceSum",
        [intermediateNode],
        { keepdims: 0 },
      );
      const reduceSumNode = g
        .addNode(`${node.id}_reducesum${row}${col}`, node.parent)
        .init(reduceSumBuilder)
        .as(OperationNode);

      // Create output tensor node
      const newOutput = g
        .addNode(`${output.id}${row}${col}`)
        .init(outputBuilder)
        .as(TensorNode);
      newOutputs.push(newOutput);

      // Connect operation nodes to results
      g.addEdge(mulNode, intermediateNode).init(edgeBuilder);
      g.addEdge(reduceSumNode, newOutput).init(edgeBuilder);
    }
  }

  // Merge outputs back to original output, if output is not the final tensor
  if (output.type !== "output") {
    mergeOutputs(newOutputs, output, g);
  } else {
    // Otherwise, just replace the output
    output.remove();
  }

  // Remove original MatMul node and its edges
  for (const edge of [...node.incomers, ...node.outgoers]) {
    edge.remove();
  }
  node.remove();

  return true;
}
