import OnnxGraph from "../../../OnnxGraph.js";
import TensorNode from "../../../TensorNode.js";
import OperationNode from "../../../OperationNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import TensorSplitter from "../TensorSplitter.js";
import { int64Vec } from "@specs-feup/onnx-flow/Onnx/Utils";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";

export function decomposeMatMul(node: OperationNode.Class, g: OnnxGraph.Class, tensorSplitter: TensorSplitter): boolean {
  const [input1, input2] = node.getInputs().map((inp) => inp.as(TensorNode));
  const literalType = input1.literalType;

  if (input1.shape.length > 2 || input2.shape.length > 2) {
    throw new Error("MatMul decomposition for tensors with more than 2 dimensions is not supported.");
  }

  const canDivideFirst = input1.shape.length === 2;
  const canDivideSecond = input2.shape.length === 2;
  if (!canDivideFirst && !canDivideSecond) {
    return false;
  }

  // Create new input1 nodes
  const newInputs1: TensorNode.Class[] = tensorSplitter.getSplit(input1, false).splits;
  const numRows = newInputs1.length;

  // Create new input2 nodes
  const newInputs2: TensorNode.Class[] = tensorSplitter.getSplit(input2, true).splits;
  const numCols = newInputs2.length;

  const output = node.outgoers.at(0).target.as(TensorNode);

  // Create constant zero (for unsqueezes)
  const zeroBuilder = new TensorNode.Builder(
    DataType.INT64,
    [1],
    "constant",
    int64Vec([0]),
  );
  const zeroNode = g
    .addNode(`${node.id}_zero`, node.parent)
    .init(zeroBuilder)
    .as(TensorNode);

  // Organize outputs
  const newOutputs: TensorNode.Class[] = tensorSplitter.getSplit(output, false).splits;

  for (let row = 0; row < numRows; row++) {
    const unsqueezes: OperationNode.Class[] = [];

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

      // Create mul output tensor
      const mulOutputBuilder = new TensorNode.Builder(
        literalType,
        [input1.shape[1]],
        "intermediate",
      );
      const mulOutputNode = g
        .addNode(`${node.id}_intermediate${row}${col}`, node.parent)
        .init(mulOutputBuilder)
        .as(TensorNode);

      const mulOutputEdgeBuilder = new OnnxEdge.Builder(
        literalType,
        [input1.shape[1]],
      );
      g.addEdge(mulNode, mulOutputNode).init(mulOutputEdgeBuilder);

      // Create ReduceSum node
      const reduceSumBuilder = new OperationNode.Builder(
        "ReduceSum",
        [mulOutputNode],
        { keepdims: 0 },
      );
      const reduceSumNode = g
        .addNode(`${node.id}_reducesum${row}${col}`, node.parent)
        .init(reduceSumBuilder)
        .as(OperationNode);

      // Create ReduceSum output tensor
      const resultElementBuilder = new TensorNode.Builder(
        literalType,
        [],
        "intermediate",
      );
      const reduceSumOutput = g
        .addNode(`${node.id}_reducesum_out${row}${col}`, node.parent)
        .init(resultElementBuilder)
        .as(TensorNode);

      const reduceSumOutputEdgeBuilder = new OnnxEdge.Builder(
        literalType,
        [],
      );
      g.addEdge(reduceSumNode, reduceSumOutput).init(reduceSumOutputEdgeBuilder);

      // Create unsqueeze node
      const unsqueezeBuilder = new OperationNode.Builder("Unsqueeze", [
        reduceSumOutput,
        zeroNode,
      ]);
      const unsqueezeNode = g
        .addNode(`${node.id}_unsqueeze${row}${col}`, node.parent)
        .init(unsqueezeBuilder)
        .as(OperationNode);

      unsqueezes.push(unsqueezeNode);
    }

    const rowOutput = newOutputs[row];
    const unsqueezeOutputEdgeBuilder = new OnnxEdge.Builder(
      literalType,
      [1],
    );

    // Connect ReduceSum outputs to final output
    if (unsqueezes.length === 1) {
      // Directly connect if only one column
      g.addEdge(unsqueezes[0], rowOutput).init(unsqueezeOutputEdgeBuilder);

    } else {
      // Create unsqueeze outputs
      const unsqueezeOutputBuilder = new TensorNode.Builder(
        literalType,
        [1],
        "intermediate",
      );

      const unsqueezeOutputs = unsqueezes.map((unsq, index) => {
        const unsqOut = g
          .addNode(`${node.id}_unsq_out${row}${index}`, node.parent)
          .init(unsqueezeOutputBuilder)
          .as(TensorNode);

        g.addEdge(unsq, unsqOut).init(unsqueezeOutputEdgeBuilder);
        return unsqOut;
      });

      // Add Concat node
      const concatBuilder = new OperationNode.Builder("Concat", unsqueezeOutputs, {
        axis: 1,
      });
      const concatNode = g
        .addNode(`${node.id}_concat${row}`, node.parent)
        .init(concatBuilder)
        .as(OperationNode);

      const concatOutEdgeBuilder = new OnnxEdge.Builder(
        literalType,
        [numCols],
      );
      g.addEdge(concatNode, rowOutput).init(concatOutEdgeBuilder);
    }
  }

  // Remove original MatMul node and its edges
  for (const edge of [...node.incomers, ...node.outgoers]) {
    edge.remove();
  }
  node.remove();

  return true;
}
