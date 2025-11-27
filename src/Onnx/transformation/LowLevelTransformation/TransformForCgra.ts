import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import { DataType } from "../../OnnxTypes.js";
import { int64Vec } from "../Utilities.js";

// function getTopologicalOrder(g: OnnxGraph.Class): OperationNode.Class[] {
//     const visited = new Set<OperationNode.Class>();
//     const order: OperationNode.Class[] = [];

//     function dfs(node: OperationNode.Class) {
//         if (visited.has(node)) return;
//         visited.add(node);

//         const children = node.children.filterIs(OperationNode);
//         for (const child of children) {
//             dfs(child);
//         }

//         order.push(node);
//     }

//     const allNodes = g.getOperationNodes();
//     for (const node of allNodes) {
//         dfs(node);
//     }

//     return order.reverse();
// }

function splitInput(
  input: TensorNode.Class,
  g: OnnxGraph.Class,
  rowWise: boolean,
): TensorNode.Class[] {
  const newInputs: TensorNode.Class[] = [];
  const literalType = input.literalType;

  if (input.type !== "input") {
    const edgeBuilder = new OnnxEdge.Builder();

    const numDivs = rowWise
      ? (input.shape[0] as number)
      : (input.shape[1] as number);
    const newShape = rowWise ? [input.shape[1]] : [input.shape[0]];

    const inputBuilder = new TensorNode.Builder(
      literalType,
      newShape,
      "intermediate",
    );

    if (numDivs > 1) {
      // Need to split input into multiple parts
      const splitBuilder = new OperationNode.Builder("Split", [input], {
        axis: rowWise ? 0 : 1,
      });
      const split = g
        .addNode(`${input.id}_split`)
        .init(splitBuilder)
        .as(OperationNode);

      const splitOutBuilder = new TensorNode.Builder(
        literalType,
        rowWise ? [1, input.shape[1]] : [input.shape[0], 1],
        "intermediate",
      );

      for (let i = 0; i < numDivs; i++) {
        const splitOutput = g
          .addNode(`${input.id}${i}_unsqz`)
          .init(splitOutBuilder)
          .as(TensorNode);
        g.addEdge(split, splitOutput).init(edgeBuilder);

        const squeezeBuilder = new OperationNode.Builder("Squeeze", [
          splitOutput,
        ]);
        const squeeze = g
          .addNode(`${input.id}${i}_squeeze`)
          .init(squeezeBuilder)
          .as(OperationNode);

        const newInput = g
          .addNode(input.id + i.toString())
          .init(inputBuilder)
          .as(TensorNode);
        g.addEdge(squeeze, newInput).init(edgeBuilder);
        newInputs.push(newInput);
      }
    } else {
      // Just needs to squeeze
      const squeezeBuilder = new OperationNode.Builder("Squeeze", [input]);
      const squeeze = g
        .addNode(`${input.id}_squeeze`)
        .init(squeezeBuilder)
        .as(OperationNode);

      const newInput = g
        .addNode(input.id + "0")
        .init(inputBuilder)
        .as(TensorNode);
      g.addEdge(squeeze, newInput).init(edgeBuilder);
      newInputs.push(newInput);
    }
  } else {
    // For true input nodes inputs, return copies
    const numDivs = rowWise
      ? (input.shape[0] as number)
      : (input.shape[1] as number);
    const newShape = rowWise ? [input.shape[1]] : [input.shape[0]];

    if (numDivs > 1) {
      for (let i = 0; i < numDivs; i++) {
        const inputBuilder = new TensorNode.Builder(
          literalType,
          newShape,
          input.type,
        );
        const newInput = g
          .addNode(input.id + i.toString())
          .init(inputBuilder)
          .as(TensorNode);
        newInputs.push(newInput);
      }

      // Remove the original input node
      input.remove();
    } else {
      // Return original, no split needed
      newInputs.push(input);
    }
  }

  return newInputs;
}

function mergeOutputs(
  outputs: TensorNode.Class[],
  originalOutput: TensorNode.Class,
  g: OnnxGraph.Class,
) {
  const literalType = originalOutput.literalType;

  const edgeBuilder = new OnnxEdge.Builder();

  const oneConstBuilder = new TensorNode.Builder(
    DataType.INT64,
    [1],
    "constant",
    int64Vec([1]),
  );
  const oneConst = g
    .addNode(`${originalOutput.id}_one`)
    .init(oneConstBuilder)
    .as(TensorNode);

  const shapeBuilder = new TensorNode.Builder(
    DataType.INT64,
    [2],
    "constant",
    int64Vec(originalOutput.shape as number[]),
  );
  const shapeConst = g
    .addNode(`${originalOutput.id}_shape`)
    .init(shapeBuilder)
    .as(TensorNode);

  // Add Unsqueeze nodes
  const unsqOuts: TensorNode.Class[] = [];
  for (let i = 0; i < outputs.length; i++) {
    const unsqueezeBuilder = new OperationNode.Builder("Unsqueeze", [
      outputs[i],
      oneConst,
    ]);
    const unsqueeze = g
      .addNode(`${outputs[i].id}_unsqueeze`)
      .init(unsqueezeBuilder)
      .as(OperationNode);

    const unsqOutBuilder = new TensorNode.Builder(
      literalType,
      [1, 1],
      "intermediate",
    );
    const unsqOut = g
      .addNode(`${outputs[i].id}_unsq_out`)
      .init(unsqOutBuilder)
      .as(TensorNode);
    g.addEdge(unsqueeze, unsqOut).init(edgeBuilder);

    unsqOuts.push(unsqOut);
  }

  // Add Concat node
  const concatBuilder = new OperationNode.Builder("Concat", unsqOuts, {
    axis: 0,
  });
  const concat = g
    .addNode(`${originalOutput.id}_concat`)
    .init(concatBuilder)
    .as(OperationNode);

  // Add Concat output node
  const concatOutBuilder = new TensorNode.Builder(
    literalType,
    [outputs.length],
    "intermediate",
  );
  const concatOut = g
    .addNode(`${originalOutput.id}_concat_out`)
    .init(concatOutBuilder)
    .as(TensorNode);
  g.addEdge(concat, concatOut).init(edgeBuilder);

  // Add Reshape node to get original output shape
  const reshapeBuilder = new OperationNode.Builder("Reshape", [
    concatOut,
    shapeConst,
  ]);
  const reshape = g
    .addNode(`${originalOutput.id}_reshape`)
    .init(reshapeBuilder)
    .as(OperationNode);

  g.addEdge(reshape, originalOutput).init(edgeBuilder);
}

function divideMatMul(node: OperationNode.Class, g: OnnxGraph.Class): boolean {
  const [input1, input2] = node.getInputs().map((inp) => inp.as(TensorNode));
  const literalType = input1.literalType;
  const edgeBuilder = new OnnxEdge.Builder();

  const canDivideFirst = input1.shape.length == 2;
  const canDivideSecond = input2.shape.length == 2;
  if (!canDivideFirst && !canDivideSecond) {
    return false;
  }

  // Create new input1 nodes
  const newInputs1: TensorNode.Class[] = splitInput(input1, g, true);
  const numRows = newInputs1.length;

  // Create new input2 nodes
  const newInputs2: TensorNode.Class[] = splitInput(input2, g, false);
  const numCols = newInputs2.length;

  // Create new operation nodes and outputs
  const muls: OperationNode.Class[] = [];
  const intermediates: TensorNode.Class[] = [];
  const reduceSums: OperationNode.Class[] = [];

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
      muls.push(mulNode);

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
      intermediates.push(intermediateNode);

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
      reduceSums.push(reduceSumNode);

      // Connect Mul to intermediate
      g.addEdge(mulNode, intermediateNode).init(edgeBuilder);

      // Connect ReduceSum to output
      const output = node.outgoers.at(0).target.as(TensorNode);
      const newOutput = g
        .addNode(`${output.id}${row}${col}`)
        .init(outputBuilder)
        .as(TensorNode);
      newOutputs.push(newOutput);

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

  // // Remove original node, input and output
  // output.remove();
  // node.remove();
  // input1.remove();
  // input2.remove();

  // Remove original MatMul node and its edges
  for (const edge of [...node.incomers, ...node.outgoers]) {
    edge.remove();
  }
  node.remove();

  return true;
}

export default function transformForCgra(g: OnnxGraph.Class) {
  let done = false;

  while (!done) {
    for (const node of g.getOperationNodes()) {
      if (node.type === "MatMul" && divideMatMul(node, g)) {
        // continue;
      }
    }

    done = true;
  }
}
