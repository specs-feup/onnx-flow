import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js";
import { DataType } from "../../OnnxTypes.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import { int64Vec } from "../../Utils.js";

/**
 * @brief Splits a 2D tensor input node into multiple 1D tensor nodes either row-wise or column-wise.
 *
 * @param {TensorNode.Class} input - The input tensor node to be split
 * @param {OnnxGraph.Class} g - The ONNX graph to which the nodes belong
 * @param {boolean} rowWise - If true, splits the input row-wise; otherwise, splits column-wise
 * @returns {TensorNode.Class[]} An array of new tensor nodes resulting from the split
 */
export function splitInput(
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
        .addNode(`${input.id}_split`, input.parent)
        .init(splitBuilder)
        .as(OperationNode);

      const splitOutBuilder = new TensorNode.Builder(
        literalType,
        rowWise ? [1, input.shape[1]] : [input.shape[0], 1],
        "intermediate",
      );

      for (let i = 0; i < numDivs; i++) {
        const splitOutput = g
          .addNode(`${input.id}${i}_unsqz`, input.parent)
          .init(splitOutBuilder)
          .as(TensorNode);
        g.addEdge(split, splitOutput).init(edgeBuilder);

        const squeezeBuilder = new OperationNode.Builder("Squeeze", [
          splitOutput,
        ]);
        const squeeze = g
          .addNode(`${input.id}${i}_squeeze`, input.parent)
          .init(squeezeBuilder)
          .as(OperationNode);

        const newInput = g
          .addNode(input.id + i.toString(), input.parent)
          .init(inputBuilder)
          .as(TensorNode);
        g.addEdge(squeeze, newInput).init(edgeBuilder);
        newInputs.push(newInput);
      }
    } else {
      // Just needs to squeeze
      const squeezeBuilder = new OperationNode.Builder("Squeeze", [input]);
      const squeeze = g
        .addNode(`${input.id}_squeeze`, input.parent)
        .init(squeezeBuilder)
        .as(OperationNode);

      const newInput = g
        .addNode(input.id + "0", input.parent)
        .init(inputBuilder)
        .as(TensorNode);
      g.addEdge(squeeze, newInput).init(edgeBuilder);
      newInputs.push(newInput);
    }
  } else {
    // For true input nodes, return copies
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
          .addNode(input.id + i.toString(), input.parent)
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

/**
 * @brief Merges multiple output scalar nodes into a single tensor node with the original output shape.
 *
 * @param {TensorNode.Class[]} outputs - The array of output nodes to be merged.
 * @param {TensorNode.Class} originalOutput - The original output tensor node to restore the shape.
 * @param {OnnxGraph.Class} g - The ONNX graph to which the nodes belong.
 */
export function mergeOutputs(
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
    .addNode(`${originalOutput.id}_one`, originalOutput.parent)
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
      .addNode(`${outputs[i].id}_unsqueeze`, originalOutput.parent)
      .init(unsqueezeBuilder)
      .as(OperationNode);

    const unsqOutBuilder = new TensorNode.Builder(
      literalType,
      [1, 1],
      "intermediate",
    );
    const unsqOut = g
      .addNode(`${outputs[i].id}_unsq_out`, originalOutput.parent)
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
    .addNode(`${originalOutput.id}_concat_out`, originalOutput.parent)
    .init(concatOutBuilder)
    .as(TensorNode);
  g.addEdge(concat, concatOut).init(edgeBuilder);

  // Add Reshape node to get original output shape
  const reshapeBuilder = new OperationNode.Builder("Reshape", [
    concatOut,
    shapeConst,
  ]);

  const reshape = g
    .addNode(`${originalOutput.id}_reshape`, originalOutput.parent)
    .init(reshapeBuilder)
    .as(OperationNode);

  g.addEdge(reshape, originalOutput).init(edgeBuilder);
}