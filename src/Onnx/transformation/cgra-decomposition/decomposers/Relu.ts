import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { mergeOutputs, splitInput } from "../utils.js";
import { makeTensorProto } from "../../../Utils.js";
import { DataType } from "../../../OnnxTypes.js";

export default function decomposeRelu(node: OperationNode.Class, g: OnnxGraph.Class): boolean {
  const [input] = node.getInputs().map((inp) => inp.as(TensorNode));
  const literalType = input.literalType;
  const edgeBuilder = new OnnxEdge.Builder();

  if (input.shape.length > 2) {
    throw new Error("Relu decomposition for tensors with more than 2 dimensions is not supported.");
  }

  const inputs: TensorNode.Class[] = input.shape.length === 2
    ? splitInput(input, g, true)
    : [input];

  // Create constant zero (for comparisons)
  const reluZeroBuilder = new TensorNode.Builder(
    literalType,
    [],
    "constant",
    makeTensorProto(literalType, [], [0]),
  );

  const reluZero = g.addNode(`${node.id}_zero`, node.parent)
    .init(reluZeroBuilder)
    .as(TensorNode);

  const newOutputs: TensorNode.Class[] = [];
  const output = node.outgoers.at(0).target.as(TensorNode);
  const outputBuilder = new TensorNode.Builder(
    output.literalType,
    output.shape.slice(1),
    output.type,
  );

  for (let i = 0; i < inputs.length; i++) {
    // Create Greater node (serving as ">0" node)
    const greaterBuilder = new OperationNode.Builder("Greater", [
      inputs[i],
      reluZero,
    ]);
    const greaterNode = g
      .addNode(`${node.id}_greater${i}`, node.parent)
      .init(greaterBuilder)
      .as(OperationNode);

    const greaterOutBuilder = new TensorNode.Builder(
      DataType.BOOL,
      inputs[i].shape.slice(),
      "intermediate",
    );
    const greaterOut = g
      .addNode(`${node.id}_greater_out${i}`, node.parent)
      .init(greaterOutBuilder)
      .as(TensorNode);

    // Create Where node (serving as "mux" node)
    const whereBuilder = new OperationNode.Builder("Where", [
      greaterOut,
      inputs[i],
      reluZero,
    ]);
    const whereNode = g
      .addNode(`${node.id}_where${i}`, node.parent)
      .init(whereBuilder)
      .as(OperationNode);

    const newOutput = g
      .addNode(`${output.id}${i}`, node.parent)
      .init(outputBuilder)
      .as(TensorNode);
    newOutputs.push(newOutput);

    // Connect operation nodes to results
    g.addEdge(greaterNode, greaterOut).init(edgeBuilder);
    g.addEdge(whereNode, newOutput).init(edgeBuilder);
  }

  // Merge outputs
  if (output.type !== "output") {
    mergeOutputs(newOutputs, output, g);
  } else {
    // Otherwise, just replace the output
    output.remove();
  }

  // Remove original Relu node and its edges
  for (const edge of [...node.incomers, ...node.outgoers]) {
    edge.remove();
  }
  node.remove();

  // Relu decomposition is not supported
  return true;
}
