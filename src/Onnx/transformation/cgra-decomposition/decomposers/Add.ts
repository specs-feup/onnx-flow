import OnnxEdge from "../../../OnnxEdge.js";
import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import TensorSplitter from "../TensorSplitter.js";
import { shapesEqual } from "../utils.js";

export default function decomposeAdd(
    node: OperationNode.Class,
    g: OnnxGraph.Class,
    tensorSplitter: TensorSplitter,
): boolean {
    const [input1, input2] = node.getInputs().map((inp) => inp.as(TensorNode));

    if (!shapesEqual(input1, input2)) {
        throw new Error("Add decomposition is only supported for inputs with the same shape.");
    }

    const canDivide = input1.shape.length === 2;
    if (!canDivide) {
        return false;
    }

    const newInputs1: TensorNode.Class[] = tensorSplitter.getSplit(input1, false).splits;
    const newInputs2: TensorNode.Class[] = tensorSplitter.getSplit(input2, false).splits;

    const output = node.outgoers.at(0).target.as(TensorNode);
    const outputs = tensorSplitter.getSplit(output, false).splits;

    for (let i = 0; i < newInputs1.length; i++) {
        // Create Add node
        const addBuilder = new OperationNode.Builder("Add", [newInputs1[i], newInputs2[i]]);

        const addNode = g
            .addNode(`${node.id}_add${i}`, node.parent)
            .init(addBuilder)
            .as(OperationNode);

        // Connect Add output to final output tensor
        const edgeBuilder = new OnnxEdge.Builder(output.literalType, output.shape.slice(1));
        g.addEdge(addNode, outputs[i]).init(edgeBuilder);
    }

    // Remove original Add node and its edges
    for (const edge of [...node.incomers, ...node.outgoers]) {
        edge.remove();
    }
    node.remove();

    return true;
}
