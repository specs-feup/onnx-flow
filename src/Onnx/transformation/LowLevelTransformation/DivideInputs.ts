import OnnxGraph from "../../OnnxGraph.js";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";

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

function divideMatMul(
    node: OperationNode.Class,
    g: OnnxGraph.Class
): boolean {
    const [input1, input2] = node.getInputs().map(inp => inp.as(TensorNode));

    if (input1.shape.length != 2 || input1.shape[0] === 1) return false;

    // Create new input nodes
    const numRows = input1.shape[0] as number;
    const newInputs: TensorNode.Class[] = [];

    for (let i = 0; i < numRows; i++) {
        const tensorBuilder = new TensorNode.Builder(
            input1.literalType,
            [1, input1.shape[1]],
            input1.type,
        );
        const newInput = g.addNode(input1.id + i.toString(), input1.parent).init(tensorBuilder).as(TensorNode);
        newInputs.push(newInput);
    }

    // Create new MatMul nodes
    const newMatMuls: OperationNode.Class[] = [];

    for (let i = 0; i < numRows; i++) {
        const matMulBuilder = new OperationNode.Builder("MatMul", [newInputs[i], input2]);
        const newMatMul = g.addNode(node.id + i.toString(), node.parent).init(matMulBuilder).as(OperationNode);
        newMatMuls.push(newMatMul);
    }

    // Create new output tensor nodes
    const output = node.outgoers.at(0).target.as(TensorNode);
    const newOutputs: TensorNode.Class[] = [];

    for (let i = 0; i < numRows; i++) {
        const outputTensorBuilder = new TensorNode.Builder(
            output.literalType,
            [1, output.shape[1]],
            output.type,
        );
        const edgeBuilder = new OnnxEdge.Builder();

        const newOutput = g.addNode(output.id + i.toString(), output.parent).init(outputTensorBuilder).as(TensorNode);
        g.addEdge(newMatMuls[i], newOutput).init(edgeBuilder);
        newOutputs.push(newOutput);
    }

    // Remove original node, first input and output
    output.remove();
    node.remove();
    input1.remove();

    return true;
}

export default function divideInputs(g: OnnxGraph.Class) {
    let done = false;

    while (!done) {
        for (const node of g.getOperationNodes()) {
            if (node.type === "MatMul" && divideMatMul(node, g)) {
                continue;
            }
        }

        done = true;
    }
}