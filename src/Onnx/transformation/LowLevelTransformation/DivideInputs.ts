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
    const literalType = input1.literalType;

    const canDivideFirst = input1.shape.length == 2 && input1.shape[0] as number > 1;
    const canDivideSecond = input2.shape.length == 2 && input2.shape[0] as number > 1;
    if (!canDivideFirst && !canDivideSecond) {
        return false;
    }

    // Create new input1 nodes
    const numRows = input1.shape[0] as number;
    const newInputs1: TensorNode.Class[] = [];
    const input1Builder = new TensorNode.Builder(
        literalType,
        [input1.shape[1]],
        input1.type,
    );

    for (let i = 0; i < numRows; i++) {
        const newInput = g.addNode(input1.id + i.toString(), input1.parent).init(input1Builder).as(TensorNode);
        newInputs1.push(newInput);
    }

    // Create new input2 nodes
    const numCols = input2.shape[1] as number;
    const newInputs2: TensorNode.Class[] = [];
    const input2Builder = new TensorNode.Builder(
        literalType,
        [input2.shape[0]],
        input2.type,
    );

    for (let i = 0; i < numCols; i++) {
        const newInput = g.addNode(input2.id + i.toString(), input2.parent).init(input2Builder).as(TensorNode);
        newInputs2.push(newInput);
    }

    // // Create new MatMul nodes
    // const newMatMuls: OperationNode.Class[] = [];

    // for (let row = 0; row < numRows; row++) {
    //     for (let col = 0; col < numCols; col++) {
    //         const matMulBuilder = new OperationNode.Builder("MatMul", [newInputs1[row], newInputs2[col]]);
    //         const newMatMul = g.addNode(node.id + row.toString() + col.toString(), node.parent).init(matMulBuilder).as(OperationNode);
    //         newMatMuls.push(newMatMul);
    //     }
    // }

    // // Create new output tensor nodes
    // const output = node.outgoers.at(0).target.as(TensorNode);
    // const newOutputs: TensorNode.Class[] = [];
    // const outputBuilder = new TensorNode.Builder(
    //     output.literalType,
    //     [],
    //     output.type,
    // );
    // const edgeBuilder = new OnnxEdge.Builder();

    // for (let row = 0; row < numRows; row++) {
    //     for (let col = 0; col < numCols; col++) {
    //         const newOutput = g.addNode(output.id + row.toString() + col.toString()).init(outputBuilder).as(TensorNode);
    //         newOutputs.push(newOutput);

    //         // Connect new MatMul to new output
    //         const correspondingMatMul = newMatMuls[row * numCols + col];
    //         g.addEdge(correspondingMatMul, newOutput).init(edgeBuilder);
    //     }
    // }

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
    const edgeBuilder = new OnnxEdge.Builder();

    for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < numCols; col++) {
            // Create Mul node
            const mulBuilder = new OperationNode.Builder("Mul", [newInputs1[row], newInputs2[col]]);
            const mulNode = g.addNode(
                `${node.id}_mul${row}${col}`,
                node.parent
            ).init(mulBuilder).as(OperationNode);
            muls.push(mulNode);

            // Create intermediate tensor node
            const intermediateBuilder = new TensorNode.Builder(
                literalType,
                [input1.shape[1]],
                "intermediate",
            );
            const intermediateNode = g.addNode(
                `${node.id}_intermediate${row}${col}`,
                node.parent
            ).init(intermediateBuilder).as(TensorNode);
            intermediates.push(intermediateNode);

            // Create ReduceSum node
            const reduceSumBuilder = new OperationNode.Builder("ReduceSum", [intermediateNode]);
            const reduceSumNode = g.addNode(
                `${node.id}_reducesum${row}${col}`,
                node.parent
            ).init(reduceSumBuilder).as(OperationNode);
            reduceSums.push(reduceSumNode);

            // Connect Mul to intermediate
            g.addEdge(mulNode, intermediateNode).init(edgeBuilder);

            // Connect ReduceSum to output
            const output = node.outgoers.at(0).target.as(TensorNode);
            const outputBuilder = new TensorNode.Builder(
                output.literalType,
                [],
                output.type,
            );
            const newOutput = g.addNode(
                `${output.id}${row}${col}`,
            ).init(outputBuilder).as(TensorNode);
            newOutputs.push(newOutput);

            g.addEdge(reduceSumNode, newOutput).init(edgeBuilder);
        }
    }

    // Remove original node, input and output
    output.remove();
    node.remove();
    input1.remove();
    input2.remove();

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