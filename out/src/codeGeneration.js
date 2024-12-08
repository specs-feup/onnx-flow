import TensorNode from "./Onnx/TensorNode.js";
import OperationNode from "./Onnx/OperationNode.js";
import ConstantNode from "./Onnx/ConstantNode.js";
import VariableNode from "./Onnx/VariableNode.js";
import OnnxEdge from "./Onnx/OnnxEdge.js";
import OnnxInnerEdge from "./Onnx/OnnxInnerEdge.js";
const variables = new Map();
const operations = new Map();
let indentationValue = "       ";
function handleOperation(graph, source, target, outputName) {
    let code = '';
    switch (source.type) {
        case 'Addition':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    variables.set(source.id, `(${variables.get(inputs[0])} + ${variables.get(inputs[1])})`);
                }
            }
            break;
        case 'Subtraction':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    variables.set(source.id, `(${variables.get(inputs[0])} - ${variables.get(inputs[1])})`);
                }
            }
            break;
        case 'Multiplication':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    variables.set(source.id, `(${variables.get(inputs[0])} * ${variables.get(inputs[1])})`);
                }
            }
            break;
        case 'Division':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    variables.set(source.id, `(${variables.get(inputs[0])} / ${variables.get(inputs[1])})`);
                }
            }
            break;
        case 'Load':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    const input0Node = graph.getNodeById(inputs[0]);
                    const input1Node = graph.getNodeById(inputs[1]);
                    if (input0Node && input1Node) {
                        if (input0Node.is(VariableNode)) {
                            if (input0Node.as(VariableNode).type === 'output') {
                                variables.set(source.id, `${outputName}[${variables.get(input1Node.id)}]`);
                            }
                            else
                                variables.set(source.id, `${variables.get(input0Node.id)}[${variables.get(input1Node.id)}]`);
                        }
                        else if (input1Node.is(VariableNode)) {
                            if (input1Node.as(VariableNode).type === 'output') {
                                variables.set(source.id, `${outputName}[${variables.get(input0Node.id)}]`);
                            }
                            else
                                variables.set(source.id, `${variables.get(input1Node.id)}[${variables.get(input0Node.id)}]`);
                        }
                    }
                }
            }
            break;
        case 'Not':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    variables.set(source.id, `!${variables.get(inputs[0])}`);
                }
            }
            break;
        case 'Equality':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    variables.set(source.id, `(${variables.get(inputs[0])} === ${variables.get(inputs[1])})`);
                }
            }
            break;
        case 'Store':
            if (!variables.get(source.id)) {
                const inputs = operations.get(source.id);
                if (inputs) {
                    if (target.is(VariableNode) && target.as(VariableNode).type === 'output') {
                        variables.set(source.id, indentationValue + `${target.id}[${variables.get(inputs[1])}] = ${variables.get(inputs[0])}\n`);
                        code += indentationValue + `${outputName}[${variables.get(inputs[0])}] = ${variables.get(inputs[1])}\n`;
                    }
                    else {
                        variables.set(source.id, indentationValue + `${target.id}[${variables.get(inputs[1])}] = ${variables.get(inputs[0])}\n`);
                        code += indentationValue + `${outputName}[${variables.get(inputs[0])}] = ${variables.get(inputs[1])}\n`;
                    }
                }
            }
            break;
    }
    return code;
}
function handleEdges(edge, graph, outputName) {
    let code = "";
    const source = edge.source;
    const target = edge.target;
    if (source.is(OperationNode)) {
        code += handleOperation(graph, source.as(OperationNode), target, outputName);
        if (target.is(VariableNode) && (target.as(VariableNode).type === 'index_aux' || target.as(VariableNode).type === 'index')) {
            code += `       ${target.id} = ${variables.get(source.id)}\n`;
        }
    }
    else if (source.is(ConstantNode)) {
        variables.set(source.id, source.as(ConstantNode).value.toString());
    }
    else if (source.is(VariableNode)) {
        if (source.as(VariableNode).type === 'input') {
            variables.set(source.id, `tensor_${source.as(VariableNode).name.substring(1)}`);
        }
        else
            variables.set(source.id, source.id);
    }
    const targetOperation = target.tryAs(OperationNode);
    if (targetOperation) {
        const targetOps = operations.get(targetOperation.id);
        if (targetOps) {
            targetOps.push(source.id);
        }
        else {
            operations.set(targetOperation.id, [source.id]);
        }
    }
    return code;
}
function handleOuterOperationNode(node, graph) {
    let code = "";
    const loopIterationsNode = graph.getNodeById(`Loop_iterations_${node.id}`)?.tryAs(ConstantNode);
    const outputOutgoers = node.outgoers.filter(edge => edge.target.is(TensorNode) && edge.target.as(TensorNode).type === 'output');
    const outgoers = node.outgoers;
    let outputName = "tensor_" + node.id;
    const orderedEdges = graph.edges.filter(edge => edge.source.parent?.tryAs(OperationNode)?.id === node.id)
        .filterIs(OnnxInnerEdge).sort((a, b) => a.order - b.order);
    if (loopIterationsNode && node.children.length) {
        let shape;
        if (outgoers.length && outgoers[0].is(OnnxEdge))
            shape = outgoers[0].as(OnnxEdge).shape;
        if (outputOutgoers.length) {
            outputName = outputOutgoers[0].target.id;
            if (outputOutgoers[0].is(OnnxEdge)) {
                shape = outputOutgoers[0].as(OnnxEdge).shape;
            }
        }
        const displacementInMemoryNode = graph.getNodeById(`displacementInMemory_${node.id}`);
        if (displacementInMemoryNode && shape) {
            if (displacementInMemoryNode.is(ConstantNode)) {
                const displacementInMemory = displacementInMemoryNode.as(ConstantNode).value;
                const totalElements = shape.reduce((acc, val) => acc * val, 1);
                code += `   let ${outputName} = {`;
                for (let i = 0; i < totalElements; i++) {
                    const index = i * displacementInMemory;
                    code += `${index}: 0, `;
                }
                code = code.slice(0, -2) + "};\n";
            }
        }
        else {
            code += `   let ${outputName} = {};\n`;
        }
        const indexNode = node.children.filterIs(VariableNode).filter(node => node.type === 'index');
        if (indexNode && indexNode.length === 1) {
            code += `   let ${indexNode[0].id} = 0\n`;
        }
        else
            return "";
        const indexAuxNodes = node.children.filterIs(VariableNode).filter(node => node.type === 'index_aux');
        if (indexAuxNodes) {
            indexAuxNodes.forEach(node => code += `   let ${node.id} = 0\n`);
        }
        if (loopIterationsNode) {
            code += `   while (${indexNode[0].id} < ${loopIterationsNode?.value}) {\n`;
        }
        orderedEdges.forEach(edge => {
            code += handleEdges(edge, graph, outputName);
        });
        code += "   }\n\n";
    }
    else {
        indentationValue = "   ";
        if (outputOutgoers.length) {
            outputName = outputOutgoers[0].target.id;
        }
        code += `   let ${outputName}= {0: 0};\n`;
        orderedEdges.forEach(edge => {
            code += handleEdges(edge, graph, outputName);
        });
        indentationValue = "        ";
    }
    console.log("\n");
    return code;
}
//posso dar replace do k pelo index na otimização do matmul
//ver ordem dos nos quando dou replace de nos
export function generateCode(graph) {
    let code = "function onnxGraph(";
    graph.nodes.filterIs(TensorNode).filter(node => node.type === 'input' && node.parent === undefined)
        .forEach(node => code += `tensor_${node.id}, `);
    code = code.slice(0, -2) + ') {\n\n';
    const outerNodes = graph.nodes.filterIs(OperationNode).filter(node => node.parent === undefined);
    outerNodes.forEach(node => {
        code += handleOuterOperationNode(node, graph);
    });
    code += "   return ";
    const outputNodes = graph.nodes.filterIs(TensorNode).filter(node => node.type === 'output');
    if (outputNodes.length > 1) {
        code += "{";
        outputNodes.forEach(node => {
            code += `${node.id}, `;
        });
        code = code.slice(0, -2);
        code += "};\n";
    }
    else if (outputNodes.length === 1) {
        code += `${outputNodes[0].id};\n`;
    }
    code += "\n}\n";
    return code;
}
//# sourceMappingURL=codeGeneration.js.map