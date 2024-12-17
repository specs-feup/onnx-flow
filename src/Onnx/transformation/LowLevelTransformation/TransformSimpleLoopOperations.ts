import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js"
import ConstantNode from "../../ConstantNode.js";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import VariableNode from "../../VariableNode.js";
import { typeSizeMap, formatId } from "../Utilities.js";
import OnnxInnerEdge from "../../OnnxInnerEdge.js";


export default function transformSimpleLoopOperations(node: OperationNode.Class, graph: OnnxGraph.Class): void {

    let order = 0;

    let opType : string

    switch (node.type) {
        case "Add":
            opType = 'Addition'
            break
        case "Sub":
            opType = 'Subtraction'
            break
        case "Mul":
            opType = 'Multiplication'
            break
        case "Div":
            opType = 'Division'
            break
        default:
            return;
    }

    const nodeId : string = node.id

    const incomingEdges = node.incomers.filterIs(OnnxEdge);
    
    if (incomingEdges.length !== 2) return;
    
    const type = incomingEdges[0].literalType;
    const shape = incomingEdges[0].shape;
    const numberOfIterations = shape.reduce((product, value) => product * value, 1);
    
    let displacementInMemory: number;
    
    if (type !== undefined) {
        displacementInMemory = typeSizeMap[type];
    } else return;
    

    const loopIterationsNode = graph.addNode(formatId("Loop_iterations", nodeId)).init(new ConstantNode.Builder(numberOfIterations)).as(ConstantNode);
    const indexNode = graph.addNode(formatId("Index", nodeId), node).init(new VariableNode.Builder(6,"index", 'index')).as(VariableNode);
    const displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
    const input0Node = graph.addNode(formatId(incomingEdges[0].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[0].source.id}`, 'input')).as(VariableNode);
    const input1Node = graph.addNode(formatId(incomingEdges[1].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[1].source.id}`, 'input')).as(VariableNode);
    const outputNode = graph.addNode(formatId(node.type, nodeId), node).init(new VariableNode.Builder(type, '&Result', 'output')).as(VariableNode);
    const multiplicationNode = graph.addNode(formatId("Multiplication0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const load0Node = graph.addNode(formatId("Load0", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
    const load1Node = graph.addNode(formatId("Load1", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
    const opTypeNode = graph.addNode(formatId(opType, nodeId), node).init(new OperationNode.Builder(opType)).as(OperationNode);
    const additionNode = graph.addNode(formatId("Addition0", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const storeNode = graph.addNode(formatId("Store", nodeId), node).init(new OperationNode.Builder("Store")).as(OperationNode);
    const addToIndexNode = graph.addNode(formatId("addToIndexNode", nodeId), node).init(new ConstantNode.Builder(1)).as(ConstantNode);
    
    graph.addEdge(loopIterationsNode, node);

    graph.addEdge(indexNode, multiplicationNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(displacementInMemoryNode, multiplicationNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(input0Node, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(multiplicationNode, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(input1Node, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(multiplicationNode, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(load0Node, opTypeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(load1Node, opTypeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplicationNode, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(opTypeNode, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(storeNode, outputNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(indexNode, additionNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addToIndexNode, additionNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(additionNode, indexNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

}