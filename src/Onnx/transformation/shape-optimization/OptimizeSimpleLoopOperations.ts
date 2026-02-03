import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js"
import ConstantNode from "../../ConstantNode.js";
import VariableNode from "../../VariableNode.js";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import OnnxInnerEdge from "../../OnnxInnerEdge.js";
import { formatId } from "../../Utils.js";


export default function optimizeSimpleLoopOperations(node: OperationNode.Class, graph: OnnxGraph.Class): void {

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

    const incomingEdges = node.incomers.filterIs(OnnxEdge);
    const loopIterationsNode = node.incomers.filterIs(BaseEdge).sources.filterIs(ConstantNode).first();

    if (incomingEdges.length !== 2 || loopIterationsNode === undefined) return;
    
    const nodeId : string = node.id
    const type = incomingEdges[0].literalType;
    const shape = incomingEdges[0].shape;

    if (type !== undefined && shape) {
        
        if (shape[0] === 1 && shape[1] === 1) {
            
            const nodeChildren = node.children
            nodeChildren.forEach(child => {
                child.remove();
            });

            loopIterationsNode.remove();
    
            const input0Node = graph.addNode(formatId(incomingEdges[0].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[0].source.id}`, 'input')).as(VariableNode);
            const input1Node = graph.addNode(formatId(incomingEdges[1].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[1].source.id}`, 'input')).as(VariableNode);
            const outputNode = graph.addNode(formatId(node.type, nodeId), node).init(new VariableNode.Builder(type, '&Result', 'output')).as(VariableNode);
            const load0Node = graph.addNode(formatId("Load0", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
            const load1Node = graph.addNode(formatId("Load1", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
            const opTypeNode = graph.addNode(formatId(opType, nodeId), node).init(new OperationNode.Builder(opType)).as(OperationNode);
            const zero = graph.addNode(formatId("zero_offset", nodeId), node).init(new ConstantNode.Builder(0)).as(ConstantNode);
            const storeNode = graph.addNode(formatId("Store", nodeId), node).init(new OperationNode.Builder("Store")).as(OperationNode);

            graph.addEdge(input0Node, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(zero, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

            graph.addEdge(input1Node, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(zero, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

            graph.addEdge(load0Node, opTypeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(load1Node, opTypeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

            graph.addEdge(zero, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(opTypeNode, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

            graph.addEdge(storeNode, outputNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

        }
    }



}