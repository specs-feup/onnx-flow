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


export default function transformMatMul(node: OperationNode.Class, graph: OnnxGraph.Class): void {

    let order = 0;


    const nodeId : string = node.id

    const incomingEdges = node.incomers.filterIs(OnnxEdge);

    if (incomingEdges.length !== 2) return;
    
    const type = incomingEdges[0].literalType;
    const shape0 = incomingEdges[0].shape;
    const shape1 = incomingEdges[1].shape;
    const numberOfIterations = shape0[0] * shape1[0] * shape1[1];

    let displacementInMemory: number;
    
    if (type !== undefined) {
        displacementInMemory = typeSizeMap[type];
    } else return;
    
    const loopIterationsNode = graph.addNode(formatId("Loop_iterations", nodeId)).init(new ConstantNode.Builder(numberOfIterations)).as(ConstantNode);
    graph.addEdge(loopIterationsNode, node);

    const iNode = graph.addNode(formatId("i", nodeId), node).init(new VariableNode.Builder(6, 'i', 'index_aux')).as(VariableNode);
    const jNode = graph.addNode(formatId("j", nodeId), node).init(new VariableNode.Builder(6, 'j', 'index_aux')).as(VariableNode);
    const kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
    const columns1Node = graph.addNode(formatId("#columns1", nodeId), node).init(new ConstantNode.Builder(shape1[1])).as(ConstantNode);
    const rows1Node = graph.addNode(formatId("#rows1", nodeId), node).init(new ConstantNode.Builder(shape1[0])).as(ConstantNode);
    const displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
    const multiplication0Node = graph.addNode(formatId("Multiplication0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const multiplication1Node = graph.addNode(formatId("Multiplication1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const addition0Node = graph.addNode(formatId("Addition0", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const addition1Node = graph.addNode(formatId("Addition1", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);

    graph.addEdge(iNode, multiplication0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(rows1Node, multiplication0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(kNode, multiplication1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(columns1Node, multiplication1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplication0Node, addition0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(kNode, addition0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplication1Node, addition1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(jNode, addition1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(addition0Node, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(displacementInMemoryNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(addition1Node, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(displacementInMemoryNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    /*
    graph.addEdge(iNode, multiplication0Node);
    graph.addEdge(rows1Node, multiplication0Node);

    graph.addEdge(kNode, multiplication1Node);
    graph.addEdge(columns1Node, multiplication1Node);

    graph.addEdge(multiplication0Node, addition0Node);
    graph.addEdge(kNode, addition0Node);

    graph.addEdge(multiplication1Node, addition1Node);
    graph.addEdge(jNode, addition1Node);

    graph.addEdge(addition0Node, index0Node);
    graph.addEdge(displacementInMemoryNode, index0Node);

    graph.addEdge(addition1Node, index1Node);
    graph.addEdge(displacementInMemoryNode, index1Node);
    */
    const multiplication5Node = graph.addNode(formatId("Multiplication5", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const IndexResNode = graph.addNode(formatId("IndexRes", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const Addition3Node = graph.addNode(formatId("Addition3", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);

    graph.addEdge(iNode, multiplication5Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(columns1Node, multiplication5Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(jNode, Addition3Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(multiplication5Node, Addition3Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(Addition3Node, IndexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(displacementInMemoryNode, IndexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    /*
    graph.addEdge(iNode, multiplication5Node);
    graph.addEdge(columns1Node, multiplication5Node);

    graph.addEdge(jNode, Addition3Node);
    graph.addEdge(multiplication5Node, Addition3Node);

    graph.addEdge(Addition3Node, IndexResNode);
    graph.addEdge(displacementInMemoryNode, IndexResNode);
    */

    const addToIndexNode = graph.addNode(formatId("addToIndexNode", nodeId), node).init(new ConstantNode.Builder(1)).as(ConstantNode);
    const indexNode = graph.addNode(formatId("Index", nodeId), node).init(new VariableNode.Builder(6,"index", 'index')).as(VariableNode);
    const input0Node = graph.addNode(formatId(incomingEdges[0].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[0].source.id}`, 'input')).as(VariableNode);
    const input1Node = graph.addNode(formatId(incomingEdges[1].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[1].source.id}`, 'input')).as(VariableNode);
    const outputNode = graph.addNode(formatId(node.type, nodeId), node).init(new VariableNode.Builder(type, '&Result', 'output')).as(VariableNode);
    const multiplication4Node = graph.addNode(formatId("Multiplication4", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const addition2Node = graph.addNode(formatId("Addition2", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const addition8Node = graph.addNode(formatId("Addition8", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const addition9Node = graph.addNode(formatId("Addition9", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const load0Node = graph.addNode(formatId("Load0", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
    const load1Node = graph.addNode(formatId("Load1", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
    const load2Node = graph.addNode(formatId("Load2", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
    const storeNode = graph.addNode(formatId("Store", nodeId), node).init(new OperationNode.Builder("Store")).as(OperationNode);

    graph.addEdge(input0Node, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(index0Node, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(input1Node, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(index1Node, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(load0Node, multiplication4Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(load1Node, multiplication4Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(outputNode, load2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(IndexResNode, load2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(load2Node, addition2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(multiplication4Node, addition2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(IndexResNode, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addition2Node, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(storeNode, outputNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(indexNode, addition8Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addToIndexNode, addition8Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addition8Node, indexNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(kNode, addition9Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addToIndexNode, addition9Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addition9Node, kNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    /*
    graph.addEdge(input0Node, load0Node);
    graph.addEdge(index0Node, load0Node);

    graph.addEdge(input1Node, load1Node);
    graph.addEdge(index1Node, load1Node);

    graph.addEdge(load0Node, multiplication4Node);
    graph.addEdge(load1Node, multiplication4Node);

    graph.addEdge(outputNode, load2Node);
    graph.addEdge(IndexResNode, load2Node);

    graph.addEdge(load2Node, addition2Node);
    graph.addEdge(multiplication4Node, addition2Node);

    graph.addEdge(IndexResNode, storeNode);
    graph.addEdge(addition2Node, storeNode);
    
    graph.addEdge(storeNode, outputNode);

    graph.addEdge(indexNode, addition8Node);
    graph.addEdge(addToIndexNode, addition8Node);
    graph.addEdge(addition8Node, indexNode);

    graph.addEdge(kNode, addition9Node);
    graph.addEdge(addToIndexNode, addition9Node);
    graph.addEdge(addition9Node, kNode);
    */

    const addition4Node = graph.addNode(formatId("Addition4", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const addition5Node = graph.addNode(formatId("Addition5", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const addition6Node = graph.addNode(formatId("Addition6", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const addition7Node = graph.addNode(formatId("Addition7", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
    const multiplication7Node = graph.addNode(formatId("Multiplication7", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const multiplication8Node = graph.addNode(formatId("Multiplication8", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const multiplication9Node = graph.addNode(formatId("Multiplication9", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const multiplication10Node = graph.addNode(formatId("Multiplication10", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const multiplication11Node = graph.addNode(formatId("Multiplication11", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const multiplication12Node = graph.addNode(formatId("Multiplication12", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
    const equality0Node = graph.addNode(formatId("Equality0", nodeId), node).init(new OperationNode.Builder("Equality")).as(OperationNode);
    const equality1Node = graph.addNode(formatId("Equality1", nodeId), node).init(new OperationNode.Builder("Equality")).as(OperationNode);
    const not0Node = graph.addNode(formatId("Not0", nodeId), node).init(new OperationNode.Builder("Not")).as(OperationNode);
    const not1Node = graph.addNode(formatId("Not1", nodeId), node).init(new OperationNode.Builder("Not")).as(OperationNode);


    graph.addEdge(kNode, equality0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(rows1Node, equality0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(addToIndexNode, addition4Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(jNode, addition4Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(equality0Node, multiplication7Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addition4Node, multiplication7Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(equality0Node, not0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(not0Node, multiplication8Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(jNode, multiplication8Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplication7Node, addition5Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(multiplication8Node, addition5Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(addition5Node, jNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(not0Node, multiplication9Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(kNode, multiplication9Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplication9Node, kNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(jNode, equality1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(columns1Node, equality1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(iNode, addition6Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addToIndexNode, addition6Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(equality1Node, multiplication10Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(addition6Node, multiplication10Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(equality1Node, not1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(not1Node, multiplication11Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(iNode, multiplication11Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplication10Node, addition7Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(multiplication11Node, addition7Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(addition7Node, iNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(not1Node, multiplication12Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(jNode, multiplication12Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(multiplication12Node, jNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    /*
    graph.addEdge(kNode, equality0Node);
    graph.addEdge(rows1Node, equality0Node);

    graph.addEdge(addToIndexNode, addition4Node);
    graph.addEdge(jNode, addition4Node);

    graph.addEdge(equality0Node, multiplication7Node);
    graph.addEdge(addition4Node, multiplication7Node);

    graph.addEdge(equality0Node, not0Node);

    graph.addEdge(not0Node, multiplication8Node);
    graph.addEdge(jNode, multiplication8Node);

    graph.addEdge(multiplication7Node, addition5Node);
    graph.addEdge(multiplication8Node, addition5Node);

    graph.addEdge(addition5Node, jNode);

    graph.addEdge(not0Node, multiplication9Node);
    graph.addEdge(kNode, multiplication9Node);

    graph.addEdge(multiplication9Node, kNode);

    graph.addEdge(jNode, equality1Node);
    graph.addEdge(columns1Node, equality1Node);

    graph.addEdge(iNode, addition6Node);
    graph.addEdge(addToIndexNode, addition6Node);

    graph.addEdge(equality1Node, multiplication10Node);
    graph.addEdge(addition6Node, multiplication10Node);

    graph.addEdge(equality1Node, not1Node);

    graph.addEdge(not1Node, multiplication11Node);
    graph.addEdge(iNode, multiplication11Node);

    graph.addEdge(multiplication10Node, addition7Node);
    graph.addEdge(multiplication11Node, addition7Node);

    graph.addEdge(addition7Node, iNode);

    graph.addEdge(not1Node, multiplication12Node);
    graph.addEdge(jNode, multiplication12Node);

    graph.addEdge(multiplication12Node, jNode);
    */

}