import { formatId } from "../Utilities.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import ConstantNode from "../../ConstantNode.js";
import VariableNode from "../../VariableNode.js";
import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import { typeSizeMap } from "../Utilities.js";
import OnnxInnerEdge from "../../OnnxInnerEdge.js";
export default function optimizeMatMul(node, graph) {
    let order = 0;
    const incomingEdges = node.incomers.filterIs(OnnxEdge);
    const outgoingEdges = node.outgoers.filterIs(OnnxEdge);
    const loopIterationsNode = node.incomers.filterIs(BaseEdge).sources.filterIs(ConstantNode).first();
    if (incomingEdges.length !== 2 || loopIterationsNode === undefined)
        return;
    const shape0 = incomingEdges[0].shape;
    const shape1 = incomingEdges[1].shape;
    const type = incomingEdges[0].literalType;
    let displacementInMemory;
    if (type !== undefined) {
        displacementInMemory = typeSizeMap[type];
    }
    else
        return;
    const nodeId = node.id;
    const instanceVal = [shape0[0], shape1[0], shape1[1]];
    let pattern = '';
    instanceVal.forEach(val => {
        pattern += val === 1 ? '1' : '0';
    });
    if (pattern === "000")
        return;
    const nodeChildren = node.children;
    nodeChildren.forEach(child => {
        child.remove();
    });
    if (pattern === "111") {
        loopIterationsNode.remove();
        const input0Node = graph.addNode(formatId(incomingEdges[0].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[0].source.id}`, 'input')).as(VariableNode);
        const input1Node = graph.addNode(formatId(incomingEdges[1].source.id, nodeId), node).init(new VariableNode.Builder(type, `&${incomingEdges[1].source.id}`, 'input')).as(VariableNode);
        const outputNode = graph.addNode(formatId(node.type, nodeId), node).init(new VariableNode.Builder(type, '&Result', 'output')).as(VariableNode);
        const load0Node = graph.addNode(formatId("Load0", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
        const load1Node = graph.addNode(formatId("Load1", nodeId), node).init(new OperationNode.Builder("Load")).as(OperationNode);
        const multiplicationNode = graph.addNode(formatId("Multiplication", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
        const zero = graph.addNode(formatId("zero_offset", nodeId), node).init(new ConstantNode.Builder(0)).as(ConstantNode);
        const storeNode = graph.addNode(formatId("Store", nodeId), node).init(new OperationNode.Builder("Store")).as(OperationNode);
        graph.addEdge(input0Node, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(zero, load0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(input1Node, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(zero, load1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(load0Node, multiplicationNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(load1Node, multiplicationNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(zero, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(multiplicationNode, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(storeNode, outputNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        /*
        graph.addEdge(input0Node, load0Node);
        graph.addEdge(zero, load0Node);

        graph.addEdge(input1Node, load1Node);
        graph.addEdge(zero, load1Node);

        graph.addEdge(load0Node, multiplicationNode);
        graph.addEdge(load1Node, multiplicationNode);
    
        graph.addEdge(zero, storeNode);
        graph.addEdge(multiplicationNode, storeNode);
    
        graph.addEdge(storeNode, outputNode);
        */
        return;
    }
    let jNode, kNode, columns1Node, rows1Node, displacementInMemoryNode, multiplication0Node, multiplication1Node, multiplication5Node, index0Node, index1Node, indexResNode, addition0Node, addition1Node, addition3Node;
    const addToIndexNode = graph.addNode(formatId("addToIndexNode", nodeId), node).init(new ConstantNode.Builder(1)).as(ConstantNode);
    switch (pattern) {
        case '100':
            jNode = graph.addNode(formatId("j", nodeId), node).init(new VariableNode.Builder(6, 'j', 'index_aux')).as(VariableNode);
            kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
            columns1Node = graph.addNode(formatId("#columns1", nodeId), node).init(new ConstantNode.Builder(shape1[1])).as(ConstantNode);
            rows1Node = graph.addNode(formatId("#rows1", nodeId), node).init(new ConstantNode.Builder(shape1[0])).as(ConstantNode);
            displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
            multiplication1Node = graph.addNode(formatId("Multiplication1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            addition1Node = graph.addNode(formatId("Addition1", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
            graph.addEdge(kNode, multiplication1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(columns1Node, multiplication1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(kNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(jNode, addition1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(multiplication1Node, addition1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(addition1Node, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(kNode, multiplication1Node);
            graph.addEdge(columns1Node, multiplication1Node);

            graph.addEdge(kNode, index0Node);
            graph.addEdge(displacementInMemoryNode, index0Node);

            graph.addEdge(jNode, addition1Node);
            graph.addEdge(multiplication1Node, addition1Node);

            graph.addEdge(addition1Node, index1Node);
            graph.addEdge(displacementInMemoryNode, index1Node);
            */
            indexResNode = graph.addNode(formatId("IndexRes", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            graph.addEdge(jNode, indexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, indexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(jNode, indexResNode);
            graph.addEdge(displacementInMemoryNode, indexResNode);
            */
            break;
        case '001':
            jNode = graph.addNode(formatId("j", nodeId), node).init(new VariableNode.Builder(6, 'j', 'index_aux')).as(VariableNode);
            kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
            rows1Node = graph.addNode(formatId("#rows1", nodeId), node).init(new ConstantNode.Builder(shape1[0])).as(ConstantNode);
            displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
            multiplication0Node = graph.addNode(formatId("Multiplication0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            addition0Node = graph.addNode(formatId("Addition0", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
            graph.addEdge(jNode, multiplication0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(rows1Node, multiplication0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(multiplication0Node, addition0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(kNode, addition0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(kNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(addition0Node, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(jNode, multiplication0Node);
            graph.addEdge(rows1Node, multiplication0Node);

            graph.addEdge(multiplication0Node, addition0Node);
            graph.addEdge(kNode, addition0Node);

            graph.addEdge(kNode, index1Node);
            graph.addEdge(displacementInMemoryNode, index1Node);

            graph.addEdge(addition0Node, index0Node);
            graph.addEdge(displacementInMemoryNode, index0Node);
            */
            indexResNode = graph.addNode(formatId("IndexRes", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            graph.addEdge(jNode, indexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, indexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(jNode, indexResNode);
            graph.addEdge(displacementInMemoryNode, indexResNode);
            */
            break;
        case '101':
            kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
            displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
            index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            indexResNode = graph.addNode(formatId("IndexResNode", nodeId), node).init(new ConstantNode.Builder(0)).as(ConstantNode);
            graph.addEdge(kNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(kNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(kNode, index0Node);
            graph.addEdge(displacementInMemoryNode, index0Node);
            graph.addEdge(kNode, index1Node);
            graph.addEdge(displacementInMemoryNode, index1Node);
            */
            break;
        case '010':
            jNode = graph.addNode(formatId("j", nodeId), node).init(new VariableNode.Builder(6, 'j', 'index_aux')).as(VariableNode);
            kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
            columns1Node = graph.addNode(formatId("#columns1", nodeId), node).init(new ConstantNode.Builder(shape1[1])).as(ConstantNode);
            displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
            index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            graph.addEdge(jNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(kNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(jNode, index0Node);
            graph.addEdge(displacementInMemoryNode, index0Node);

            graph.addEdge(kNode, index1Node);
            graph.addEdge(displacementInMemoryNode, index1Node);
            */
            multiplication5Node = graph.addNode(formatId("Multiplication5", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            indexResNode = graph.addNode(formatId("IndexRes", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            addition3Node = graph.addNode(formatId("Addition3", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
            graph.addEdge(jNode, multiplication5Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(columns1Node, multiplication5Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(kNode, addition3Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(multiplication5Node, addition3Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(addition3Node, indexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, indexResNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(jNode, multiplication5Node);
            graph.addEdge(columns1Node, multiplication5Node);

            graph.addEdge(kNode, addition3Node);
            graph.addEdge(multiplication5Node, addition3Node);

            graph.addEdge(addition3Node, indexResNode);
            graph.addEdge(displacementInMemoryNode, indexResNode);
            */
            break;
        case '110':
            kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
            displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
            index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new ConstantNode.Builder(0)).as(ConstantNode);
            indexResNode = index1Node;
            graph.addEdge(kNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index1Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(kNode, index1Node);
            graph.addEdge(displacementInMemoryNode, index1Node);
            */
            break;
        case '011':
            kNode = graph.addNode(formatId("k", nodeId), node).init(new VariableNode.Builder(6, 'k', 'index_aux')).as(VariableNode);
            displacementInMemoryNode = graph.addNode(formatId("displacementInMemory", nodeId), node).init(new ConstantNode.Builder(displacementInMemory)).as(ConstantNode);
            index0Node = graph.addNode(formatId("Index0", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            index1Node = graph.addNode(formatId("Index1", nodeId), node).init(new ConstantNode.Builder(0)).as(ConstantNode);
            indexResNode = index0Node;
            graph.addEdge(kNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            graph.addEdge(displacementInMemoryNode, index0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            /*
            graph.addEdge(kNode, index0Node);
            graph.addEdge(displacementInMemoryNode, index0Node);
            */
            break;
    }
    if (index0Node && index1Node && indexResNode && kNode) {
        const indexNode = graph.addNode(formatId("Index", nodeId), node).init(new VariableNode.Builder(6, "index", 'index')).as(VariableNode);
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
        graph.addEdge(indexResNode, load2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(load2Node, addition2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(multiplication4Node, addition2Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
        graph.addEdge(indexResNode, storeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
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
        graph.addEdge(indexResNode, load2Node);

        graph.addEdge(load2Node, addition2Node);
        graph.addEdge(multiplication4Node, addition2Node);

        graph.addEdge(addition2Node, storeNode);
        graph.addEdge(indexResNode, storeNode);

        graph.addEdge(storeNode, outputNode);

        graph.addEdge(indexNode, addition8Node);
        graph.addEdge(addToIndexNode, addition8Node);
        graph.addEdge(addition8Node, indexNode);

        graph.addEdge(kNode, addition9Node);
        graph.addEdge(addToIndexNode, addition9Node);
        graph.addEdge(addition9Node, kNode);
        */
        if (jNode) {
            const addition4Node = graph.addNode(formatId("Addition4", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
            const addition5Node = graph.addNode(formatId("Addition5", nodeId), node).init(new OperationNode.Builder("Addition")).as(OperationNode);
            const multiplication7Node = graph.addNode(formatId("Multiplication7", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            const multiplication8Node = graph.addNode(formatId("Multiplication8", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            const multiplication9Node = graph.addNode(formatId("Multiplication9", nodeId), node).init(new OperationNode.Builder("Multiplication")).as(OperationNode);
            const equality0Node = graph.addNode(formatId("Equality0", nodeId), node).init(new OperationNode.Builder("Equality")).as(OperationNode);
            const not0Node = graph.addNode(formatId("Not0", nodeId), node).init(new OperationNode.Builder("Not")).as(OperationNode);
            graph.addEdge(kNode, equality0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            //graph.addEdge(kNode, equality0Node); 
            if (pattern === '010' && columns1Node)
                graph.addEdge(columns1Node, equality0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            //graph.addEdge(columns1Node, equality0Node);
            else if (rows1Node)
                graph.addEdge(rows1Node, equality0Node).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
            //graph.addEdge(rows1Node, equality0Node);
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
            /*
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
            */
        }
    }
}
//# sourceMappingURL=OptimizeMatMul.js.map