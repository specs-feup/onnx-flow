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
    
    if (type === undefined) return;
    
    // Creating nodes for the loop visualization
    
    // Trip count (number of iterations)
    const tripCountNode = graph.addNode(formatId("trip_count", nodeId))
        .init(new ConstantNode.Builder(numberOfIterations)).as(ConstantNode);
    
    // Loop condition (always true)
    const loopCondNode = graph.addNode(formatId("loop_cond", nodeId))
        .init(new ConstantNode.Builder(1)).as(ConstantNode);
    
    // Loop node
    const loopNode = graph.addNode(formatId("Loop", nodeId))
        .init(new OperationNode.Builder("Loop")).as(OperationNode);
        
    // Connect trip count and condition to loop node
    graph.addEdge(tripCountNode, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(loopCondNode, loopNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // Constants outside the loop
    const columns1Node = graph.addNode(formatId("#columns1", nodeId))
        .init(new ConstantNode.Builder(shape1[1])).as(ConstantNode);
    
    // Loop condition handling
    const condIn = graph.addNode(formatId("cond_in", nodeId), loopNode)
        .init(new VariableNode.Builder(typeSizeMap["bool"], "cond_in", "input")).as(VariableNode);
    const identityCondOp = graph.addNode(formatId("identity_cond", nodeId), loopNode)
        .init(new OperationNode.Builder("Identity")).as(OperationNode);
    const condOut = graph.addNode(formatId("cond_out", nodeId), loopNode)
        .init(new VariableNode.Builder(typeSizeMap["bool"], "cond_out", "output")).as(VariableNode);
        
    graph.addEdge(condIn, identityCondOp).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(identityCondOp, condOut).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // Loop iteration index
    const iterIdx = graph.addNode(formatId("iter", nodeId), loopNode)
        .init(new VariableNode.Builder(typeSizeMap["int64"], "iter", "index")).as(VariableNode);
    
    // Carry handling
    const carryIn = graph.addNode(formatId("carry_in", nodeId), loopNode)
        .init(new VariableNode.Builder(type, "carry_in", "input")).as(VariableNode);
    const carryOut = graph.addNode(formatId("carry_out", nodeId), loopNode)
        .init(new VariableNode.Builder(type, "carry_out", "output")).as(VariableNode);
    
    // Connect inputs to gather operations from outside the loop
    // We use direct references to the original inputs
    
    // 1. Mod and Div operations on the iterator
    const modNode = graph.addNode(formatId("Mod", nodeId), loopNode)
        .init(new OperationNode.Builder("Mod")).as(OperationNode);
    const divNode = graph.addNode(formatId("Div", nodeId), loopNode)
        .init(new OperationNode.Builder("Div")).as(OperationNode);
    const unsqueezeNode = graph.addNode(formatId("Unsqueeze", nodeId), loopNode)
        .init(new OperationNode.Builder("Unsqueeze")).as(OperationNode);
    
    // Connect iterator to Mod and Div
    graph.addEdge(iterIdx, modNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(columns1Node, modNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    graph.addEdge(iterIdx, divNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(columns1Node, divNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);

    graph.addEdge(iterIdx, unsqueezeNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 2. Unsqueeze operations
    const unsqueezeModNode = graph.addNode(formatId("UnsqueezeMod", nodeId), loopNode)
        .init(new OperationNode.Builder("Unsqueeze")).as(OperationNode);
    const unsqueezeDivNode = graph.addNode(formatId("UnsqueezeDiv", nodeId), loopNode)
        .init(new OperationNode.Builder("Unsqueeze")).as(OperationNode);
    
    graph.addEdge(modNode, unsqueezeModNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(divNode, unsqueezeDivNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 3. Gather operations - directly connect to input sources from outside the loop
    const gatherANode = graph.addNode(formatId("GatherA", nodeId), loopNode)
        .init(new OperationNode.Builder("Gather")).as(OperationNode);
    const gatherBNode = graph.addNode(formatId("GatherB", nodeId), loopNode)
        .init(new OperationNode.Builder("Gather")).as(OperationNode);
    
    graph.addEdge(incomingEdges[0].source, gatherANode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(unsqueezeDivNode, gatherANode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    graph.addEdge(incomingEdges[1].source, gatherBNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(unsqueezeModNode, gatherBNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 4. Reshape operations
    const reshapeANode = graph.addNode(formatId("ReshapeA", nodeId), loopNode)
        .init(new OperationNode.Builder("Reshape")).as(OperationNode);
    const reshapeBNode = graph.addNode(formatId("ReshapeB", nodeId), loopNode)
        .init(new OperationNode.Builder("Reshape")).as(OperationNode);
    
    graph.addEdge(gatherANode, reshapeANode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(gatherBNode, reshapeBNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 5. Mul operation to combine paths
    const mulNode = graph.addNode(formatId("Mul", nodeId), loopNode)
        .init(new OperationNode.Builder("Mul")).as(OperationNode);
    
    graph.addEdge(reshapeANode, mulNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(reshapeBNode, mulNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 6. ReduceSum operation
    const reduceSumNode = graph.addNode(formatId("ReduceSum", nodeId), loopNode)
        .init(new OperationNode.Builder("ReduceSum")).as(OperationNode);
    
    graph.addEdge(mulNode, reduceSumNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 7. Reshape after ReduceSum
    const reshapeOutputNode = graph.addNode(formatId("ReshapeOutput", nodeId), loopNode)
        .init(new OperationNode.Builder("Reshape")).as(OperationNode);
    
    graph.addEdge(reduceSumNode, reshapeOutputNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // 8. ScatterElements operation using the unsqueeze
    const scatterNode = graph.addNode(formatId("ScatterElements", nodeId), loopNode)
        .init(new OperationNode.Builder("ScatterElements")).as(OperationNode);
    
    // Connect carry and reshape output to ScatterElements
    graph.addEdge(unsqueezeNode, scatterNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(reshapeOutputNode, scatterNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(carryIn, scatterNode).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    graph.addEdge(scatterNode, carryOut).init(new OnnxInnerEdge.Builder(order++)).as(OnnxInnerEdge);
    
    // Connect the output of the loop (final carry value) to all targets of the MatMul node
    node.outgoers.forEach(edge => {
        const target = edge.target;
        graph.addEdge(loopNode, target).init(
            new OnnxEdge.Builder(type, shape0) // Assuming the shape of the output is the same as input0
        ).as(OnnxEdge);
        
        edge.remove(); // Remove the original edge
    });
    
    // Remove the original MatMul node as it's now represented by the loop
    if (graph.hasNode(node.id)) {
        node.remove();
    }
}