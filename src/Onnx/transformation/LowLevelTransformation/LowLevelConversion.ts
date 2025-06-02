import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js"
import transformSimpleLoopOperations from "./TransformSimpleLoopOperations.js";
import transformMatMul from "./TransformMatMul.js";
import transformLoop from "./TransformLoop.js";
import transformChain from "./TransformChain.js";


export default class OnnxGraphTransformer
    implements Graph.Transformation<OnnxGraph.Class, OnnxGraph.Class>
{
    apply(graph: OnnxGraph.Class): OnnxGraph.Class {
        const operationCount = graph.nodes.filterIs(OperationNode).length;

        if (operationCount === 3) {
            const addNodes = graph.nodes.filterIs(OperationNode).filter(node => node.type === "Add").toArray();
            transformChain(addNodes, graph);
        } else {
            graph.nodes.filterIs(OperationNode).forEach(node => {
                if (node.type == "MatMul") {
                    transformMatMul(node, graph)
                } else if (node.type == "Loop") {
                    transformLoop(node, graph)
                } else {
                    transformSimpleLoopOperations(node, graph)
                }
            })
        }

        // Return the modified graph
        return graph;
    }
}