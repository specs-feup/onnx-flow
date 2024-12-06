import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import TensorNode from "../../TensorNode.js";
import OperationNode from "../../OperationNode.js";
import OnnxEdge from "../../OnnxEdge.js";
import OnnxGraph from "../../OnnxGraph.js"
import transformSimpleLoopOperations from "./TransformSimpleLoopOperations.js";
import transformMatMul from "./TransformMatMul.js";


export default class OnnxGraphTransformer
    implements Graph.Transformation<OnnxGraph.Class, OnnxGraph.Class>
{
    apply(graph: OnnxGraph.Class): OnnxGraph.Class {

        graph.nodes.filterIs(OperationNode).forEach(node => {
            if (node.type == "MatMul") {
                transformMatMul(node, graph)
            } else {
                transformSimpleLoopOperations(node, graph)
            }
        })

        // Return the modified graph
        return graph;
    }
}