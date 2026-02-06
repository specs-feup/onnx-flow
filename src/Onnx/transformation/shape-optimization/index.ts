import Graph from "@specs-feup/flow/graph/Graph";
import OperationNode from "../../OperationNode.js";
import OnnxGraph from "../../OnnxGraph.js";
import optimizeSimpleLoopOperations from "./OptimizeSimpleLoopOperations.js";
import optimizeMatMul from "./OptimizeMatMul.js";

export default class OnnxGraphOptimizer implements Graph.Transformation<
    OnnxGraph.Class,
    OnnxGraph.Class
> {
    apply(graph: OnnxGraph.Class): OnnxGraph.Class {
        graph.nodes
            .filterIs(OperationNode)
            .filter((node) => node.isParent)
            .forEach((node) => {
                if (node.type == "MatMul") {
                    optimizeMatMul(node, graph);
                } else {
                    optimizeSimpleLoopOperations(node, graph);
                }
            });

        // Return the modified graph
        return graph;
    }
}
