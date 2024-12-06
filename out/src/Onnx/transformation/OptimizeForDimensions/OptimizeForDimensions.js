import OperationNode from "../../OperationNode.js";
import optimizeSimpleLoopOperations from "./OptimizeSimpleLoopOperations.js";
import optimizeMatMul from "./OptimizeMatMul.js";
export default class OnnxGraphOptimizer {
    apply(graph) {
        graph.nodes.filterIs(OperationNode).filter(node => node.isParent).forEach(node => {
            if (node.type == "MatMul") {
                optimizeMatMul(node, graph);
            }
            else {
                optimizeSimpleLoopOperations(node, graph);
            }
        });
        // Return the modified graph
        return graph;
    }
}
//# sourceMappingURL=OptimizeForDimensions.js.map