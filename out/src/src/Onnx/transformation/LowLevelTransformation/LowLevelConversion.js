import OperationNode from "../../OperationNode.js";
import transformSimpleLoopOperations from "./TransformSimpleLoopOperations.js";
import transformMatMul from "./TransformMatMul.js";
export default class OnnxGraphTransformer {
    apply(graph) {
        graph.nodes.filterIs(OperationNode).forEach(node => {
            if (node.type == "MatMul") {
                transformMatMul(node, graph);
            }
            else {
                transformSimpleLoopOperations(node, graph);
            }
        });
        // Return the modified graph
        return graph;
    }
}
//# sourceMappingURL=LowLevelConversion.js.map