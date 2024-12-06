import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import tensorNode from "./tensorNode.js";
import operationNode from "./operationNode.js";
var onnxGraph;
(function (onnxGraph) {
    onnxGraph.TAG = "__specs-onnx__onnx_graph";
    onnxGraph.VERSION = "1";
    class Class extends BaseGraph.Class {
        // Retrieve all tensorNodes with type 'input'
        getInputTensorNodes() {
            return this.nodes.filterIs(tensorNode).filter(n => n.type === "input");
        }
        // Retrieve all tensorNodes with type 'output'
        getOutputTensorNodes() {
            return this.nodes.filterIs(tensorNode).filter(n => n.type === "output");
        }
        // Retrieve all operationNodes
        getOperationNodes() {
            return this.nodes.filterIs(operationNode);
        }
    }
    onnxGraph.Class = Class;
    class Builder {
        buildData(data) {
            return {
                ...data,
                [onnxGraph.TAG]: {
                    version: onnxGraph.VERSION
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    onnxGraph.Builder = Builder;
    onnxGraph.TypeGuard = Graph.TagTypeGuard(onnxGraph.TAG, onnxGraph.VERSION);
})(onnxGraph || (onnxGraph = {}));
export default onnxGraph;
//# sourceMappingURL=onnxGraph.js.map