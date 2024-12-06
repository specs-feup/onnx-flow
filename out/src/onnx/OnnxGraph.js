import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import TensorNode from "./TensorNode.js";
import OperationNode from "./OperationNode.js";
var OnnxGraph;
(function (OnnxGraph) {
    OnnxGraph.TAG = "__specs-onnx__onnx_graph";
    OnnxGraph.VERSION = "1";
    class Class extends BaseGraph.Class {
        // Retrieve all TensorNodes with type 'input'
        getInputTensorNodes() {
            return this.nodes.filterIs(TensorNode).filter(n => n.type === "input");
        }
        // Retrieve all TensorNodes with type 'output'
        getOutputTensorNodes() {
            return this.nodes.filterIs(TensorNode).filter(n => n.type === "output");
        }
        // Retrieve all OperationNodes
        getOperationNodes() {
            return this.nodes.filterIs(OperationNode);
        }
    }
    OnnxGraph.Class = Class;
    class Builder {
        buildData(data) {
            return {
                ...data,
                [OnnxGraph.TAG]: {
                    version: OnnxGraph.VERSION
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    OnnxGraph.Builder = Builder;
    OnnxGraph.TypeGuard = Graph.TagTypeGuard(OnnxGraph.TAG, OnnxGraph.VERSION);
})(OnnxGraph || (OnnxGraph = {}));
export default OnnxGraph;
//# sourceMappingURL=OnnxGraph.js.map