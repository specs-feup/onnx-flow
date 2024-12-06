import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import TensorNode from "./TensorNode.js";
import OperationNode from "./OperationNode.js";
//preciso de nodes constant (têm um value associado), operationNode (igual ao outro, mas precisam de um parent),
//compound Node (pode ser um operationNode, só que é parent)
//edges dentro do parent vão precisar de quê
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