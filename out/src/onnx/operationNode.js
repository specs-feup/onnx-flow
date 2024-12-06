import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import graphEdge from "./graphEdge.js";
var operationNode;
(function (operationNode) {
    operationNode.TAG = "__specs-onnx__operation_node";
    operationNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get type() {
            return this.data[operationNode.TAG].type;
        }
        get geIncomers() {
            return this.incomers.filterIs(graphEdge);
        }
        get geOutgoers() {
            return this.outgoers.filterIs(graphEdge);
        }
    }
    operationNode.Class = Class;
    class Builder {
        type;
        constructor(type) {
            this.type = type;
        }
        buildData(data) {
            return {
                ...data,
                [operationNode.TAG]: {
                    version: operationNode.VERSION,
                    type: this.type,
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    operationNode.Builder = Builder;
    operationNode.TypeGuard = Node.TagTypeGuard(operationNode.TAG, operationNode.VERSION);
})(operationNode || (operationNode = {}));
export default operationNode;
//# sourceMappingURL=operationNode.js.map