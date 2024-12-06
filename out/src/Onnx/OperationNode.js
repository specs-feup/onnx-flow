import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import OnnxEdge from "./OnnxEdge.js";
var OperationNode;
(function (OperationNode) {
    OperationNode.TAG = "__specs-onnx__operation_node";
    OperationNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get type() {
            return this.data[OperationNode.TAG].type;
        }
        set type(newType) {
            this.data[OperationNode.TAG].type = newType;
        }
        get geIncomers() {
            return this.incomers.filterIs(OnnxEdge);
        }
        get geOutgoers() {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }
    OperationNode.Class = Class;
    class Builder {
        type;
        constructor(type) {
            this.type = type;
        }
        buildData(data) {
            return {
                ...data,
                [OperationNode.TAG]: {
                    version: OperationNode.VERSION,
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
    OperationNode.Builder = Builder;
    OperationNode.TypeGuard = Node.TagTypeGuard(OperationNode.TAG, OperationNode.VERSION);
})(OperationNode || (OperationNode = {}));
export default OperationNode;
//# sourceMappingURL=OperationNode.js.map