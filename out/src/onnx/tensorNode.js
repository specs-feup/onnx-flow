import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import graphEdge from "./graphEdge.js";
var tensorNode;
(function (tensorNode) {
    tensorNode.TAG = "__specs-onnx__tensor_node";
    tensorNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get literalType() {
            return this.data[tensorNode.TAG].literalType;
        }
        get shape() {
            return this.data[tensorNode.TAG].shape;
        }
        get type() {
            return this.data[tensorNode.TAG].type;
        }
        get geIncomers() {
            return this.incomers.filterIs(graphEdge);
        }
        get geOutgoers() {
            return this.outgoers.filterIs(graphEdge);
        }
    }
    tensorNode.Class = Class;
    class Builder {
        literalType;
        shape;
        type;
        constructor(literalType, shape, type) {
            this.literalType = literalType;
            this.shape = shape;
            this.type = type;
        }
        buildData(data) {
            return {
                ...data,
                [tensorNode.TAG]: {
                    version: tensorNode.VERSION,
                    literalType: this.literalType,
                    shape: this.shape,
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
    tensorNode.Builder = Builder;
    tensorNode.TypeGuard = Node.TagTypeGuard(tensorNode.TAG, tensorNode.VERSION);
})(tensorNode || (tensorNode = {}));
export default tensorNode;
//# sourceMappingURL=tensorNode.js.map