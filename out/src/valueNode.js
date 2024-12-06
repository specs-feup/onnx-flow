import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
var valueNode;
(function (valueNode) {
    valueNode.TAG = "__specs-onnx__value_node";
    valueNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get literalType() {
            return this.data[valueNode.TAG].literalType;
        }
        get shape() {
            return this.data[valueNode.TAG].shape;
        }
        get type() {
            return this.data[valueNode.TAG].type;
        }
    }
    valueNode.Class = Class;
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
                [valueNode.TAG]: {
                    version: valueNode.VERSION,
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
    valueNode.Builder = Builder;
    valueNode.TypeGuard = Node.TagTypeGuard(valueNode.TAG, valueNode.VERSION);
})(valueNode || (valueNode = {}));
export default valueNode;
//# sourceMappingURL=valueNode.js.map