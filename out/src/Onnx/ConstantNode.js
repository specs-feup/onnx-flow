import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
var ConstantNode;
(function (ConstantNode) {
    ConstantNode.TAG = "__specs-onnx__constant_node";
    ConstantNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get value() {
            return this.data[ConstantNode.TAG].value;
        }
    }
    ConstantNode.Class = Class;
    class Builder {
        value;
        constructor(value) {
            this.value = value;
        }
        buildData(data) {
            return {
                ...data,
                [ConstantNode.TAG]: {
                    version: ConstantNode.VERSION,
                    value: this.value
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    ConstantNode.Builder = Builder;
    ConstantNode.TypeGuard = Node.TagTypeGuard(ConstantNode.TAG, ConstantNode.VERSION);
})(ConstantNode || (ConstantNode = {}));
export default ConstantNode;
//# sourceMappingURL=ConstantNode.js.map