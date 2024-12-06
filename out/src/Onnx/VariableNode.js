import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
var VariableNode;
(function (VariableNode) {
    VariableNode.TAG = "__specs-onnx__variable_node";
    VariableNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get literalType() {
            return this.data[VariableNode.TAG].literalType;
        }
        get name() {
            return this.data[VariableNode.TAG].name;
        }
        get type() {
            return this.data[VariableNode.TAG].type;
        }
    }
    VariableNode.Class = Class;
    class Builder {
        literalType;
        name;
        type;
        constructor(literalType, name, type) {
            this.literalType = literalType;
            this.name = name;
            this.type = type;
        }
        buildData(data) {
            return {
                ...data,
                [VariableNode.TAG]: {
                    version: VariableNode.VERSION,
                    literalType: this.literalType,
                    name: this.name,
                    type: this.type
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    VariableNode.Builder = Builder;
    VariableNode.TypeGuard = Node.TagTypeGuard(VariableNode.TAG, VariableNode.VERSION);
})(VariableNode || (VariableNode = {}));
export default VariableNode;
//# sourceMappingURL=VariableNode.js.map