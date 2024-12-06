import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import OnnxEdge from "./OnnxEdge.js";
var TensorNode;
(function (TensorNode) {
    TensorNode.TAG = "__specs-onnx__tensor_node";
    TensorNode.VERSION = "1";
    class Class extends BaseNode.Class {
        get literalType() {
            return this.data[TensorNode.TAG].literalType;
        }
        get shape() {
            return this.data[TensorNode.TAG].shape;
        }
        get type() {
            return this.data[TensorNode.TAG].type;
        }
        get geIncomers() {
            return this.incomers.filterIs(OnnxEdge);
        }
        get geOutgoers() {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }
    TensorNode.Class = Class;
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
                [TensorNode.TAG]: {
                    version: TensorNode.VERSION,
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
    TensorNode.Builder = Builder;
    TensorNode.TypeGuard = Node.TagTypeGuard(TensorNode.TAG, TensorNode.VERSION);
})(TensorNode || (TensorNode = {}));
export default TensorNode;
//# sourceMappingURL=TensorNode.js.map