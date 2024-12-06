import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
var OnnxEdge;
(function (OnnxEdge) {
    OnnxEdge.TAG = "__specs-onnx__graph_edge";
    OnnxEdge.VERSION = "1";
    class Class extends BaseEdge.Class {
        get literalType() {
            return this.data[OnnxEdge.TAG].literalType;
        }
        set literalType(value) {
            this.data[OnnxEdge.TAG].literalType = value;
        }
        get shape() {
            return this.data[OnnxEdge.TAG].shape;
        }
        set shape(value) {
            this.data[OnnxEdge.TAG].shape = value;
        }
    }
    OnnxEdge.Class = Class;
    class Builder {
        literalType;
        shape;
        constructor(literalType, shape = []) {
            this.literalType = literalType;
            this.shape = shape;
        }
        buildData(data) {
            return {
                ...data,
                [OnnxEdge.TAG]: {
                    version: OnnxEdge.VERSION,
                    literalType: this.literalType,
                    shape: this.shape,
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    OnnxEdge.Builder = Builder;
    OnnxEdge.TypeGuard = Edge.TagTypeGuard(OnnxEdge.TAG, OnnxEdge.VERSION);
})(OnnxEdge || (OnnxEdge = {}));
export default OnnxEdge;
//# sourceMappingURL=OnnxEdge.js.map