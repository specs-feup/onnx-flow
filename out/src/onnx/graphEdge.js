import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
var graphEdge;
(function (graphEdge) {
    graphEdge.TAG = "__specs-onnx__graph_edge";
    graphEdge.VERSION = "1";
    class Class extends BaseEdge.Class {
        get literalType() {
            return this.data[graphEdge.TAG].literalType;
        }
        set literalType(value) {
            this.data[graphEdge.TAG].literalType = value;
        }
        get shape() {
            return this.data[graphEdge.TAG].shape;
        }
        set shape(value) {
            this.data[graphEdge.TAG].shape = value;
        }
    }
    graphEdge.Class = Class;
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
                [graphEdge.TAG]: {
                    version: graphEdge.VERSION,
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
    graphEdge.Builder = Builder;
    graphEdge.TypeGuard = Edge.TagTypeGuard(graphEdge.TAG, graphEdge.VERSION);
})(graphEdge || (graphEdge = {}));
export default graphEdge;
//# sourceMappingURL=graphEdge.js.map