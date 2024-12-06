import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
var GraphEdge;
(function (GraphEdge) {
    GraphEdge.TAG = "__specs-onnx__graph_edge";
    GraphEdge.VERSION = "1";
    class Class extends BaseEdge.Class {
        get literalType() {
            return this.data[GraphEdge.TAG].literalType;
        }
        set literalType(value) {
            this.data[GraphEdge.TAG].literalType = value;
        }
        get shape() {
            return this.data[GraphEdge.TAG].shape;
        }
        set shape(value) {
            this.data[GraphEdge.TAG].shape = value;
        }
    }
    GraphEdge.Class = Class;
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
                [GraphEdge.TAG]: {
                    version: GraphEdge.VERSION,
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
    GraphEdge.Builder = Builder;
    GraphEdge.TypeGuard = Edge.TagTypeGuard(GraphEdge.TAG, GraphEdge.VERSION);
})(GraphEdge || (GraphEdge = {}));
export default GraphEdge;
//# sourceMappingURL=GraphEdge.js.map