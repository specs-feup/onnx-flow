import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
var graphEdge;
(function (graphEdge) {
    graphEdge.TAG = "__specs-onnx__graph_edge";
    graphEdge.VERSION = "1";
    class Class extends BaseEdge.Class {
        get elemType() {
            return this.data[graphEdge.TAG].elemType;
        }
        get shape() {
            return this.data[graphEdge.TAG].shape;
        }
    }
    graphEdge.Class = Class;
    class Builder {
        elemType;
        shape;
        constructor(elemType, shape) {
            this.elemType = elemType;
            this.shape = shape;
        }
        buildData(data) {
            return {
                ...data,
                [graphEdge.TAG]: {
                    version: graphEdge.VERSION,
                    elemType: this.elemType,
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