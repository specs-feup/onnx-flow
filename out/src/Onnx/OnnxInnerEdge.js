import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
var OnnxInnerEdge;
(function (OnnxInnerEdge) {
    OnnxInnerEdge.TAG = "__specs-onnx__onnx_inner_edge";
    OnnxInnerEdge.VERSION = "1";
    class Class extends BaseEdge.Class {
        get order() {
            return this.data[OnnxInnerEdge.TAG].order;
        }
    }
    OnnxInnerEdge.Class = Class;
    class Builder {
        order;
        constructor(order) {
            this.order = order;
        }
        buildData(data) {
            return {
                ...data,
                [OnnxInnerEdge.TAG]: {
                    version: OnnxInnerEdge.VERSION,
                    order: this.order,
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    OnnxInnerEdge.Builder = Builder;
    OnnxInnerEdge.TypeGuard = Edge.TagTypeGuard(OnnxInnerEdge.TAG, OnnxInnerEdge.VERSION);
})(OnnxInnerEdge || (OnnxInnerEdge = {}));
export default OnnxInnerEdge;
//# sourceMappingURL=OnnxInnerEdge.js.map