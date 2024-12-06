import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
var initialGraph;
(function (initialGraph) {
    initialGraph.TAG = "__specs-onnx__initial_graph";
    initialGraph.VERSION = "1";
    class Class extends BaseGraph.Class {
    }
    initialGraph.Class = Class;
    class Builder {
        buildData(data) {
            return {
                ...data,
                [initialGraph.TAG]: {
                    version: initialGraph.VERSION
                },
            };
        }
        buildScratchData(scratchData) {
            return {
                ...scratchData,
            };
        }
    }
    initialGraph.Builder = Builder;
    initialGraph.TypeGuard = Graph.TagTypeGuard(initialGraph.TAG, initialGraph.VERSION);
})(initialGraph || (initialGraph = {}));
export default initialGraph;
//# sourceMappingURL=initialGraph.js.map