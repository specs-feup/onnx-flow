import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
declare namespace initialGraph {
    const TAG = "__specs-onnx__initial_graph";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseGraph.Class<D, S> {
    }
    class Builder implements Graph.Builder<Data, ScratchData> {
        buildData(data: BaseGraph.Data): Data;
        buildScratchData(scratchData: BaseGraph.ScratchData): ScratchData;
    }
    const TypeGuard: Graph.TypeGuard<Data, ScratchData>;
    interface Data extends BaseGraph.Data {
        [TAG]: {
            version: typeof VERSION;
        };
    }
    interface ScratchData extends BaseGraph.ScratchData {
    }
}
export default initialGraph;
//# sourceMappingURL=initialGraph.d.ts.map