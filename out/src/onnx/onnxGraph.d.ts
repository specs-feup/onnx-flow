import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import { NodeCollection } from "@specs-feup/flow/graph/NodeCollection";
import tensorNode from "./tensorNode.js";
import operationNode from "./operationNode.js";
declare namespace onnxGraph {
    const TAG = "__specs-onnx__onnx_graph";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseGraph.Class<D, S> {
        getInputTensorNodes(): NodeCollection<tensorNode.Data, tensorNode.ScratchData, tensorNode.Class>;
        getOutputTensorNodes(): NodeCollection<tensorNode.Data, tensorNode.ScratchData, tensorNode.Class>;
        getOperationNodes(): NodeCollection<operationNode.Data, operationNode.ScratchData, operationNode.Class>;
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
export default onnxGraph;
//# sourceMappingURL=onnxGraph.d.ts.map