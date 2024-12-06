import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
declare namespace OnnxInnerEdge {
    const TAG = "__specs-onnx__onnx_inner_edge";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseEdge.Class<D, S> {
        get order(): number;
    }
    class Builder implements Edge.Builder<Data, ScratchData> {
        private order;
        constructor(order: number);
        buildData(data: BaseEdge.Data): Data;
        buildScratchData(scratchData: BaseEdge.ScratchData): ScratchData;
    }
    const TypeGuard: Edge.TypeGuard<Data, ScratchData>;
    interface Data extends BaseEdge.Data {
        [TAG]: {
            version: typeof VERSION;
            order: number;
        };
    }
    interface ScratchData extends BaseEdge.ScratchData {
    }
}
export default OnnxInnerEdge;
//# sourceMappingURL=OnnxInnerEdge.d.ts.map