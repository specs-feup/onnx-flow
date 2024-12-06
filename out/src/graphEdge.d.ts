import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
declare namespace graphEdge {
    const TAG = "__specs-onnx__graph_edge";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseEdge.Class<D, S> {
        get elemType(): string;
        get shape(): number[];
    }
    class Builder implements Edge.Builder<Data, ScratchData> {
        private elemType;
        private shape;
        constructor(elemType: string, shape: number[]);
        buildData(data: BaseEdge.Data): Data;
        buildScratchData(scratchData: BaseEdge.ScratchData): ScratchData;
    }
    const TypeGuard: Edge.TypeGuard<Data, ScratchData>;
    interface Data extends BaseEdge.Data {
        [TAG]: {
            version: typeof VERSION;
            elemType: string;
            shape: number[];
        };
    }
    interface ScratchData extends BaseEdge.ScratchData {
    }
}
export default graphEdge;
//# sourceMappingURL=graphEdge.d.ts.map