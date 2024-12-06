import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
declare namespace OnnxEdge {
    const TAG = "__specs-onnx__graph_edge";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseEdge.Class<D, S> {
        get literalType(): number | undefined;
        set literalType(value: number | undefined);
        get shape(): number[];
        set shape(value: number[]);
    }
    class Builder implements Edge.Builder<Data, ScratchData> {
        private literalType?;
        private shape;
        constructor(literalType?: number, shape?: number[]);
        buildData(data: BaseEdge.Data): Data;
        buildScratchData(scratchData: BaseEdge.ScratchData): ScratchData;
    }
    const TypeGuard: Edge.TypeGuard<Data, ScratchData>;
    interface Data extends BaseEdge.Data {
        [TAG]: {
            version: typeof VERSION;
            literalType?: number;
            shape: number[];
        };
    }
    interface ScratchData extends BaseEdge.ScratchData {
    }
}
export default OnnxEdge;
//# sourceMappingURL=OnnxEdge.d.ts.map