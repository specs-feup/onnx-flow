import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
declare namespace TensorNode {
    const TAG = "__specs-onnx__tensor_node";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseNode.Class<D, S> {
        get literalType(): number;
        get shape(): number[];
        get type(): string;
        get geIncomers(): EdgeCollection<OnnxEdge.Data, OnnxEdge.ScratchData, OnnxEdge.Class>;
        get geOutgoers(): EdgeCollection<OnnxEdge.Data, OnnxEdge.ScratchData, OnnxEdge.Class>;
    }
    class Builder implements Node.Builder<Data, ScratchData> {
        private literalType;
        private shape;
        private type;
        constructor(literalType: number, shape: number[], type: string);
        buildData(data: BaseNode.Data): Data;
        buildScratchData(scratchData: BaseNode.ScratchData): ScratchData;
    }
    const TypeGuard: Node.TypeGuard<Data, ScratchData>;
    interface Data extends BaseNode.Data {
        [TAG]: {
            version: typeof VERSION;
            literalType: number;
            shape: number[];
            type: string;
        };
    }
    interface ScratchData extends BaseNode.ScratchData {
    }
}
export default TensorNode;
//# sourceMappingURL=TensorNode.d.ts.map