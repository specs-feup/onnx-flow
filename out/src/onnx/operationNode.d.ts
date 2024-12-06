import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import graphEdge from "./graphEdge.js";
declare namespace operationNode {
    const TAG = "__specs-onnx__operation_node";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseNode.Class<D, S> {
        get type(): string;
        get geIncomers(): EdgeCollection<graphEdge.Data, graphEdge.ScratchData, graphEdge.Class>;
        get geOutgoers(): EdgeCollection<graphEdge.Data, graphEdge.ScratchData, graphEdge.Class>;
    }
    class Builder implements Node.Builder<Data, ScratchData> {
        private type;
        constructor(type: string);
        buildData(data: BaseNode.Data): Data;
        buildScratchData(scratchData: BaseNode.ScratchData): ScratchData;
    }
    const TypeGuard: Node.TypeGuard<Data, ScratchData>;
    interface Data extends BaseNode.Data {
        [TAG]: {
            version: typeof VERSION;
            type: string;
        };
    }
    interface ScratchData extends BaseNode.ScratchData {
    }
}
export default operationNode;
//# sourceMappingURL=operationNode.d.ts.map