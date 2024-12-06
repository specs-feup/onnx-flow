import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
declare namespace ConstantNode {
    const TAG = "__specs-onnx__constant_node";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseNode.Class<D, S> {
        get value(): number;
    }
    class Builder implements Node.Builder<Data, ScratchData> {
        private value;
        constructor(value: number);
        buildData(data: BaseNode.Data): Data;
        buildScratchData(scratchData: BaseNode.ScratchData): ScratchData;
    }
    const TypeGuard: Node.TypeGuard<Data, ScratchData>;
    interface Data extends BaseNode.Data {
        [TAG]: {
            version: typeof VERSION;
            value: number;
        };
    }
    interface ScratchData extends BaseNode.ScratchData {
    }
}
export default ConstantNode;
//# sourceMappingURL=ConstantNode.d.ts.map