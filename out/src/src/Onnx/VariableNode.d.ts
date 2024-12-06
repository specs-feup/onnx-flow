import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
declare namespace VariableNode {
    const TAG = "__specs-onnx__variable_node";
    const VERSION = "1";
    class Class<D extends Data = Data, S extends ScratchData = ScratchData> extends BaseNode.Class<D, S> {
        get literalType(): number;
        get name(): string;
        get type(): string;
    }
    class Builder implements Node.Builder<Data, ScratchData> {
        private literalType;
        private name;
        private type;
        constructor(literalType: number, name: string, type: string);
        buildData(data: BaseNode.Data): Data;
        buildScratchData(scratchData: BaseNode.ScratchData): ScratchData;
    }
    const TypeGuard: Node.TypeGuard<Data, ScratchData>;
    interface Data extends BaseNode.Data {
        [TAG]: {
            version: typeof VERSION;
            literalType: number;
            name: string;
            type: string;
        };
    }
    interface ScratchData extends BaseNode.ScratchData {
    }
}
export default VariableNode;
//# sourceMappingURL=VariableNode.d.ts.map