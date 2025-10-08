import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";

namespace VariableNode {

    export const TAG = "__specs-onnx__variable_node";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {

        get literalType(): number {
            return this.data[TAG].literalType;
        }

        get name(): string {
            return this.data[TAG].name;
        }

        get type(): string {
            return this.data[TAG].type;
        }

        get address(): number {
            return this.data[TAG].address;
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private literalType : number;
        private name : string;
        private type : string;
        private address : number;

        constructor(literalType: number, name : string, type : string) {
            this.literalType = literalType;
            this.name = name;
            this.type = type;
            this.address = 0;  // TODO(Process-ing): Make this not hardcoded
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    literalType: this.literalType,
                    name: this.name,
                    type: this.type,
                    address: this.address,
                },
            };
        }

        buildScratchData(scratchData: BaseNode.ScratchData): ScratchData {
            return {
                ...scratchData,
            };
        }
    }

    export const TypeGuard = Node.TagTypeGuard<Data, ScratchData>(TAG, VERSION);

    export interface Data extends BaseNode.Data {
        [TAG]: {
            version: typeof VERSION;
            literalType : number;
            name : string;
            type : string;
            address : number;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}

}
export default VariableNode;