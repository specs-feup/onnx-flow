import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";

namespace ConstantNode {

    export const TAG = "__specs-onnx__constant_node";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {

        get value(): number {
            return this.data[TAG].value;
        }

    }

    export class Builder implements Node.Builder<Data, ScratchData> {

        private value: number;

        constructor(value: number | String) {
            if (value instanceof String) {
                value = null;
            } else {
                this.value = value;
            }
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    value: this.value
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
            value: number;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData { }

}
export default ConstantNode;