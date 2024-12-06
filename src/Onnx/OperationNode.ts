import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";

namespace OperationNode {

    export const TAG = "__specs-onnx__operation_node";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {

        get type(): string {
            return this.data[TAG].type;
        }

        set type(newType: string) {
            this.data[TAG].type = newType;
        }

        get geIncomers(): EdgeCollection<
        OnnxEdge.Data,
        OnnxEdge.ScratchData,
        OnnxEdge.Class
        > {
            return this.incomers.filterIs(OnnxEdge);
        }

        get geOutgoers(): EdgeCollection<
        OnnxEdge.Data,
        OnnxEdge.ScratchData,
        OnnxEdge.Class
        > {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }   

    export class Builder implements Node.Builder<Data, ScratchData> {

        private type: string;

        constructor(type: string) {
            this.type = type;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    type: this.type,
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
            type: string;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}

}
export default OperationNode;