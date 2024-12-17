import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";

namespace TensorNode {

    export const TAG = "__specs-onnx__tensor_node";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {

        get literalType(): number {
            return this.data[TAG].literalType;
        }

        get shape(): number[] {
            return this.data[TAG].shape;
        }

        get type(): string {
            return this.data[TAG].type;
        }

        get geIncomers(): EdgeCollection<OnnxEdge.Class> {
            return this.incomers.filterIs(OnnxEdge);
        }

        get geOutgoers(): EdgeCollection<OnnxEdge.Class> {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }   

    export class Builder implements Node.Builder<Data, ScratchData> {

        private literalType: number;
        private shape: number[];
        private type: string;

        constructor(literalType: number, shape: number[], type: string) {
            this.literalType = literalType;
            this.shape = shape;
            this.type = type;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    literalType: this.literalType,
                    shape: this.shape,
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
            literalType: number;
            shape: number[];
            type: string;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}

}
export default TensorNode;