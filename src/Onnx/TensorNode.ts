import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import { AttributeProto } from "./OnnxTypes.js";

namespace TensorNode {
    export const TAG = "__specs-onnx__tensor_node";
    export const VERSION = "3"; // Bumped version

    export type TensorKind = "input" | "output" | "intermediate" | "index" | "index_aux";
    // Removed: "initializer", "constant"

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {
        get literalType(): number {
            return this.data[TAG].literalType;
        }

        get shape(): (number | string)[] {
            return this.data[TAG].shape;
        }

        setShape(shape: (number | string)[]): void {
            this.data[TAG].shape = shape;
        }

        setLiteralType(dtype: number): void {
            this.data[TAG].literalType = dtype;
        }

        get type(): TensorKind {
            return this.data[TAG].type;
        }

        get address(): number {
            return this.data[TAG].address;
        }

        get extraAttrs(): AttributeProto[] | undefined {
            return this.data[TAG].extraAttrs;
        }

        get getIncomers(): EdgeCollection<OnnxEdge.Class> {
            return this.incomers.filterIs(OnnxEdge);
        }

        get getOutgoers(): EdgeCollection<OnnxEdge.Class> {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private literalType: number;
        private shape: (number | string)[];
        private type: TensorKind;
        private address: number;
        private extraAttrs?: AttributeProto[];

        constructor(
            literalType: number,
            shape: (number | string)[],
            type: TensorKind,
            extraAttrs?: AttributeProto[],
        ) {
            this.literalType = literalType;
            this.shape = shape;
            this.type = type;
            this.address = 0;
            this.extraAttrs = extraAttrs;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    literalType: this.literalType,
                    shape: this.shape,
                    type: this.type,
                    address: this.address,
                    extraAttrs: this.extraAttrs,
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
            shape: (number | string)[];
            type: TensorKind;
            address: number;
            extraAttrs?: AttributeProto[];
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default TensorNode;
