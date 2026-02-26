import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import type { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import type { AttributeMap, AttributeProto, AttributeValue } from "./OnnxTypes.js";

namespace TensorNode {
    export const TAG = "__specs-onnx__tensor_node";
    export const VERSION = "4"; // Bumped version

    export type TensorKind = "input" | "output" | "intermediate" | "index" | "index_aux";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {
        get literalType(): number {
            return this.data[TAG].literalType;
        }

        get shape(): (number | string | undefined)[] {
            return this.data[TAG].shape;
        }

        setShape(shape: (number | string | undefined)[]): void {
            this.data[TAG].shape = shape;
        }

        setLiteralType(dtype: number): void {
            this.data[TAG].literalType = dtype;
        }

        get type(): TensorKind {
            return this.data[TAG].type;
        }

        setType(type: TensorKind): void {
            this.data[TAG].type = type;
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

        get metadata(): AttributeMap {
            return this.data[TAG].metadata;
        }

        getMetadata(key: string): AttributeValue | undefined {
            return this.data[TAG].metadata[key];
        }

        setMetadata(key: string, value: AttributeValue): void {
            this.data[TAG].metadata[key] = value;
        }

        // Helper for legacy 'address' support (optional, if you want to keep the API logic)
        get address(): number | undefined {
            return this.getMetadata("address") as number | undefined;
        }

        setAddress(addr: number): void {
            this.setMetadata("address", addr);
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private literalType: number;
        private shape: (number | string | undefined)[];
        private type: TensorKind;
        private extraAttrs?: AttributeProto[] | undefined;
        private metadata: AttributeMap;

        constructor(
            literalType: number,
            shape: (number | string | undefined)[],
            type: TensorKind,
            extraAttrs?: AttributeProto[],
            metadata: AttributeMap = {},
        ) {
            this.literalType = literalType;
            this.shape = shape;
            this.type = type;
            this.extraAttrs = extraAttrs;
            this.metadata = metadata;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    literalType: this.literalType,
                    shape: this.shape,
                    type: this.type,
                    ...(this.extraAttrs !== undefined ? { extraAttrs: this.extraAttrs } : {}),
                    metadata: this.metadata,
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
            shape: (number | string | undefined)[];
            type: TensorKind;
            extraAttrs?: AttributeProto[];
            metadata: AttributeMap;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default TensorNode;
