import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import { AttributeProto, TensorProto } from "./OnnxTypes.js";

namespace TensorNode {
    export const TAG = "__specs-onnx__tensor_node";
    export const VERSION = "2";

    export type TensorKind =
        | "input"
        | "output"
        | "initializer"
        | "intermediate"
        | "constant"
        | "index"
        | "index_aux";

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

        get constantValue(): TensorProto | undefined {
            return this.data[TAG].constantValue;
        }

        get originalInitializer(): TensorProto | undefined {
            return this.data[TAG].originalInitializer;
        }

        get extraAttrs(): AttributeProto[] | undefined {
            return this.data[TAG].extraAttrs;
        }

        isConstant(): boolean {
            return this.data[TAG].type === "constant" && !!this.data[TAG].constantValue;
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
        private constantValue?: TensorProto;
        private originalInitializer?: TensorProto;
        private extraAttrs?: AttributeProto[];

        constructor(
            literalType: number,
            shape: (number | string)[],
            type: TensorKind,
            constantValue?: TensorProto,
            originalInitializer?: TensorProto,
            extraAttrs?: AttributeProto[],
        ) {
            this.literalType = literalType;
            this.shape = shape;
            this.type = type;
            this.constantValue = constantValue;
            this.originalInitializer = originalInitializer;
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
                    constantValue: this.constantValue,
                    originalInitializer: this.originalInitializer,
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
            constantValue?: TensorProto;
            originalInitializer?: TensorProto;
            extraAttrs?: AttributeProto[];
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default TensorNode;
