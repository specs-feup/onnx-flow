import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { DataType, TensorProto } from "./OnnxTypes.js";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";

namespace ConstantNode {
    export const TAG = "__specs-onnx__constant_node";
    export const VERSION = "3"; // Bumped version for isInput field

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {
        /**
         * The underlying raw ONNX TensorProto.
         */
        get constantValue(): TensorProto {
            return this.data[TAG].value;
        }

        get shape(): (number | string)[] {
            return this.data[TAG].value.dims || [];
        }

        get literalType(): DataType {
            return (this.data[TAG].value.dataType as DataType) || DataType.UNDEFINED;
        }

        setShape(shape: (number | string)[]): void {
            this.data[TAG].value.dims = shape;
        }

        setLiteralType(dtype: DataType): void {
            this.data[TAG].value.dataType = dtype;
        }

        // --- Input Flag Management ---
        get isInput(): boolean {
            return this.data[TAG].isInput;
        }

        setIsInput(value: boolean): void {
            this.data[TAG].isInput = value;
        }
        // ----------------------------------

        get getIncomers(): EdgeCollection<OnnxEdge.Class> {
            return this.incomers.filterIs(OnnxEdge);
        }

        get getOutgoers(): EdgeCollection<OnnxEdge.Class> {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private value: TensorProto;
        private isInput: boolean;

        constructor(value: TensorProto, isInput: boolean = false) {
            this.value = value;
            this.isInput = isInput;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    value: this.value,
                    isInput: this.isInput,
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
            value: TensorProto;
            isInput: boolean; // New field
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}
export default ConstantNode;
