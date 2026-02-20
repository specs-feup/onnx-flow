import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { DataType } from "./OnnxTypes.js";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";

namespace RegionArgumentNode {
    export const TAG = "__specs-onnx__region_argument_node";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseNode.Class<D, S> {
        /**
         * The index of this argument in the region's input list.
         * Corresponds to the order of explicit captures in the parent Operation.
         */
        get index(): number {
            return this.data[TAG].index;
        }

        /**
         * The name of the variable in the parent scope that this argument captures.
         * Used to restore implicit links during ONNX export.
         */
        get originalName(): string {
            return this.data[TAG].originalName;
        }

        get literalType(): DataType {
            return this.data[TAG].literalType;
        }

        get shape(): (number | string)[] {
            return this.data[TAG].shape;
        }

        setLiteralType(dtype: DataType): void {
            this.data[TAG].literalType = dtype;
        }

        setShape(shape: (number | string)[]): void {
            this.data[TAG].shape = shape;
        }

        get getOutgoers(): EdgeCollection<OnnxEdge.Class> {
            return this.outgoers.filterIs(OnnxEdge);
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private index: number;
        private originalName: string;
        private literalType: DataType;
        private shape: (number | string)[];

        constructor(
            index: number,
            originalName: string,
            literalType: DataType,
            shape: (number | string)[],
        ) {
            this.index = index;
            this.originalName = originalName;
            this.literalType = literalType;
            this.shape = shape;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    index: this.index,
                    originalName: this.originalName,
                    literalType: this.literalType,
                    shape: this.shape,
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
            index: number;
            originalName: string;
            literalType: DataType;
            shape: (number | string)[];
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default RegionArgumentNode;
