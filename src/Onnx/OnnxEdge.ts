import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";

namespace OnnxEdge {
    export const TAG = "__specs-onnx__onnx_edge";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseEdge.Class<D, S> {
        get literalType(): number | undefined {
            return this.data[TAG].literalType;
        }

        set literalType(value: number | undefined) {
            this.data[TAG].literalType = value;
        }

        get shape(): (number | string)[] {
            return this.data[TAG].shape;
        }

        set shape(value: number[]) {
            this.data[TAG].shape = value;
        }
    }

    export class Builder implements Edge.Builder<Data, ScratchData> {
        private literalType?: number;
        private shape: (number | string)[];

        constructor(literalType?: number, shape: (number | string)[] = []) {
            this.literalType = literalType;
            this.shape = shape;
        }

        buildData(data: BaseEdge.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    literalType: this.literalType,
                    shape: this.shape,
                },
            };
        }

        buildScratchData(scratchData: BaseEdge.ScratchData): ScratchData {
            return {
                ...scratchData,
            };
        }
    }

    export const TypeGuard = Edge.TagTypeGuard<Data, ScratchData>(TAG, VERSION);

    export interface Data extends BaseEdge.Data {
        [TAG]: {
            version: typeof VERSION;
            literalType?: number;
            shape: (number | string)[];
        };
    }

    export interface ScratchData extends BaseEdge.ScratchData {}
}
export default OnnxEdge;
