import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";
import type { DataType, Shape } from "./OnnxTypes.js";

namespace OnnxEdge {
    export const TAG = "__specs-onnx__onnx_edge";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseEdge.Class<D, S> {
        get literalType(): DataType {
            return this.data[TAG].type;
        }

        get shape(): Shape {
            return this.data[TAG].shape;
        }
    }

    export class Builder implements Edge.Builder<Data, ScratchData> {
        private type: DataType;
        private shape: Shape;

        constructor(type: DataType, shape: Shape) {
            this.type = type;
            this.shape = shape;
        }

        buildData(data: BaseEdge.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    type: this.type,
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
            type: DataType;
            shape: Shape;
        };
    }

    export interface ScratchData extends BaseEdge.ScratchData {}
}

export default OnnxEdge;
