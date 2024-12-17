import BaseEdge from "@specs-feup/flow/graph/BaseEdge";
import Edge from "@specs-feup/flow/graph/Edge";

namespace OnnxInnerEdge {

    export const TAG = "__specs-onnx__onnx_inner_edge";
    export const VERSION = "1";

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseEdge.Class<D, S> {

        get order(): number {
            return this.data[TAG].order;
        }
    }   

    export class Builder implements Edge.Builder<Data, ScratchData> {

        private order: number;

        constructor(order: number) {
            this.order = order;
        }

        buildData(data: BaseEdge.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    order: this.order,
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
            order: number;
        };
    }

    export interface ScratchData extends BaseEdge.ScratchData {}

}
export default OnnxInnerEdge;