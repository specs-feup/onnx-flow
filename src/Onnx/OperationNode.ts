import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import OnnxGraph from "./OnnxGraph.js";

namespace OperationNode {
    export const TAG = "__specs-onnx__operation_node";
    export const VERSION = "4"; // Bumped version for regions

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

        get attributes(): Record<string, any> {
            return this.data[TAG].attributes || {};
        }

        set attributes(attrs: Record<string, any>) {
            this.data[TAG].attributes = attrs;
        }

        setAttributes(attrs: Record<string, any>): void {
            this.attributes = attrs;
        }

        getAttributes(): Record<string, any> {
            return this.attributes;
        }

        get getIncomers(): EdgeCollection<OnnxEdge.Class> {
            return this.incomers.filterIs(OnnxEdge);
        }

        get getOutgoers(): EdgeCollection<OnnxEdge.Class> {
            return this.outgoers.filterIs(OnnxEdge);
        }

        getInputs(): BaseNode.Class[] | undefined {
            return this.data[TAG].inputs;
        }

        // --- Region Management ---

        get regions(): OnnxGraph.Class[] {
            return this.data[TAG].regions ?? [];
        }

        getRegion(index: number): OnnxGraph.Class | undefined {
            return this.regions[index];
        }

        get metadata(): Record<string, any> {
            return this.data[TAG].metadata;
        }

        getMetadata<T = any>(key: string): T | undefined {
            return this.data[TAG].metadata[key];
        }

        setMetadata(key: string, value: any): void {
            this.data[TAG].metadata[key] = value;
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private type: string;
        private attributes?: Record<string, any>;
        private inputs?: BaseNode.Class[];
        private regions?: OnnxGraph.Class[];
        private metadata: Record<string, any>;

        constructor(
            type: string,
            inputs?: BaseNode.Class[],
            attributes?: Record<string, any>,
            regions?: OnnxGraph.Class[],
            metadata: Record<string, any> = {},
        ) {
            this.type = type;
            this.attributes = attributes;
            this.inputs = inputs;
            this.regions = regions;
            this.metadata = metadata;
        }

        buildData(data: BaseNode.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    type: this.type,
                    inputs: this.inputs || [],
                    attributes: this.attributes || {},
                    regions: this.regions || [],
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
            type: string;
            inputs?: BaseNode.Class[];
            attributes?: Record<string, any>;
            regions?: OnnxGraph.Class[];
            metadata: Record<string, any>;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default OperationNode;
