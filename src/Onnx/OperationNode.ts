import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import type { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import type OnnxGraph from "./OnnxGraph.js";
import type { AttributeMap, AttributeValue, ValueNode } from "./OnnxTypes.js";
import { isValueNode } from "./Utils.js";
import type { NodeSnapshot } from "./transformation/tracking/GraphActions.js";

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

        get attributes(): AttributeMap {
            return this.data[TAG].attributes ?? {};
        }

        set attributes(attrs: AttributeMap) {
            this.data[TAG].attributes = attrs;
        }

        setAttributes(attrs: AttributeMap): void {
            this.attributes = attrs;
        }

        getAttributes(): AttributeMap {
            return this.attributes;
        }

        get getIncomers(): EdgeCollection<OnnxEdge.Class> {
            return this.incomers.filterIs(OnnxEdge);
        }

        get getOutgoers(): EdgeCollection<OnnxEdge.Class> {
            return this.outgoers.filterIs(OnnxEdge);
        }

        getInputs(): ValueNode[] | undefined {
            return this.data[TAG].inputs;
        }

        getOutputs(): ValueNode[] {
            return this.outgoers.targets.toArray().filter(isValueNode);
        }

        setInputs(inputs: ValueNode[]): void {
            this.data[TAG].inputs = inputs;
        }

        // --- Region Management ---

        get regions(): OnnxGraph.Class[] {
            return this.data[TAG].regions ?? [];
        }

        getRegion(index: number): OnnxGraph.Class | undefined {
            return this.regions[index];
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

        public toSnapshot(): NodeSnapshot {
            return {
                kind: "OperationNode",
                id: this.id,
                opType: this.type,
                attributes: { ...this.attributes },
                inputs: (this.getInputs() ?? []).map((n) => n.id),
                regions: [...this.regions],
                metadata: { ...this.metadata },
            };
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private type: string;
        private attributes?: AttributeMap | undefined;
        private inputs?: ValueNode[] | undefined;
        private regions?: OnnxGraph.Class[] | undefined;
        private metadata: AttributeMap;

        constructor(
            type: string,
            inputs?: ValueNode[],
            attributes?: AttributeMap,
            regions?: OnnxGraph.Class[],
            metadata: AttributeMap = {},
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
                    ...(this.inputs !== undefined ? { inputs: this.inputs } : {}),
                    ...(this.attributes !== undefined ? { attributes: this.attributes } : {}),
                    ...(this.regions !== undefined ? { regions: this.regions } : {}),
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
            inputs?: ValueNode[];
            attributes?: AttributeMap;
            regions?: OnnxGraph.Class[];
            metadata: AttributeMap;
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default OperationNode;
