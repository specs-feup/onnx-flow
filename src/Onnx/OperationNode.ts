import BaseNode from "@specs-feup/flow/graph/BaseNode";
import Node from "@specs-feup/flow/graph/Node";
import { EdgeCollection } from "@specs-feup/flow/graph/EdgeCollection";
import OnnxEdge from "./OnnxEdge.js";
import OnnxGraph from "./OnnxGraph.js";

namespace OperationNode {
    export const TAG = "__specs-onnx__operation_node";
    export const VERSION = "3"; // Bumped version for regions

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

        // Backward compatibility helpers (mapped to regions)

        getBodySubgraph(): OnnxGraph.Class | undefined {
            // "body" is usually the first/only region in Loop/Scan
            return this.regions[0];
        }

        getThenBranch(): OnnxGraph.Class | undefined {
            // "then_branch" is usually region 0 in If
            return this.regions[0];
        }

        getElseBranch(): OnnxGraph.Class | undefined {
            // "else_branch" is usually region 1 in If
            return this.regions[1];
        }
    }

    export class Builder implements Node.Builder<Data, ScratchData> {
        private type: string;
        private attributes?: Record<string, any>;
        private inputs?: BaseNode.Class[];
        private regions?: OnnxGraph.Class[];

        constructor(
            type: string,
            inputs?: BaseNode.Class[],
            attributes?: Record<string, any>,
            regions?: OnnxGraph.Class[],
        ) {
            this.type = type;
            this.attributes = attributes;
            this.inputs = inputs;
            this.regions = regions;
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
        };
    }

    export interface ScratchData extends BaseNode.ScratchData {}
}

export default OperationNode;
