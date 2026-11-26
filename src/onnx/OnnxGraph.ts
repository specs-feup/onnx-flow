import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import type { NodeCollection } from "@specs-feup/flow/graph/NodeCollection";
import TensorNode from "./TensorNode.js";
import OperationNode from "./OperationNode.js";
import OnnxEdge from "./OnnxEdge.js";
import ConstantNode from "./ConstantNode.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type { OpsetImport } from "./OnnxTypes.js";
import { LATEST_ONNX_IR_VERSION, LATEST_ONNX_OPSET } from "./utils/Constants.js";

namespace OnnxGraph {
    export const TAG = "__specs-onnx__onnx_graph";
    export const VERSION = "1";

    /**
     * Factory method to instantiate a new OnnxGraph cleanly
     * without requiring downstream apps to import @specs-feup/flow/graph/Graph.
     */
    export function create(): Class {
        return Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    }

    export class Class<
        D extends Data = Data,
        S extends ScratchData = ScratchData,
    > extends BaseGraph.Class<D, S> {
        /** Retrieves all nodes in the graph (Operation, Tensor, and Constant). */
        getNodes(): NodeCollection<BaseNode.Class> {
            return this.nodes;
        }

        /** Retrieves all ConstantNodes (Phase 3). */
        getConstantNodes(): NodeCollection<ConstantNode.Class> {
            return this.nodes.filterIs(ConstantNode);
        }

        // Retrieve all TensorNodes with type 'input' in deterministic creation order
        getInputTensorNodes(): NodeCollection<TensorNode.Class> {
            return this.nodes
                .filterIs(TensorNode)
                .filter((n) => n.type === "input")
                .sort((a, b) => a.creationSeq - b.creationSeq);
        }

        // Retrieve all TensorNodes with type 'output' in deterministic creation order
        getOutputTensorNodes(): NodeCollection<TensorNode.Class> {
            return this.nodes
                .filterIs(TensorNode)
                .filter((n) => n.type === "output")
                .sort((a, b) => a.creationSeq - b.creationSeq);
        }

        // Retrieve all TensorNodes
        getTensorNodes(): NodeCollection<TensorNode.Class> {
            return this.nodes.filterIs(TensorNode);
        }

        // Retrieve all OperationNodes
        getOperationNodes(): NodeCollection<OperationNode.Class> {
            return this.nodes.filterIs(OperationNode);
        }

        hasNode(id: string): boolean {
            return this.getNodeById(id) !== undefined;
        }

        getEdge(sourceId: string, targetId: string): OnnxEdge.Class | undefined {
            const source = this.getNodeById(sourceId);
            const target = this.getNodeById(targetId);

            return source?.outgoers
                .filterIs(OnnxEdge)
                .toArray()
                .find((edge) => edge.target === target);
        }

        getOnnxEdgeById(edgeId: string): OnnxEdge.Class | undefined {
            return this.edges
                .filterIs(OnnxEdge)
                .toArray()
                .find((edge) => edge.id === edgeId);
        }

        /** Original model IR version (or configured target IR version). */
        getIrVersion(): number {
            return this.data[TAG].irVersion ?? LATEST_ONNX_IR_VERSION;
        }

        hasExplicitIrVersion(): boolean {
            return this.data[TAG].explicitIrVersion === true;
        }

        setIrVersion(irVersion: number, explicit: boolean = true): void {
            this.data[TAG].irVersion = irVersion;
            if (explicit) this.data[TAG].explicitIrVersion = true;
        }

        /** Imported opset table recorded from the source model (empty when built programmatically). */
        getOpsetImport(): OpsetImport[] {
            return this.data[TAG].opsetImport ?? [];
        }

        setOpsetImport(imports: OpsetImport[]): void {
            this.data[TAG].opsetImport = imports;
        }

        /** Source model default-domain opset prior to frontend upgrades, if imported. */
        getSourceOpset(): number | undefined {
            return this.data[TAG].sourceOpset;
        }

        setSourceOpset(opset: number): void {
            this.data[TAG].sourceOpset = opset;
        }

        /** Target ONNX opset version used by schema lookup, shape inference, and lowering. */
        getTargetOpset(): number {
            return this.data[TAG].targetOpset ?? LATEST_ONNX_OPSET;
        }

        hasExplicitTargetOpset(): boolean {
            return this.data[TAG].explicitTargetOpset === true;
        }

        setTargetOpset(opset: number, explicit: boolean = true): void {
            this.data[TAG].targetOpset = opset;
            if (explicit) this.data[TAG].explicitTargetOpset = true;
        }

        /**
         * Resolves a node by its type and instance number (e.g., "MatMul", 2).
         * Instance is 1-based (first occurrence is 1).
         */
        getNodeByTypeAndInstance(type: string, instance: number): OperationNode.Class | undefined {
            const matchingOps = this.getOperationNodes()
                .toArray()
                .filter((op) => op.type === type);

            return matchingOps[instance - 1]; // 1-based to 0-based
        }
    }

    export class Builder implements Graph.Builder<Data, ScratchData> {
        private targetOpset?: number | undefined;
        private irVersion?: number | undefined;

        constructor(options?: { targetOpset?: number; irVersion?: number }) {
            this.targetOpset = options?.targetOpset;
            this.irVersion = options?.irVersion;
        }

        buildData(data: BaseGraph.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
                    ...(this.targetOpset !== undefined
                        ? { targetOpset: this.targetOpset, explicitTargetOpset: true }
                        : {}),
                    ...(this.irVersion !== undefined
                        ? { irVersion: this.irVersion, explicitIrVersion: true }
                        : {}),
                },
            };
        }
        buildScratchData(scratchData: BaseGraph.ScratchData): ScratchData {
            return {
                ...scratchData,
            };
        }
    }

    export const TypeGuard = Graph.TagTypeGuard<Data, ScratchData>(TAG, VERSION);

    export interface Data extends BaseGraph.Data {
        [TAG]: {
            version: typeof VERSION;
            irVersion?: number;
            opsetImport?: OpsetImport[];
            sourceOpset?: number;
            targetOpset?: number;
            explicitTargetOpset?: boolean;
            explicitIrVersion?: boolean;
        };
    }

    export interface ScratchData extends BaseGraph.ScratchData {}
}
export default OnnxGraph;
