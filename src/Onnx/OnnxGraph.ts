import BaseGraph from "@specs-feup/flow/graph/BaseGraph";
import Graph from "@specs-feup/flow/graph/Graph";
import type { NodeCollection } from "@specs-feup/flow/graph/NodeCollection";
import TensorNode from "./TensorNode.js";
import OperationNode from "./OperationNode.js";
import OnnxEdge from "./OnnxEdge.js";
import ConstantNode from "./ConstantNode.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";

namespace OnnxGraph {
    export const TAG = "__specs-onnx__onnx_graph";
    export const VERSION = "1";

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

        // Retrieve all TensorNodes with type 'input'
        getInputTensorNodes(): NodeCollection<TensorNode.Class> {
            return this.nodes.filterIs(TensorNode).filter((n) => n.type === "input");
        }

        // Retrieve all TensorNodes with type 'output'
        getOutputTensorNodes(): NodeCollection<TensorNode.Class> {
            return this.nodes.filterIs(TensorNode).filter((n) => n.type === "output");
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
            if (!source || !target) return undefined;

            return source.outgoers
                .filterIs(OnnxEdge)
                .toArray()
                .find((edge) => edge.target === target);
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
        buildData(data: BaseGraph.Data): Data {
            return {
                ...data,
                [TAG]: {
                    version: VERSION,
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
        };
    }

    export interface ScratchData extends BaseGraph.ScratchData {}
}
export default OnnxGraph;
