import { GraphBuilder } from "../../GraphBuilder.js";
import type Node from "@specs-feup/flow/graph/Node";
import type OnnxGraph from "../../OnnxGraph.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxEdge from "../../OnnxEdge.js";
import type { DataType, KnownShape, ValueNode } from "../../OnnxTypes.js";
import type { GraphAction, NodeSnapshot } from "./GraphActions.js";
import {
    AddNodeAction,
    RemoveNodeAction,
    AddEdgeAction,
    RemoveEdgeAction,
    MutationPatch,
    UpdateNodeInputsAction,
} from "./GraphActions.js";
import { isOnnxNode, asOnnxNode } from "../../Utils.js";
import type OperationNode from "../../OperationNode.js";

export class TrackedGraphBuilder extends GraphBuilder {
    private actions: GraphAction[] = [];

    constructor(
        graph: OnnxGraph.Class,
        public readonly opportunityId: string,
        public readonly description: string,
        scopeTag: string = "",
    ) {
        super(graph, scopeTag);
    }

    // --- NODE INTERCEPTORS ---
    override addNode<
        T extends BaseNode.Class,
        D extends BaseNode.Data,
        S extends BaseNode.ScratchData,
    >(nodeId: string, builder: Node.Builder<D, S>): T {
        const node = super.addNode<T, D, S>(nodeId, builder);
        // Snapshot it immediately after creation so we know how to recreate it
        const snapshot = this.cloneNodeState(node);
        this.actions.push(new AddNodeAction(snapshot));
        return node;
    }

    override removeNode(node: BaseNode.Class): void {
        const connectedEdges = [...node.incomers.toArray(), ...node.outgoers.toArray()];

        for (const edge of connectedEdges) {
            if (edge.is(OnnxEdge)) {
                this.removeEdge(edge.as(OnnxEdge));
            }
        }

        const snapshot = this.cloneNodeState(node);
        super.removeNode(node);
        this.actions.push(new RemoveNodeAction(snapshot));
    }

    override replaceAllUsesWith(oldNode: ValueNode, newNode: ValueNode): void {
        const affectedOps: { op: OperationNode.Class; oldInputs: string[] }[] = [];

        // Find all operations that are about to be mutated
        const findAffected = (g: OnnxGraph.Class) => {
            for (const op of g.getOperationNodes().toArray()) {
                const currentInputs = op.getInputs() ?? [];
                if (currentInputs.some((input) => input.id === oldNode.id)) {
                    affectedOps.push({
                        op,
                        oldInputs: currentInputs.map((n) => n.id),
                    });
                }
                for (const sub of op.regions) {
                    findAffected(sub);
                }
            }
        };
        findAffected(this.graph);

        for (const { op, oldInputs } of affectedOps) {
            const newInputs = oldInputs.map((id) => (id === oldNode.id ? newNode.id : id));
            this.actions.push(new UpdateNodeInputsAction(op.id, oldInputs, newInputs));
        }

        super.replaceAllUsesWith(oldNode, newNode);
    }

    // --- EDGE INTERCEPTORS ---

    override addEdge(
        source: BaseNode.Class,
        target: BaseNode.Class,
        literalType: DataType,
        shape: KnownShape,
    ): OnnxEdge.Class {
        const edge = super.addEdge(source, target, literalType, shape);
        this.actions.push(new AddEdgeAction(source.id, target.id, literalType, shape));
        return edge;
    }

    override removeEdge(edge: OnnxEdge.Class): void {
        this.actions.push(
            new RemoveEdgeAction(edge.source.id, edge.target.id, edge.literalType, edge.shape),
        );
        super.removeEdge(edge);
    }

    // --- PATCH COMMIT ---

    public commitPatch(): MutationPatch {
        const patch = new MutationPatch(this.opportunityId, this.description, [...this.actions]);
        this.actions = []; // Clear the buffer
        return patch;
    }

    // --- SNAPSHOT SERIALIZATION ---

    /**
     * Extracts all raw data necessary to completely rebuild a node from scratch.
     */
    private cloneNodeState(node: BaseNode.Class): NodeSnapshot {
        if (isOnnxNode(node)) {
            return asOnnxNode(node).toSnapshot();
        }
        throw new Error(`Unsupported node type for snapshotting: ${node.id}`);
    }
}
