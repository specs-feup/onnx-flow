import { GraphBuilder } from "../../GraphBuilder.js";
import type Node from "@specs-feup/flow/graph/Node";
import type OnnxGraph from "../../OnnxGraph.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type OnnxEdge from "../../OnnxEdge.js";
import type { DataType, KnownShape } from "../../OnnxTypes.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import OperationNode from "../../OperationNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";

import type { GraphAction, NodeSnapshot } from "./GraphActions.js";
import {
    AddNodeAction,
    RemoveNodeAction,
    AddEdgeAction,
    RemoveEdgeAction,
    MutationPatch,
} from "./GraphActions.js";

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
        const snapshot = this.cloneNodeState(node);
        super.removeNode(node);
        this.actions.push(new RemoveNodeAction(snapshot));
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
        if (node.is(TensorNode)) {
            const tn = node.as(TensorNode);
            return {
                kind: "TensorNode",
                id: tn.id,
                literalType: tn.literalType,
                shape: tn.shape,
                tensorType: tn.type,
                extraAttrs: tn.extraAttrs,
                metadata: { ...tn.metadata },
            };
        } else if (node.is(ConstantNode)) {
            const cn = node.as(ConstantNode);
            // Deep copy the TensorProto if necessary, but shallow might be enough if immutable
            return {
                kind: "ConstantNode",
                id: cn.id,
                proto: cn.constantValue,
                isInput: cn.isInput,
                metadata: { ...cn.metadata },
            };
        } else if (node.is(OperationNode)) {
            const op = node.as(OperationNode);
            return {
                kind: "OperationNode",
                id: op.id,
                opType: op.type,
                // We only store the STRING IDs of inputs, because storing references
                // breaks if the target nodes are removed and restored!
                inputs: (op.getInputs() ?? []).map((n) => n.id),
                attributes: { ...op.attributes },
                regions: [...op.regions],
                metadata: { ...op.metadata },
            };
        } else if (node.is(RegionArgumentNode)) {
            const rn = node.as(RegionArgumentNode);
            return {
                kind: "RegionArgumentNode",
                id: rn.id,
                index: rn.index,
                originalName: rn.originalName,
                literalType: rn.literalType,
                shape: rn.shape as KnownShape,
            };
        }
        throw new Error(`Unsupported node type for snapshotting: ${node.id}`);
    }
}
