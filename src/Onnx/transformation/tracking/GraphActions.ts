import type OnnxGraph from "../../OnnxGraph.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxEdge from "../../OnnxEdge.js";
import OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import type {
    AttributeMap,
    AttributeProto,
    DataType,
    KnownShape,
    Shape,
    TensorProto,
    ValueNode,
} from "../../OnnxTypes.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";

/** * Represents an atomic change to the graph.
 * Can be serialized to JSON for history exports.
 */
export interface GraphAction {
    readonly type: "ADD_NODE" | "REMOVE_NODE" | "ADD_EDGE" | "REMOVE_EDGE";
    apply(graph: OnnxGraph.Class): void;
    revert(graph: OnnxGraph.Class): void;
}

// ------------------------------------------------------------------
// 1. ADD / REMOVE EDGE ACTIONS
// ------------------------------------------------------------------

export class AddEdgeAction implements GraphAction {
    public readonly type = "ADD_EDGE";
    constructor(
        public readonly sourceId: string,
        public readonly targetId: string,
        public readonly literalType: DataType,
        public readonly shape: KnownShape,
    ) {}

    apply(graph: OnnxGraph.Class): void {
        const source = graph.getNodeById(this.sourceId);
        const target = graph.getNodeById(this.targetId);
        if (source && target) {
            graph.addEdge(source, target).init(new OnnxEdge.Builder(this.literalType, this.shape));
        }
    }

    revert(graph: OnnxGraph.Class): void {
        const edge = graph.getEdge(this.sourceId, this.targetId);
        if (edge) edge.remove();
    }
}

export class RemoveEdgeAction implements GraphAction {
    public readonly type = "REMOVE_EDGE";
    constructor(
        public readonly sourceId: string,
        public readonly targetId: string,
        public readonly literalType: DataType,
        public readonly shape: KnownShape | Shape,
    ) {}

    apply(graph: OnnxGraph.Class): void {
        const edge = graph.getEdge(this.sourceId, this.targetId);
        if (edge) edge.remove();
    }

    revert(graph: OnnxGraph.Class): void {
        const source = graph.getNodeById(this.sourceId);
        const target = graph.getNodeById(this.targetId);
        if (source && target) {
            graph.addEdge(source, target).init(new OnnxEdge.Builder(this.literalType, this.shape));
        }
    }
}

// ------------------------------------------------------------------
// 2. ADD / REMOVE NODE ACTIONS (Requires Node State Snapshots)
// ------------------------------------------------------------------

export type NodeSnapshot =
    | {
          kind: "TensorNode";
          id: string;
          literalType: DataType;
          shape: KnownShape | Shape;
          tensorType: TensorNode.TensorKind;
          extraAttrs?: AttributeProto[] | undefined;
          metadata?: AttributeMap;
      }
    | {
          kind: "RegionArgumentNode";
          id: string;
          index: number;
          originalName: string;
          literalType: DataType;
          shape: KnownShape | Shape;
      }
    | {
          kind: "ConstantNode";
          id: string;
          proto: TensorProto;
          isInput: boolean;
          metadata?: AttributeMap;
      }
    | {
          kind: "OperationNode";
          id: string;
          opType: string;
          attributes: AttributeMap;
          inputs: string[];
          regions: OnnxGraph.Class[];
          metadata?: AttributeMap;
      };

export type EdgeSnapshot = {
    literalType: DataType;
    shape: KnownShape | Shape;
    order?: number; // Optional: in case there is a need to snapshot OnnxInnerEdge too
};

export function restoreSnapshot(graph: OnnxGraph.Class, snap: NodeSnapshot): BaseNode.Class {
    if (snap.kind === "TensorNode") {
        return graph
            .addNode(snap.id)
            .init(
                new TensorNode.Builder(
                    snap.literalType,
                    snap.shape,
                    snap.tensorType,
                    snap.extraAttrs,
                    snap.metadata,
                ),
            );
    } else if (snap.kind === "ConstantNode") {
        return graph
            .addNode(snap.id)
            .init(new ConstantNode.Builder(snap.proto, snap.isInput, snap.metadata));
    } else if (snap.kind === "OperationNode") {
        // Re-resolve inputs from the graph using their string IDs
        const resolvedInputs = snap.inputs
            .map((id) => graph.getNodeById(id) as ValueNode)
            .filter(Boolean);
        return graph
            .addNode(snap.id)
            .init(
                new OperationNode.Builder(
                    snap.opType,
                    resolvedInputs,
                    snap.attributes,
                    snap.regions,
                    snap.metadata,
                ),
            );
    } else {
        return graph
            .addNode(snap.id)
            .init(
                new RegionArgumentNode.Builder(
                    snap.index,
                    snap.originalName,
                    snap.literalType,
                    snap.shape,
                ),
            );
    }
    //throw new Error(`Cannot restore unknown snapshot kind: ${snap.kind}`);
}

export class AddNodeAction implements GraphAction {
    public readonly type = "ADD_NODE";
    constructor(public readonly snapshot: NodeSnapshot) {}

    apply(graph: OnnxGraph.Class): void {
        restoreSnapshot(graph, this.snapshot);
    }

    revert(graph: OnnxGraph.Class): void {
        graph.getNodeById(this.snapshot.id)?.remove();
    }
}

export class RemoveNodeAction implements GraphAction {
    public readonly type = "REMOVE_NODE";
    constructor(public readonly snapshot: NodeSnapshot) {}

    apply(graph: OnnxGraph.Class): void {
        graph.getNodeById(this.snapshot.id)?.remove();
    }

    revert(graph: OnnxGraph.Class): void {
        restoreSnapshot(graph, this.snapshot);
    }
}

// ------------------------------------------------------------------
// 3. THE MUTATION PATCH
// ------------------------------------------------------------------

export class MutationPatch {
    constructor(
        public readonly opportunityId: string,
        public readonly description: string,
        public readonly actions: GraphAction[],
    ) {}

    public apply(graph: OnnxGraph.Class): void {
        // Redo applies forward
        for (const action of this.actions) {
            action.apply(graph);
        }
    }

    public revert(graph: OnnxGraph.Class): void {
        // Undo MUST apply in exactly reverse order (e.g., remove edges before removing nodes)
        for (let i = this.actions.length - 1; i >= 0; i--) {
            this.actions[i].revert(graph);
        }
    }
}
