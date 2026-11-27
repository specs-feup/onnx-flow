import type OnnxGraph from "../ir/OnnxGraph.js";
import { AncestorSplitStrategy } from "./PartitionStrategy.js";

export interface PartitionSets {
    head: Set<string>;
    tail: Set<string>;
}

/**
 * Convenience helper that delegates to `AncestorSplitStrategy` and returns `{ head, tail }` node ID sets.
 */
export function splitByAncestor(graph: OnnxGraph.Class, splitNodeId: string): PartitionSets {
    const assignment = new AncestorSplitStrategy(splitNodeId).split(graph);
    return {
        head: assignment.get("head")!,
        tail: assignment.get("tail")!,
    };
}
