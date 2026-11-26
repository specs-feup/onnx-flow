import type OnnxGraph from "../ir/OnnxGraph.js";
import { GraphPartitioner } from "./GraphPartitioner.js";
import type { PartitionedGraph } from "./PartitionedGraph.js";
import type { PartitionSets } from "./Strategies.js";

/**
 * Splits `originalGraph` into a 2-way `PartitionedGraph` (`head` and `tail`) according to `sets`,
 * promoting boundary tensors and cloning shared initializers as needed.
 *
 * The returned `PartitionedGraph` exposes `.head` and `.tail` getters for backwards compatibility
 * alongside multi-partition methods (`getPartitions()`, `getExecutionOrder()`, `connections`).
 */
export function partitionGraph(
    originalGraph: OnnxGraph.Class,
    sets: PartitionSets,
): PartitionedGraph {
    return new GraphPartitioner().partition(
        originalGraph,
        new Map<string, Set<string>>([
            ["head", sets.head],
            ["tail", sets.tail],
        ]),
    );
}
