import OnnxGraph from "../OnnxGraph.js";
import OperationNode from "../OperationNode.js";

/**
 * Validates that all edges in the graph stay within the graph's scope.
 * - Edges cannot point to nodes in a different graph instance.
 * - Implicit captures must use RegionArgumentNode.
 */
export default function validateScope(graph: OnnxGraph.Class, path: string = "root"): void {
    const nodes = graph.getNodes();

    for (const node of nodes) {
        // 1. Validate Outgoing Edges
        for (const edge of node.outgoers) {
            const target = edge.target;
            if (!graph.hasNode(target.id)) {
                throw new Error(
                    `[Scope Error] Node '${node.id}' in graph '${path}' links to '${target.id}' which is NOT in the same graph. ` +
                        `Cross-region edges are forbidden. Use RegionArgumentNode captures.`,
                );
            }
        }

        // 2. Validate Incoming Edges
        for (const edge of node.incomers) {
            const source = edge.source;
            if (!graph.hasNode(source.id)) {
                throw new Error(
                    `[Scope Error] Node '${node.id}' in graph '${path}' receives link from '${source.id}' which is NOT in the same graph.`,
                );
            }
        }

        // 3. Recursive check for Regions
        if (node.is(OperationNode)) {
            const op = node.as(OperationNode);
            op.regions.forEach((region, i) => {
                validateScope(region, `${path}/${op.id}_region_${i}`);
            });
        }
    }
}
