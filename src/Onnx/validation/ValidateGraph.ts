import OnnxGraph from "../OnnxGraph.js";
import TensorNode from "../TensorNode.js";
import OperationNode from "../OperationNode.js";
import ConstantNode from "../ConstantNode.js";
import { DataType } from "../OnnxTypes.js";
import validateScope from "./ValidateScope.js";

/**
 * Global Graph Validation Options
 */
export interface ValidationOptions {
    checkDanglingEdges?: boolean; // Edges pointing to non-existent nodes
    checkOrphans?: boolean; // Nodes with no connections (warnings)
    checkTypeConsistency?: boolean; // Output dtype matches Input dtype expectation
    checkScope?: boolean; // Enforce RegionArgumentNode boundaries
}

/**
 * Validates the graph integrity. Throws errors on structural failures.
 */
export default function validateGraph(
    graph: OnnxGraph.Class,
    options: ValidationOptions = { checkDanglingEdges: true, checkScope: true },
): void {
    const nodes = graph.getNodes();
    const nodeSet = new Set<string>(nodes.toArray().map((n) => n.id));

    for (const node of nodes) {
        // 1. Check Outgoing Edges
        if (options.checkDanglingEdges) {
            for (const edge of node.outgoers) {
                if (!nodeSet.has(edge.target.id)) {
                    throw new Error(
                        `[Integrity Error] Node '${node.id}' has outgoing edge to missing node '${edge.target.id}'.`,
                    );
                }
            }
        }

        // 2. Check Incoming Edges
        if (options.checkDanglingEdges) {
            for (const edge of node.incomers) {
                if (!nodeSet.has(edge.source.id)) {
                    throw new Error(
                        `[Integrity Error] Node '${node.id}' has incoming edge from missing node '${edge.source.id}'.`,
                    );
                }
            }
        }

        // 3. Recursive Region Checks
        if (node.is(OperationNode)) {
            const op = node.as(OperationNode);
            op.regions.forEach((region) => {
                // Recurse into child regions
                validateGraph(region, options);
            });
        }

        // 4. Type Consistency (Basic)
        if (options.checkTypeConsistency && node.is(TensorNode)) {
            const tn = node.as(TensorNode);
            if (tn.literalType === DataType.UNDEFINED && tn.type !== "index") {
                console.warn(`[Validation Warning] Tensor '${tn.id}' has UNDEFINED data type.`);
            }
        }

        // 5. Orphan Check
        if (options.checkOrphans) {
            if (node.is(ConstantNode) && node.outgoers.length === 0) {
                console.warn(`[Validation Warning] Constant '${node.id}' is unused (orphan).`);
            }
        }
    }

    // 6. Scope Validation
    if (options.checkScope) {
        validateScope(graph);
    }
}
