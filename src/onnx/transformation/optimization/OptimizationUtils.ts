import type OnnxGraph from "../../ir/OnnxGraph.js";
import ConstantNode from "../../ir/ConstantNode.js";
import RegionArgumentNode from "../../ir/RegionArgumentNode.js";
import TensorNode from "../../ir/TensorNode.js";
import type { ValueNode } from "../../ir/OnnxTypes.js";

/**
 * Collects all node IDs referenced via implicit capture inside nested control-flow
 * regions (`Loop`, `If`, `Scan`) so optimization passes never remove or bypass them.
 */
export function collectCapturedIds(
    graph: OnnxGraph.Class,
    out: Set<string> = new Set<string>(),
): Set<string> {
    for (const op of graph.getOperationNodes()) {
        for (const region of op.regions) {
            for (const arg of region.getNodes().filterIs(RegionArgumentNode)) {
                out.add(arg.id);
                out.add(arg.originalName);
            }
            for (const innerOp of region.getOperationNodes()) {
                for (const inp of innerOp.getInputs() ?? []) {
                    if (!region.hasNode(inp.id)) {
                        out.add(inp.id);
                    }
                }
            }
            collectCapturedIds(region, out);
        }
    }
    return out;
}

/**
 * Returns `out` as a `TensorNode` if it is a safe-to-rewrite local intermediate tensor
 * (not a graph input/output and not captured by a nested region), otherwise `undefined`.
 */
export function asRewritableIntermediate(
    out: ValueNode | undefined,
    capturedIds: Set<string>,
): TensorNode.Class | undefined {
    if (out === undefined || !out.is(TensorNode)) return undefined;
    const tn = out.as(TensorNode);
    if (tn.type !== "intermediate") return undefined;
    if (capturedIds.has(tn.id)) return undefined;
    return tn;
}

/** Removes an orphaned `ConstantNode` if it has no remaining consumers and is not captured. */
export function cleanupOrphanConstant(
    node: ValueNode | undefined,
    graph: OnnxGraph.Class,
    capturedIds: Set<string>,
): void {
    if (
        node !== undefined &&
        node.is(ConstantNode) &&
        graph.hasNode(node.id) &&
        node.outgoers.length === 0 &&
        !capturedIds.has(node.id)
    ) {
        node.remove();
    }
}
