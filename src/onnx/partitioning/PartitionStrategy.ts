import type OnnxGraph from "../ir/OnnxGraph.js";
import OperationNode from "../ir/OperationNode.js";
import TensorNode from "../ir/TensorNode.js";

/** Maps each partition ID to the set of original graph node IDs assigned to that partition. */
export type PartitionAssignment = Map<string, Set<string>>;

/**
 * Strategy contract for deciding how nodes of an `OnnxGraph` should be assigned
 * across two or more named partitions.
 */
export interface PartitionStrategy {
    readonly name: string;

    /** Assigns every node ID in `graph` to a partition bucket. */
    split(graph: OnnxGraph.Class): PartitionAssignment;
}

/**
 * Splits an `OnnxGraph` into two partitions (`head` and `tail`) by placing `splitNodeId`
 * and all of its backward ancestors (plus output tensors of head operations) in `head`,
 * and all remaining nodes in `tail`.
 */
export class AncestorSplitStrategy implements PartitionStrategy {
    public readonly name = "AncestorSplit";

    constructor(
        public readonly splitNodeId: string,
        private readonly headId: string = "head",
        private readonly tailId: string = "tail",
    ) {}

    public split(graph: OnnxGraph.Class): PartitionAssignment {
        const splitNode = graph.getNodeById(this.splitNodeId);
        if (splitNode === undefined) {
            throw new Error(
                `[AncestorSplitStrategy] Node '${this.splitNodeId}' was not found in graph.`,
            );
        }

        const headSet = new Set<string>();
        const stack = [splitNode];

        // 1. Backward traversal over ancestors of the split node
        while (stack.length > 0) {
            const curr = stack.pop()!;
            if (headSet.has(curr.id)) continue;

            headSet.add(curr.id);

            if (curr.is(TensorNode) && curr.as(TensorNode).type === "input") {
                continue;
            }

            if (curr.is(OperationNode)) {
                const op = curr.as(OperationNode);
                const inputs = op.getInputs() ?? [];
                for (const input of inputs) {
                    if (graph.hasNode(input.id)) {
                        stack.push(input);
                    }
                }
            }

            curr.incomers.forEach((edge) => {
                stack.push(edge.source);
            });
        }

        // 2. Forward correction: if an operation is in head, its output tensors must also be in head
        const initialHeadNodes = Array.from(headSet);
        for (const nodeId of initialHeadNodes) {
            const node = graph.getNodeById(nodeId);
            if (node !== undefined && node.is(OperationNode)) {
                node.outgoers.forEach((edge) => {
                    if (edge.target.is(TensorNode)) {
                        headSet.add(edge.target.id);
                    }
                });
            }
        }

        // 3. Complement tail set
        const tailSet = new Set<string>();
        graph.nodes.forEach((node) => {
            if (!headSet.has(node.id)) {
                tailSet.add(node.id);
            }
        });

        return new Map<string, Set<string>>([
            [this.headId, headSet],
            [this.tailId, tailSet],
        ]);
    }
}
