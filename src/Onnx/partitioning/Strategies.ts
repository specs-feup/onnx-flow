import OnnxGraph from "../OnnxGraph.js";
import OperationNode from "../OperationNode.js";
import TensorNode from "../TensorNode.js";

export interface PartitionSets {
    head: Set<string>;
    tail: Set<string>;
}

export function splitByAncestor(graph: OnnxGraph.Class, splitNodeId: string): PartitionSets {
    let splitNode = graph.getNodeById(splitNodeId);

    if (!splitNode) {
        throw new Error(`Split node '${splitNodeId}' not found in graph.`);
    }

    // Smart Bubble-Up
    while (splitNode.parent) {
        console.warn(
            `Node ${splitNode.id} is inside a subgraph. Bubbling up to parent ${splitNode.parent.id}.`,
        );
        splitNode = splitNode.parent;
    }

    const headSet = new Set<string>();
    const stack = [splitNode];

    // 1. Backward Traversal (Ancestors)
    while (stack.length > 0) {
        const curr = stack.pop()!;
        if (headSet.has(curr.id)) continue;

        headSet.add(curr.id);

        if (curr.is(TensorNode)) {
            const t = curr.as(TensorNode);
            if (t.type === "input" || t.type === "initializer" || t.type === "constant") {
                continue;
            }
        }

        curr.incomers.forEach((edge) => {
            stack.push(edge.source);
        });
    }

    // 2. Forward Correction: Include Outputs of Head Operations
    // If an Operation is in Head, its output tensors MUST be in Head.
    // We iterate a snapshot of the set to avoid infinite loops if we were adding parents,
    // but here we are adding children (tensors), so it's safe.
    const initialHeadNodes = Array.from(headSet);
    initialHeadNodes.forEach((nodeId) => {
        const node = graph.getNodeById(nodeId);
        // If it's an operation, pull its output tensors into Head
        if (node.is(OperationNode)) {
            node.outgoers.forEach((edge) => {
                if (edge.target.is(TensorNode)) {
                    headSet.add(edge.target.id);
                }
            });
        }
    });

    // 3. Tail Set
    const tailSet = new Set<string>();
    graph.nodes.forEach((node) => {
        if (!headSet.has(node.id)) {
            tailSet.add(node.id);
        }
    });

    return { head: headSet, tail: tailSet };
}
