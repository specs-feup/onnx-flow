import Graph from "@specs-feup/flow/graph/Graph";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxGraph from "../ir/OnnxGraph.js";
import TensorNode from "../ir/TensorNode.js";
import OperationNode from "../ir/OperationNode.js";
import OnnxEdge from "../ir/OnnxEdge.js";
import ConstantNode from "../ir/ConstantNode.js";
import RegionArgumentNode from "../ir/RegionArgumentNode.js";
import type { ValueNode } from "../ir/OnnxTypes.js";
import { asValueNode } from "../utils/index.js";
import type { PartitionSets } from "./Strategies.js";

function cloneTensor(t: TensorNode.Class, targetGraph: OnnxGraph.Class): TensorNode.Class {
    return targetGraph
        .addNode(t.id)
        .init(new TensorNode.Builder(t.literalType, t.shape, t.type, t.extraAttrs))
        .as(TensorNode);
}

function cloneConstant(c: ConstantNode.Class, targetGraph: OnnxGraph.Class): ConstantNode.Class {
    return targetGraph
        .addNode(c.id)
        .init(new ConstantNode.Builder(c.constantValue, c.isInput))
        .as(ConstantNode);
}

/** Deep-clones a control-flow region subgraph recursively using `partitionGraph`. */
function deepCloneRegion(region: OnnxGraph.Class): OnnxGraph.Class {
    const allIds = new Set<string>();
    region.nodes.forEach((n) => allIds.add(n.id));
    return partitionGraph(region, { head: allIds, tail: new Set() }).head;
}

function cloneOp(op: OperationNode.Class, targetGraph: OnnxGraph.Class): OperationNode.Class {
    const clonedRegions =
        op.regions.length > 0 ? op.regions.map((region) => deepCloneRegion(region)) : [];

    return targetGraph
        .addNode(op.id)
        .init(new OperationNode.Builder(op.type, [], op.attributes, clonedRegions))
        .as(OperationNode);
}

function cloneNodeIntoGraph(
    node: BaseNode.Class,
    targetGraph: OnnxGraph.Class,
): BaseNode.Class | undefined {
    if (node.is(TensorNode)) return cloneTensor(node.as(TensorNode), targetGraph);
    if (node.is(OperationNode)) return cloneOp(node.as(OperationNode), targetGraph);
    if (node.is(ConstantNode)) return cloneConstant(node.as(ConstantNode), targetGraph);
    return undefined;
}

function wireTensorInputEdge(
    targetGraph: OnnxGraph.Class,
    clonedInput: BaseNode.Class,
    clonedOp: OperationNode.Class,
): void {
    if (clonedInput.is(TensorNode)) {
        const t = clonedInput.as(TensorNode);
        targetGraph
            .addEdge(t, clonedOp)
            .init(new OnnxEdge.Builder(t.literalType, t.shape))
            .as(OnnxEdge);
    }
}

function wireOpOutputEdges(
    targetGraph: OnnxGraph.Class,
    originalOp: OperationNode.Class,
    clonedOp: OperationNode.Class,
    partitionIds: Set<string>,
    nodeMap: Map<string, BaseNode.Class>,
): void {
    originalOp.outgoers.forEach((edge: OnnxEdge.Class) => {
        if (edge.target.is(TensorNode) && partitionIds.has(edge.target.id)) {
            const clonedT = nodeMap.get(edge.target.id)!.as(TensorNode);
            targetGraph
                .addEdge(clonedOp, clonedT)
                .init(new OnnxEdge.Builder(edge.literalType, edge.shape))
                .as(OnnxEdge);
        }
    });
}

/**
 * Splits `originalGraph` into two self-contained `head` and `tail` `OnnxGraph`
 * partitions according to `sets`, promoting boundary tensors to head outputs /
 * tail inputs and cloning shared initializers into both partitions as needed.
 */
export function partitionGraph(
    originalGraph: OnnxGraph.Class,
    sets: PartitionSets,
): { head: OnnxGraph.Class; tail: OnnxGraph.Class } {
    const headGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const tailGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    const { head: headIds, tail: tailIds } = sets;
    const headMap = new Map<string, BaseNode.Class>();
    const tailMap = new Map<string, BaseNode.Class>();

    // 1. Clone nodes into their assigned partition graph
    originalGraph.nodes.forEach((node) => {
        if (headIds.has(node.id)) {
            const cloned = cloneNodeIntoGraph(node, headGraph);
            if (cloned !== undefined) headMap.set(node.id, cloned);
        } else if (tailIds.has(node.id)) {
            const cloned = cloneNodeIntoGraph(node, tailGraph);
            if (cloned !== undefined) tailMap.set(node.id, cloned);
        }
    });

    // 2. Track constants belonging to Head so shared initializers can be cloned into Tail
    const headInitializers = new Set<string>();
    headMap.forEach((node, id) => {
        if (node.is(ConstantNode)) headInitializers.add(id);
    });

    // 3. Rewire operations, edges, and boundary inputs/outputs
    for (const originalOp of originalGraph.getOperationNodes()) {
        const originalInputs = originalOp.getInputs() ?? [];

        if (headIds.has(originalOp.id)) {
            const clonedOp = headMap.get(originalOp.id)!.as(OperationNode);
            const newInputs: ValueNode[] = [];

            for (const input of originalInputs) {
                const clonedInput = headMap.get(input.id);
                if (clonedInput === undefined) {
                    throw new Error(
                        `[Partition] Op '${originalOp.id}' in Head depends on '${input.id}' which is not in Head.`,
                    );
                }
                newInputs.push(asValueNode(clonedInput));
                wireTensorInputEdge(headGraph, clonedInput, clonedOp);
            }

            clonedOp.setInputs(newInputs);
            wireOpOutputEdges(headGraph, originalOp, clonedOp, headIds, headMap);
        } else if (tailIds.has(originalOp.id)) {
            const clonedOp = tailMap.get(originalOp.id)!.as(OperationNode);
            const newInputs: ValueNode[] = [];

            for (const input of originalInputs) {
                // Case 1: Input already resides in Tail
                const existingTail = tailMap.get(input.id);
                if (existingTail !== undefined) {
                    newInputs.push(asValueNode(existingTail));
                    wireTensorInputEdge(tailGraph, existingTail, clonedOp);
                    continue;
                }

                // Case 2: Input crosses from Head (either shared initializer or boundary tensor)
                if (headIds.has(input.id)) {
                    if (headInitializers.has(input.id)) {
                        const origNode = originalGraph.getNodeById(input.id);
                        if (origNode !== undefined && origNode.is(ConstantNode)) {
                            tailMap.set(
                                input.id,
                                cloneConstant(origNode.as(ConstantNode), tailGraph),
                            );
                        } else if (origNode !== undefined && origNode.is(RegionArgumentNode)) {
                            const ra = origNode.as(RegionArgumentNode);
                            tailMap.set(
                                ra.id,
                                tailGraph
                                    .addNode(ra.id)
                                    .init(
                                        new RegionArgumentNode.Builder(
                                            ra.index,
                                            ra.originalName,
                                            ra.literalType,
                                            ra.shape,
                                        ),
                                    )
                                    .as(RegionArgumentNode),
                            );
                        } else {
                            tailMap.set(input.id, cloneTensor(origNode!.as(TensorNode), tailGraph));
                        }

                        const clonedInput = tailMap.get(input.id)!;
                        newInputs.push(asValueNode(clonedInput));
                        wireTensorInputEdge(tailGraph, clonedInput, clonedOp);
                        continue;
                    }

                    // Boundary Tensor: promote Head tensor to output and synthesize a Tail input
                    const headNode = headMap.get(input.id);
                    if (headNode !== undefined && headNode.is(TensorNode)) {
                        headNode.as(TensorNode).setType("output");
                    } else if (headNode !== undefined && headNode.is(ConstantNode)) {
                        throw new Error(
                            `[Partition] ConstantNode '${input.id}' bypassed the initializer cloning phase.`,
                        );
                    }

                    const ghost = tailGraph
                        .addNode(input.id)
                        .init(new TensorNode.Builder(input.literalType, input.shape, "input"))
                        .as(TensorNode);
                    tailMap.set(input.id, ghost);

                    newInputs.push(asValueNode(ghost));
                    wireTensorInputEdge(tailGraph, ghost, clonedOp);
                }
            }

            clonedOp.setInputs(newInputs);
            wireOpOutputEdges(tailGraph, originalOp, clonedOp, tailIds, tailMap);
        }
    }

    return { head: headGraph, tail: tailGraph };
}
