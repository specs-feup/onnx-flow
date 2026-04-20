import Graph from "@specs-feup/flow/graph/Graph";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxGraph from "../OnnxGraph.js";
import TensorNode from "../TensorNode.js";
import OperationNode from "../OperationNode.js";
import OnnxEdge from "../OnnxEdge.js";
import type { PartitionSets } from "./Strategies.js";
import ConstantNode from "../ConstantNode.js";
import type { ValueNode } from "../OnnxTypes.js";
import { asValueNode } from "../Utils.js";
import RegionArgumentNode from "../RegionArgumentNode.js";

/**
 * Clones a TensorNode into the target graph.
 */
function cloneTensor(t: TensorNode.Class, targetGraph: OnnxGraph.Class): TensorNode.Class {
    return targetGraph
        .addNode(t.id)
        .init(new TensorNode.Builder(t.literalType, t.shape, t.type, t.extraAttrs))
        .as(TensorNode);
}

/**
 * Clones a Graph recursively to safely copy internal regions (Loops/Ifs).
 */
function deepCloneRegion(region: OnnxGraph.Class): OnnxGraph.Class {
    // Put all top-level nodes of this region into the 'head' set
    const allIds = new Set<string>();
    region.nodes.forEach((n) => allIds.add(n.id));

    // Recursively call partitionGraph. Everything goes to 'head', 'tail' is empty.
    const { head } = partitionGraph(region, { head: allIds, tail: new Set() });
    return head;
}

/**
 * Clones an OperationNode into the target graph (without inputs initially).
 */
function cloneOp(op: OperationNode.Class, targetGraph: OnnxGraph.Class): OperationNode.Class {
    // Deeply clone all inner regions (subgraphs)
    const clonedRegions = op.regions ? op.regions.map((region) => deepCloneRegion(region)) : [];

    return targetGraph
        .addNode(op.id)
        .init(
            new OperationNode.Builder(
                op.type,
                [], // Inputs populated later to preserve order
                op.attributes,
                clonedRegions,
            ),
        )
        .as(OperationNode);
}

function cloneConstant(c: ConstantNode.Class, targetGraph: OnnxGraph.Class): ConstantNode.Class {
    return targetGraph
        .addNode(c.id)
        .init(new ConstantNode.Builder(c.constantValue, c.isInput))
        .as(ConstantNode);
}

/**
 * Updates the internal inputs list of an OperationNode.
 */
function setOpInputs(op: OperationNode.Class, inputs: ValueNode[]) {
    op.setInputs(inputs);
}

export function partitionGraph(
    originalGraph: OnnxGraph.Class,
    sets: PartitionSets,
): { head: OnnxGraph.Class; tail: OnnxGraph.Class } {
    const headGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const tailGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    const { head: headIds, tail: tailIds } = sets;

    const headMap = new Map<string, BaseNode.Class>();
    const tailMap = new Map<string, BaseNode.Class>();

    // 1. Initial Clone of All Nodes
    originalGraph.nodes.forEach((node) => {
        if (headIds.has(node.id)) {
            if (node.is(TensorNode)) {
                headMap.set(node.id, cloneTensor(node.as(TensorNode), headGraph));
            } else if (node.is(OperationNode)) {
                headMap.set(node.id, cloneOp(node.as(OperationNode), headGraph));
            } else if (node.is(ConstantNode)) {
                headMap.set(node.id, cloneConstant(node.as(ConstantNode), headGraph));
            }
        } else if (tailIds.has(node.id)) {
            if (node.is(TensorNode)) {
                tailMap.set(node.id, cloneTensor(node.as(TensorNode), tailGraph));
            } else if (node.is(OperationNode)) {
                tailMap.set(node.id, cloneOp(node.as(OperationNode), tailGraph));
            } else if (node.is(ConstantNode)) {
                tailMap.set(node.id, cloneConstant(node.as(ConstantNode), tailGraph));
            }
        }
    });

    // 2. Handle Shared Initializers
    const headInitializers = new Set<string>();
    headMap.forEach((node, id) => {
        if (node.is(ConstantNode)) {
            headInitializers.add(id);
        }
    });

    // 3. Wiring Phase
    const ops = originalGraph.getOperationNodes();

    for (const originalOp of ops) {
        const originalInputs = originalOp.getInputs() ?? [];

        // --- Case A: Op is in HEAD ---
        if (headIds.has(originalOp.id)) {
            const clonedOp = headMap.get(originalOp.id)!.as(OperationNode);
            const newInputs: ValueNode[] = [];

            // 3.1 Head Inputs (Tensor -> Op)
            for (const input of originalInputs) {
                if (!headMap.has(input.id)) {
                    throw new Error(
                        `[Partition] Op '${originalOp.id}' in Head depends on '${input.id}' which is not in Head.`,
                    );
                }

                const clonedInput = headMap.get(input.id)!;
                newInputs.push(asValueNode(clonedInput));

                if (clonedInput.is(TensorNode)) {
                    const t = clonedInput.as(TensorNode);
                    headGraph
                        .addEdge(t, clonedOp)
                        .init(new OnnxEdge.Builder(t.literalType, t.shape))
                        .as(OnnxEdge);
                }
            }
            setOpInputs(clonedOp, newInputs);

            // 3.2 Head Outputs (Op -> Tensor)
            originalOp.outgoers.forEach((edge: OnnxEdge.Class) => {
                if (edge.target.is(TensorNode) && headIds.has(edge.target.id)) {
                    const clonedT = headMap.get(edge.target.id)!.as(TensorNode);
                    headGraph
                        .addEdge(clonedOp, clonedT)
                        .init(new OnnxEdge.Builder(edge.literalType, edge.shape))
                        .as(OnnxEdge);
                }
            });
        }

        // --- Case B: Op is in TAIL ---
        else if (tailIds.has(originalOp.id)) {
            const clonedOp = tailMap.get(originalOp.id)!.as(OperationNode);
            const newInputs: ValueNode[] = [];

            // 3.3 Tail Inputs (Tensor -> Op)
            for (const input of originalInputs) {
                // Option 1: Input exists in Tail (Internal flow)
                if (tailMap.has(input.id)) {
                    const clonedInput = tailMap.get(input.id)!;
                    newInputs.push(asValueNode(clonedInput));
                    if (clonedInput.is(TensorNode)) {
                        const t = clonedInput.as(TensorNode);
                        tailGraph
                            .addEdge(t, clonedOp)
                            .init(new OnnxEdge.Builder(t.literalType, t.shape))
                            .as(OnnxEdge);
                    }
                    continue;
                }

                // Option 2: Input is in Head (Boundary Crossing) or Shared Initializer
                if (headIds.has(input.id)) {
                    if (headInitializers.has(input.id)) {
                        // Clone shared initializer into Tail if missing
                        if (!tailMap.has(input.id)) {
                            const origNode = originalGraph.getNodeById(input.id);
                            if (origNode !== undefined && origNode.is(ConstantNode)) {
                                tailMap.set(
                                    input.id,
                                    cloneConstant(origNode.as(ConstantNode), tailGraph),
                                );
                            } else if (origNode !== undefined && origNode.is(RegionArgumentNode)) {
                                const ra = origNode.as(RegionArgumentNode);
                                const ghostRa = tailGraph
                                    .addNode(ra.id)
                                    .init(new RegionArgumentNode.Builder(ra.index, ra.originalName, ra.literalType, ra.shape))
                                    .as(RegionArgumentNode);
                                tailMap.set(ra.id, ghostRa);
                            } else {
                                tailMap.set(
                                    input.id,
                                    cloneTensor(origNode!.as(TensorNode), tailGraph),
                                );
                            }
                        }
                        const clonedInput = tailMap.get(input.id)!;
                        newInputs.push(asValueNode(clonedInput));

                        if (clonedInput.is(TensorNode)) {
                            const t = clonedInput.as(TensorNode);
                            tailGraph
                                .addEdge(t, clonedOp)
                                .init(new OnnxEdge.Builder(t.literalType, t.shape))
                                .as(OnnxEdge);
                        }
                        continue;
                    }

                    // Boundary Tensor
                    const headNode = headMap.get(input.id);
                    if (headNode !== undefined && headNode.is(TensorNode)) {
                        headNode.as(TensorNode).setType("output");
                    } else if (headNode !== undefined && headNode.is(ConstantNode)) {
                        // This should be unreachable due to the headInitializers check above,
                        // but if it happens, we should catch the structural error.
                        throw new Error(
                            `[Partition] ConstantNode '${input.id}' bypassed the initializer cloning phase.`,
                        );
                    }

                    if (!tailMap.has(input.id)) {
                        const origTensor = input;
                        const ghost = tailGraph
                            .addNode(input.id)
                            .init(
                                new TensorNode.Builder(
                                    origTensor.literalType,
                                    origTensor.shape,
                                    "input",
                                ),
                            )
                            .as(TensorNode);
                        tailMap.set(input.id, ghost);
                    }

                    const ghostInput = tailMap.get(input.id)!;
                    newInputs.push(asValueNode(ghostInput));

                    const t = ghostInput.as(TensorNode);
                    tailGraph
                        .addEdge(t, clonedOp)
                        .init(new OnnxEdge.Builder(t.literalType, t.shape))
                        .as(OnnxEdge);
                }
            }
            setOpInputs(clonedOp, newInputs);

            // 3.4 Tail Outputs (Op -> Tensor)
            originalOp.outgoers.forEach((edge: OnnxEdge.Class) => {
                if (edge.target.is(TensorNode) && tailIds.has(edge.target.id)) {
                    const clonedT = tailMap.get(edge.target.id)!.as(TensorNode);
                    tailGraph
                        .addEdge(clonedOp, clonedT)
                        .init(new OnnxEdge.Builder(edge.literalType, edge.shape))
                        .as(OnnxEdge);
                }
            });
        }
    }

    return { head: headGraph, tail: tailGraph };
}
