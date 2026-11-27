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
import {
    PartitionedGraph,
    type GraphPartition,
    type PartitionConnection,
} from "./PartitionedGraph.js";
import type { PartitionAssignment, PartitionStrategy } from "./PartitionStrategy.js";

interface PartitionBuildState {
    id: string;
    graph: OnnxGraph.Class;
    assignedIds: Set<string>;
    nodeMap: Map<string, BaseNode.Class>;
}

/**
 * Splits an `OnnxGraph` into N self-contained `OnnxGraph` partitions according to a
 * `PartitionStrategy` or explicit `PartitionAssignment`, automatically cloning shared
 * constants, promoting cross-partition boundary tensors, and recording `PartitionConnection`s.
 */
export class GraphPartitioner {
    /**
     * Partitions `originalGraph` into a `PartitionedGraph` holding every resulting subgraph
     * and the directed tensor connections between them.
     */
    public partition(
        originalGraph: OnnxGraph.Class,
        strategyOrAssignment: PartitionStrategy | PartitionAssignment,
    ): PartitionedGraph {
        const assignment: PartitionAssignment =
            "split" in strategyOrAssignment
                ? strategyOrAssignment.split(originalGraph)
                : strategyOrAssignment;

        const states = new Map<string, PartitionBuildState>();
        const nodeOwner = new Map<string, string>();

        for (const [partitionId, nodeIds] of assignment.entries()) {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            g.setTargetOpset(originalGraph.getTargetOpset(), false);
            g.setIrVersion(originalGraph.getIrVersion(), false);

            states.set(partitionId, {
                id: partitionId,
                graph: g,
                assignedIds: nodeIds,
                nodeMap: new Map(),
            });

            for (const nodeId of nodeIds) {
                if (!nodeOwner.has(nodeId)) {
                    nodeOwner.set(nodeId, partitionId);
                }
            }
        }

        // 1. Clone each node into its owning partition graph
        originalGraph.nodes.forEach((node) => {
            const ownerId = nodeOwner.get(node.id);
            if (ownerId === undefined) return;
            const state = states.get(ownerId)!;
            const cloned = this.cloneNodeIntoGraph(node, state.graph);
            if (cloned !== undefined) {
                state.nodeMap.set(node.id, cloned);
            }
        });

        // 2. Rewire operations, clone shared initializers, and promote boundary tensors
        const connections: PartitionConnection[] = [];
        const seenConnections = new Set<string>();

        for (const originalOp of originalGraph.getOperationNodes()) {
            const opPartitionId = nodeOwner.get(originalOp.id);
            if (opPartitionId === undefined) continue;

            const state = states.get(opPartitionId)!;
            const clonedOp = state.nodeMap.get(originalOp.id)!.as(OperationNode);
            const originalInputs = originalOp.getInputs() ?? [];
            const newInputs: ValueNode[] = [];

            for (const input of originalInputs) {
                const resolvedInput = this.resolveOperationInput(
                    originalGraph,
                    input,
                    state,
                    states,
                    nodeOwner,
                    connections,
                    seenConnections,
                );
                newInputs.push(asValueNode(resolvedInput));
                this.wireTensorInputEdge(state.graph, resolvedInput, clonedOp);
            }

            clonedOp.setInputs(newInputs);
            this.wireOpOutputEdges(state.graph, originalOp, clonedOp, state.nodeMap);
        }

        // 3. Assemble GraphPartition descriptors
        const descriptors: GraphPartition[] = [];
        for (const state of states.values()) {
            descriptors.push({
                id: state.id,
                graph: state.graph,
                inputs: state.graph
                    .getInputTensorNodes()
                    .toArray()
                    .map((t) => t.id),
                outputs: state.graph
                    .getOutputTensorNodes()
                    .toArray()
                    .map((t) => t.id),
            });
        }

        return new PartitionedGraph(descriptors, connections);
    }

    /** Deep-clones an entire `OnnxGraph` (used recursively for nested control-flow regions). */
    public cloneGraph(sourceGraph: OnnxGraph.Class): OnnxGraph.Class {
        const allIds = new Set<string>();
        sourceGraph.nodes.forEach((n) => allIds.add(n.id));
        const result = this.partition(sourceGraph, new Map([["clone", allIds]]));
        return result.getGraph("clone")!;
    }

    private resolveOperationInput(
        originalGraph: OnnxGraph.Class,
        input: ValueNode,
        consumerState: PartitionBuildState,
        allStates: Map<string, PartitionBuildState>,
        nodeOwner: Map<string, string>,
        connections: PartitionConnection[],
        seenConnections: Set<string>,
    ): BaseNode.Class {
        // Case 1: Input already exists in the consuming partition
        const existing = consumerState.nodeMap.get(input.id);
        if (existing !== undefined) {
            return existing;
        }

        const ownerId = nodeOwner.get(input.id);
        if (ownerId === undefined) {
            throw new Error(
                `[GraphPartitioner] Input '${input.id}' consumed by partition '${consumerState.id}' ` +
                    `is not assigned to any partition.`,
            );
        }

        const origNode = originalGraph.getNodeById(input.id);

        // Case 2: Shared constants and region arguments are replicated directly into the consumer
        if (origNode !== undefined && origNode.is(ConstantNode)) {
            const cloned = this.cloneConstant(origNode.as(ConstantNode), consumerState.graph);
            consumerState.nodeMap.set(input.id, cloned);
            return cloned;
        }
        if (origNode !== undefined && origNode.is(RegionArgumentNode)) {
            const cloned = this.cloneRegionArgument(
                origNode.as(RegionArgumentNode),
                consumerState.graph,
            );
            consumerState.nodeMap.set(input.id, cloned);
            return cloned;
        }

        // Case 3: Original model input shared across partitions — synthesize a graph input
        // in the consumer without mutating the original owner's input role.
        if (
            origNode !== undefined &&
            origNode.is(TensorNode) &&
            origNode.as(TensorNode).type === "input"
        ) {
            const t = origNode.as(TensorNode);
            const ghostInput = consumerState.graph
                .addNode(t.id)
                .init(new TensorNode.Builder(t.literalType, t.shape, "input", t.extraAttrs))
                .as(TensorNode);
            consumerState.nodeMap.set(t.id, ghostInput);
            return ghostInput;
        }

        // Case 4: Boundary tensor produced in `ownerId` and consumed in `consumerState.id`
        const ownerState = allStates.get(ownerId)!;
        const ownerNode = ownerState.nodeMap.get(input.id);
        if (ownerNode !== undefined && ownerNode.is(TensorNode)) {
            ownerNode.as(TensorNode).setType("output");
        } else if (ownerNode !== undefined && ownerNode.is(ConstantNode)) {
            throw new Error(
                `[GraphPartitioner] ConstantNode '${input.id}' bypassed the initializer cloning phase.`,
            );
        }

        const ghost = consumerState.graph
            .addNode(input.id)
            .init(new TensorNode.Builder(input.literalType, input.shape, "input"))
            .as(TensorNode);
        consumerState.nodeMap.set(input.id, ghost);

        const connKey = `${ownerId}->${consumerState.id}:${input.id}`;
        if (!seenConnections.has(connKey)) {
            seenConnections.add(connKey);
            connections.push({
                fromPartition: ownerId,
                toPartition: consumerState.id,
                tensorId: input.id,
            });
        }

        return ghost;
    }

    private cloneTensor(t: TensorNode.Class, targetGraph: OnnxGraph.Class): TensorNode.Class {
        return targetGraph
            .addNode(t.id)
            .init(new TensorNode.Builder(t.literalType, t.shape, t.type, t.extraAttrs))
            .as(TensorNode);
    }

    private cloneConstant(c: ConstantNode.Class, targetGraph: OnnxGraph.Class): ConstantNode.Class {
        return targetGraph
            .addNode(c.id)
            .init(new ConstantNode.Builder(c.constantValue, c.isInput))
            .as(ConstantNode);
    }

    private cloneRegionArgument(
        ra: RegionArgumentNode.Class,
        targetGraph: OnnxGraph.Class,
    ): RegionArgumentNode.Class {
        return targetGraph
            .addNode(ra.id)
            .init(
                new RegionArgumentNode.Builder(ra.index, ra.originalName, ra.literalType, ra.shape),
            )
            .as(RegionArgumentNode);
    }

    private cloneOp(op: OperationNode.Class, targetGraph: OnnxGraph.Class): OperationNode.Class {
        const clonedRegions =
            op.regions.length > 0 ? op.regions.map((region) => this.cloneGraph(region)) : [];

        return targetGraph
            .addNode(op.id)
            .init(new OperationNode.Builder(op.type, [], op.attributes, clonedRegions))
            .as(OperationNode);
    }

    private cloneNodeIntoGraph(
        node: BaseNode.Class,
        targetGraph: OnnxGraph.Class,
    ): BaseNode.Class | undefined {
        if (node.is(TensorNode)) return this.cloneTensor(node.as(TensorNode), targetGraph);
        if (node.is(OperationNode)) return this.cloneOp(node.as(OperationNode), targetGraph);
        if (node.is(ConstantNode)) return this.cloneConstant(node.as(ConstantNode), targetGraph);
        if (node.is(RegionArgumentNode)) {
            return this.cloneRegionArgument(node.as(RegionArgumentNode), targetGraph);
        }
        return undefined;
    }

    private wireTensorInputEdge(
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

    private wireOpOutputEdges(
        targetGraph: OnnxGraph.Class,
        originalOp: OperationNode.Class,
        clonedOp: OperationNode.Class,
        nodeMap: Map<string, BaseNode.Class>,
    ): void {
        originalOp.outgoers.forEach((edge: OnnxEdge.Class) => {
            if (edge.target.is(TensorNode) && nodeMap.has(edge.target.id)) {
                const clonedT = nodeMap.get(edge.target.id)!.as(TensorNode);
                targetGraph
                    .addEdge(clonedOp, clonedT)
                    .init(new OnnxEdge.Builder(edge.literalType, edge.shape))
                    .as(OnnxEdge);
            }
        });
    }
}
