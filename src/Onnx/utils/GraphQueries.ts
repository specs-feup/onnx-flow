import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../ConstantNode.js";
import type { GraphBuilder } from "../GraphBuilder.js";
import OnnxEdge from "../OnnxEdge.js";
import type OnnxGraph from "../OnnxGraph.js";
import type { ValueNode, Dim, ConcreteValueNode } from "../OnnxTypes.js";
import { DataType } from "../OnnxTypes.js";
import OperationNode from "../OperationNode.js";
import RegionArgumentNode from "../RegionArgumentNode.js";
import TensorNode from "../TensorNode.js";
import { toStaticShape } from "./ShapeMath.js";
import { makeTensorProto } from "./TensorData.js";

export function formatId(name: string, nodeId: string): string {
    return `${name}_${nodeId}`;
}

export function addEdge(
    g: OnnxGraph.Class,
    srcOp: BaseNode.Class,
    dstTensor: ValueNode,
    dtype: DataType,
    shape?: Array<Dim>,
): void {
    g.addEdge(srcOp, dstTensor)
        .init(new OnnxEdge.Builder(dtype, shape ?? dstTensor.shape))
        .as(OnnxEdge);
}

export function toArrayLike<T = unknown>(nc: unknown): T[] {
    const obj = nc as { toArray?: () => T[] };
    return obj.toArray?.() ?? ((Array.isArray(nc) ? nc : []) as T[]);
}

/** Remove an initializer entry from the graph metadata (cleanup). */
export function removeInitializerByName(g: OnnxGraph.Class, name?: string): void {
    if (name === undefined) return;
    const gRecord = g as unknown as Record<string, unknown>;
    const model = (gRecord["rawModel"] ?? gRecord["model"]) as Record<string, unknown> | undefined;
    const graph = (model?.["graph"] ?? gRecord["graph"]) as Record<string, unknown[]> | undefined;
    for (const f of ["initializer", "sparse_initializer", "input", "value_info"]) {
        if (graph && Array.isArray(graph[f])) {
            // Accept 'unknown', then safely cast it to check the name
            graph[f] = graph[f].filter((x: unknown) => (x as { name?: string }).name !== name);
        }
    }
}

/** Removes a ConstantNode if it has no consumers. */
export function maybeRemoveOrphanConstant(g: OnnxGraph.Class, node?: BaseNode.Class): void {
    // Strict check for ConstantNode
    if (node && node.is(ConstantNode)) {
        const consumers = toArrayLike(node.outgoers.targets);
        if (consumers.length === 0) {
            const onnxName = node.id;
            node.remove();
            removeInitializerByName(g, onnxName);
        }
    }
}

/** Looks up a tensor-like node (TensorNode or ConstantNode) by ID or original name. */
export function findTensorByOnnxName(
    g: OnnxGraph.Class,
    name?: string,
): ConcreteValueNode | undefined {
    if (name === undefined) return undefined;

    // Check Constants
    const constants = g.nodes.filterIs(ConstantNode).toArray() as ConstantNode.Class[];
    const tConst = constants.find((n) => n.id === name || n.constantValue.name === name);
    if (tConst) return tConst;

    // Check Tensors
    const tensors = g.getTensorNodes().toArray() as TensorNode.Class[];
    const t = tensors.find((n) => n.id === name);
    return t;
}

export function findConstantProducerAsTensor(
    g: OnnxGraph.Class,
    onnxName?: string,
): ConstantNode.Class | undefined {
    if (onnxName === undefined) return undefined;
    // In Phase 3, constants are just ConstantNodes.
    return findTensorByOnnxName(g, onnxName)?.tryAs(ConstantNode);
}

export function isValueNode(node: BaseNode.Class): node is ValueNode {
    return node.is(TensorNode) || node.is(ConstantNode) || node.is(RegionArgumentNode);
}

export function asValueNode(node: BaseNode.Class): ValueNode {
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);
    if (node.is(RegionArgumentNode)) return node.as(RegionArgumentNode);

    throw new Error(`Expected a ValueNode, but got a different node type for ID: ${node.id}`);
}

/**
 * Safely attempts to cast a node to a ValueNode.
 * Returns undefined if the node is missing or is not a Tensor/Constant.
 */
export function tryAsValueNode(node: BaseNode.Class | undefined): ValueNode | undefined {
    if (!node) return undefined;
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);
    if (node.is(RegionArgumentNode)) return node.as(RegionArgumentNode);
    return undefined;
}

export function isConcreteValueNode(node: BaseNode.Class): node is ConcreteValueNode {
    return node.is(TensorNode) || node.is(ConstantNode);
}

export function asConcreteValueNode(node: BaseNode.Class): ConcreteValueNode {
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);

    throw new Error(
        `Expected a ConcreteValueNode, but got a different node type for ID: ${node.id}`,
    );
}

/**
 * Safely attempts to cast a node to a ConcreteValueNode.
 * Returns undefined if the node is missing or is not a Tensor/Constant.
 */
export function tryAsConcreteValueNode(
    node: BaseNode.Class | undefined,
): ConcreteValueNode | undefined {
    if (!node) return undefined;
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);
    return undefined;
}

/**
 * Slices a tensor along a specific axis into 1D chunks using ONNX Gather.
 * Example: A [M, K] tensor chunked on axis 0 returns M tensors of shape [K].
 */
export function chunkTensor(
    builder: GraphBuilder,
    tensor: ConcreteValueNode,
    axis: number,
): ConcreteValueNode[] {
    const shape = toStaticShape(tensor.shape);
    const dim = shape[axis];
    const chunks: ConcreteValueNode[] = [];

    for (let i = 0; i < dim; i++) {
        // Create a scalar index for Gather
        const idxConst = builder.createConstant(
            `${tensor.id}_idx_${i}`,
            makeTensorProto(DataType.INT64, [1], [i]),
        );

        // Gather(tensor, index) returns the slice
        const gatherOut = builder.createOp("Gather", [tensor, idxConst], { axis })[0];
        chunks.push(gatherOut);
    }

    return chunks;
}

export function topologicalSortOperationNodes(graph: OnnxGraph.Class): OperationNode.Class[] {
    const opNodes = graph.getOperationNodes().toArray();
    return topologicalSortOperationNodesSubset(graph, opNodes);
}

export function topologicalSortOperationNodesSubset(
    graph: OnnxGraph.Class,
    opNodes: OperationNode.Class[],
): OperationNode.Class[] {
    const sorted: OperationNode.Class[] = [];
    const visited = new Set<string>();
    const temp = new Set<string>();

    // Map tensor id -> producing op in the CURRENT graph
    const tensorProducers = new Map<string, OperationNode.Class>();
    for (const op of opNodes) {
        const outTensors = op.getOutgoers.targets.filter((n) => n.is(TensorNode)).toArray();
        for (const t of outTensors as TensorNode.Class[]) {
            tensorProducers.set(t.id, op);
        }
    }

    // Extra deps from implicit subgraph captures
    const extraDeps = new Map<string, Set<OperationNode.Class>>();

    // Helper to recursively find dependencies from inner regions
    const findImplicitDeps = (opId: string, sg: OnnxGraph.Class) => {
        const innerOps = sg.getOperationNodes().toArray();
        for (const innerOp of innerOps) {
            const inputs = innerOp.getInputs() ?? [];
            for (const input of inputs) {
                // Check if this input comes from the outer graph
                const parentProd = tensorProducers.get(input.id);

                if (parentProd && parentProd.id !== opId) {
                    let deps = extraDeps.get(opId);
                    if (!deps) {
                        deps = new Set<OperationNode.Class>();
                        extraDeps.set(opId, deps);
                    }
                    deps.add(parentProd);
                }
            }

            // Recurse into nested regions (e.g., loops inside loops)
            for (const nestedSg of innerOp.regions) {
                findImplicitDeps(opId, nestedSg);
            }
        }
    };

    // Find all implicit dependencies across boundaries
    for (const op of opNodes) {
        for (const sg of op.regions) {
            findImplicitDeps(op.id, sg);
        }
    }

    const visit = (node: OperationNode.Class) => {
        if (visited.has(node.id) || !graph.hasNode(node.id)) return;
        if (temp.has(node.id)) {
            console.warn(`[TopoSort] Cycle detected at ${node.id}`);
            return;
        }
        temp.add(node.id);

        // 1. Implicit Closure dependencies
        const implicitPreds = extraDeps.get(node.id);
        if (implicitPreds) implicitPreds.forEach(visit);

        // 2. Explicit dependencies (Inputs)
        const checkPred = (n: BaseNode.Class) => {
            if (n.is(OperationNode)) {
                const op = n.as(OperationNode);
                // Follow intermediate tensor inputs recursively
                for (const input of op.getInputs() ?? []) {
                    if (input.is(TensorNode) && input.as(TensorNode).type === "intermediate") {
                        checkPred(input);
                    }
                }
            }
            // Check incomers (edges)
            const incomers = n.incomers.toArray();
            for (const edge of incomers) {
                const src = edge.source;
                if (src.is(OperationNode)) visit(src.as(OperationNode));
                else if (src.is(TensorNode) && src.as(TensorNode).type === "intermediate")
                    checkPred(src);
            }
        };

        checkPred(node);

        temp.delete(node.id);
        visited.add(node.id);
        sorted.push(node);
    };

    opNodes.forEach(visit);
    return sorted;
}
