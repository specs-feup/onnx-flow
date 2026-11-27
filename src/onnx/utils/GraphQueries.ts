import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../ir/ConstantNode.js";
import type { GraphBuilder } from "../ir/GraphBuilder.js";
import OnnxEdge from "../ir/OnnxEdge.js";
import type OnnxGraph from "../ir/OnnxGraph.js";
import type { ValueNode, Dim, ConcreteValueNode, OnnxNode } from "../ir/OnnxTypes.js";
import { DataType } from "../ir/OnnxTypes.js";
import OperationNode from "../ir/OperationNode.js";
import RegionArgumentNode from "../ir/RegionArgumentNode.js";
import TensorNode from "../ir/TensorNode.js";
import { toStaticShape } from "./ShapeMath.js";
import { makeTensorProto } from "./TensorData.js";

/** Adds a typed `OnnxEdge` from `srcOp` to `dstTensor`. */
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

/** Looks up a tensor-like node (TensorNode or ConstantNode) by ID or original name. */
export function findTensorByOnnxName(
    g: OnnxGraph.Class,
    name?: string,
): ConcreteValueNode | undefined {
    if (name === undefined) return undefined;

    const constants = g.nodes.filterIs(ConstantNode).toArray() as ConstantNode.Class[];
    const tConst = constants.find((n) => n.id === name || n.constantValue.name === name);
    if (tConst) return tConst;

    const tensors = g.getTensorNodes().toArray() as TensorNode.Class[];
    return tensors.find((n) => n.id === name);
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
 * Safely attempts to cast a node to a `ValueNode`.
 * Returns `undefined` if the node is missing or is not a Tensor/Constant/RegionArgument.
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

export function isOnnxNode(node: BaseNode.Class): node is OnnxNode {
    return (
        node.is(TensorNode) ||
        node.is(ConstantNode) ||
        node.is(OperationNode) ||
        node.is(RegionArgumentNode)
    );
}

export function asConcreteValueNode(node: BaseNode.Class): ConcreteValueNode {
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);

    throw new Error(
        `Expected a ConcreteValueNode, but got a different node type for ID: ${node.id}`,
    );
}

export function asOnnxNode(node: BaseNode.Class): OnnxNode {
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);
    if (node.is(OperationNode)) return node.as(OperationNode);
    if (node.is(RegionArgumentNode)) return node.as(RegionArgumentNode);

    throw new Error(`Expected an OnnxNode, but got a different node type for ID: ${node.id}`);
}

/**
 * Safely attempts to cast a node to a `ConcreteValueNode`.
 * Returns `undefined` if the node is missing or is not a Tensor/Constant.
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
 * Safely attempts to cast a node to an `OnnxNode`.
 * Returns `undefined` if the node is missing or is not an ONNX IR node.
 */
export function tryAsOnnxNode(node: BaseNode.Class | undefined): OnnxNode | undefined {
    if (!node) return undefined;
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);
    if (node.is(OperationNode)) return node.as(OperationNode);
    if (node.is(RegionArgumentNode)) return node.as(RegionArgumentNode);
    return undefined;
}

/**
 * Slices a tensor along a specific axis into 1D chunks using ONNX `Gather`.
 * Example: A `[M, K]` tensor chunked on axis 0 returns `M` tensors of shape `[1, K]`.
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
        const idxConst = builder.createConstant(
            `${tensor.id}_idx_${i}`,
            makeTensorProto(DataType.INT64, [1], [i]),
        );
        const gatherOut = builder.createOp("Gather", [tensor, idxConst], { axis })[0];
        chunks.push(gatherOut);
    }

    return chunks;
}

/** Returns all operations in `graph` sorted in topological dependency order (including implicit region captures). */
export function topologicalSortOperationNodes(graph: OnnxGraph.Class): OperationNode.Class[] {
    const opNodes = graph.getOperationNodes().toArray();
    return topologicalSortOperationNodesSubset(graph, opNodes);
}

/** Topologically sorts a subset of operation nodes within `graph`, accounting for implicit region captures. */
export function topologicalSortOperationNodesSubset(
    graph: OnnxGraph.Class,
    opNodes: OperationNode.Class[],
): OperationNode.Class[] {
    const sorted: OperationNode.Class[] = [];
    const visited = new Set<string>();
    const temp = new Set<string>();

    const tensorProducers = new Map<string, OperationNode.Class>();
    for (const op of opNodes) {
        const outTensors = op.getOutgoers.targets.filter((n) => n.is(TensorNode)).toArray();
        for (const t of outTensors as TensorNode.Class[]) {
            tensorProducers.set(t.id, op);
        }
    }

    const extraDeps = new Map<string, Set<OperationNode.Class>>();

    const findImplicitDeps = (opId: string, sg: OnnxGraph.Class) => {
        const innerOps = sg.getOperationNodes().toArray();
        for (const innerOp of innerOps) {
            const inputs = innerOp.getInputs() ?? [];
            for (const input of inputs) {
                if (sg.hasNode(input.id) && !input.is(RegionArgumentNode)) continue;
                const lookupId = input.is(RegionArgumentNode)
                    ? input.as(RegionArgumentNode).originalName
                    : input.id;
                const parentProd = tensorProducers.get(lookupId);

                if (parentProd && parentProd.id !== opId) {
                    let deps = extraDeps.get(opId);
                    if (!deps) {
                        deps = new Set<OperationNode.Class>();
                        extraDeps.set(opId, deps);
                    }
                    deps.add(parentProd);
                }
            }

            for (const nestedSg of innerOp.regions) {
                findImplicitDeps(opId, nestedSg);
            }
        }
    };

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

        const implicitPreds = extraDeps.get(node.id);
        if (implicitPreds) implicitPreds.forEach(visit);

        const checkPred = (n: BaseNode.Class) => {
            if (n.is(OperationNode)) {
                const op = n.as(OperationNode);
                for (const input of op.getInputs() ?? []) {
                    if (input.is(TensorNode) && input.as(TensorNode).type === "intermediate") {
                        checkPred(input);
                    }
                }
            }
            const incomers = n.incomers.toArray();
            for (const edge of incomers) {
                const src = edge.source;
                if (src.is(OperationNode)) visit(src.as(OperationNode));
                else if (src.is(TensorNode) && src.as(TensorNode).type === "intermediate") {
                    checkPred(src);
                }
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
