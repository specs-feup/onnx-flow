import type OnnxGraph from "./OnnxGraph.js";
import type Node from "@specs-feup/flow/graph/Node";
import type {
    AttributeMap,
    ValueNode,
    ConcreteValueNode,
    TensorProto,
    KnownShape,
    Shape,
} from "./OnnxTypes.js";
import { DataType } from "./OnnxTypes.js";
import OperationNode from "./OperationNode.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import RegionArgumentNode from "./RegionArgumentNode.js";
import OnnxEdge from "./OnnxEdge.js";
import { uniq } from "../utils/index.js";
import { OpRegistry } from "../schema/OpRegistry.js";
import { inferNodeShape } from "../schema/InferShapes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type BaseEdge from "@specs-feup/flow/graph/BaseEdge";

const LOGICAL_BOOL_OPS = new Set([
    "Equal",
    "Greater",
    "Less",
    "And",
    "Or",
    "Not",
    "GreaterOrEqual",
    "LessOrEqual",
]);

/**
 * Cached structural metadata for a subgraph region, invalidated whenever its
 * node count changes or nodes are added/removed through `GraphBuilder`.
 */
export interface SubgraphMeta {
    nodeCount: number;
    hasNestedRegions: boolean;
    hasRenamedArgs: boolean;
}

export const subgraphMetaCache = new WeakMap<OnnxGraph.Class, SubgraphMeta>();

/** Returns cached region metadata to avoid redundant full scans during `replaceAllUsesWith`. */
export function getSubgraphMeta(g: OnnxGraph.Class): SubgraphMeta {
    const nodeCount = g.nodes.length;
    const cached = subgraphMetaCache.get(g);
    if (cached && cached.nodeCount === nodeCount) {
        return cached;
    }
    let hasNestedRegions = false;
    for (const op of g.getOperationNodes()) {
        if (op.regions.length > 0) {
            hasNestedRegions = true;
            break;
        }
    }
    let hasRenamedArgs = false;
    for (const arg of g.getNodes().filterIs(RegionArgumentNode)) {
        if (arg.id !== arg.originalName) {
            hasRenamedArgs = true;
            break;
        }
    }
    const meta: SubgraphMeta = { nodeCount, hasNestedRegions, hasRenamedArgs };
    subgraphMetaCache.set(g, meta);
    return meta;
}

export function getDirectReplaceSources(
    g: OnnxGraph.Class,
    rootGraph: OnnxGraph.Class,
    oldNode: ValueNode,
): ValueNode[] {
    if (g === rootGraph) return [oldNode];
    const meta = getSubgraphMeta(g);
    const matched: ValueNode[] = [];
    const seenIds = new Set<string>();
    if (g.hasNode(oldNode.id)) {
        const n = g.getNodeById(oldNode.id)!;
        if (
            (n.is(RegionArgumentNode) && n.as(RegionArgumentNode).originalName === oldNode.id) ||
            n.is(TensorNode) ||
            n.is(ConstantNode)
        ) {
            seenIds.add(n.id);
            matched.push(n as ValueNode);
        }
    }
    if (meta.hasRenamedArgs) {
        for (const arg of g.getNodes().filterIs(RegionArgumentNode)) {
            if (!seenIds.has(arg.id) && arg.originalName === oldNode.id) {
                seenIds.add(arg.id);
                matched.push(arg);
            }
        }
    }
    return matched;
}

export class GraphBuilder {
    constructor(
        public graph: OnnxGraph.Class,
        private scopeTag: string = "",
    ) {}

    /**
     * Centralized method to add a node to the graph.
     * Overridden by TrackedGraphBuilder to log the creation.
     */
    public addNode<
        T extends BaseNode.Class,
        D extends BaseNode.Data,
        S extends BaseNode.ScratchData,
    >(nodeId: string, builder: Node.Builder<D, S>): T {
        subgraphMetaCache.delete(this.graph);
        return this.graph.addNode(nodeId).init(builder) as unknown as T;
    }

    /**
     * Centralized method to add an edge.
     * Overridden by TrackedGraphBuilder to log the addition.
     */
    public addEdge(
        source: BaseNode.Class,
        target: BaseNode.Class,
        literalType: DataType,
        shape: KnownShape | Shape,
    ): OnnxEdge.Class {
        return this.graph
            .addEdge(source, target)
            .init(new OnnxEdge.Builder(literalType, shape))
            .as(OnnxEdge);
    }

    /**
     * Centralized method to remove a node.
     * Overridden by TrackedGraphBuilder to snapshot the node before destruction.
     */
    public removeNode(node: BaseNode.Class): void {
        subgraphMetaCache.delete(this.graph);
        node.remove();
    }

    /**
     * Centralized method to remove an edge.
     */
    public removeEdge(edge: OnnxEdge.Class | BaseEdge.Class): void {
        edge.remove();
    }

    /**
     * Wires incoming edges from every input value node belonging to `this.graph` into `op`.
     */
    private wireInputEdges(op: OperationNode.Class, inputs: ValueNode[]): void {
        for (const input of inputs) {
            if (this.graph.hasNode(input.id)) {
                this.addEdge(input, op, input.literalType, input.shape);
            }
        }
    }

    /**
     * Computes initial fallback output dtype and shape prior to schema-based shape inference.
     */
    private inferInitialOutputSpec(
        type: string,
        inputs: ValueNode[],
        attributes: AttributeMap,
        expectedOutput?: { type: DataType; shape?: KnownShape },
    ): { type: DataType; shape: KnownShape } {
        if (expectedOutput !== undefined) {
            return { type: expectedOutput.type, shape: expectedOutput.shape ?? [] };
        }
        if (inputs.length === 0) {
            return { type: DataType.UNDEFINED, shape: [] };
        }

        let inferredType: DataType = inputs[0].literalType;
        let inferredShape: KnownShape = [];

        if (LOGICAL_BOOL_OPS.has(type)) {
            inferredType = DataType.BOOL;
        } else if (type === "Shape") {
            inferredType = DataType.INT64;
            inferredShape = [(inputs[0].shape.length as number | undefined) ?? 1];
        } else if (type === "Size") {
            inferredType = DataType.INT64;
            inferredShape = [];
        } else if (type === "Cast" && "to" in attributes) {
            inferredType = attributes["to"] as DataType;
        }

        return { type: inferredType, shape: inferredShape };
    }

    /**
     * Creates an OperationNode, wires its inputs, and provisions its output tensors.
     */
    public createOp(
        type: string,
        inputs: ValueNode[],
        attributes: AttributeMap = {},
        expectedOutputs?: { type: DataType; shape?: KnownShape }[],
        regions?: OnnxGraph.Class[],
    ): ConcreteValueNode[] {
        const op = this.addNode(
            uniq(this.graph, `${this.scopeTag}_${type}`),
            new OperationNode.Builder(type, inputs, attributes, regions),
        ).as(OperationNode);

        this.wireInputEdges(op, inputs);

        const schema = OpRegistry.getInstance().get(type, this.graph.getTargetOpset());
        let numOutputs = schema ? schema.outputs.length : 1;

        if (expectedOutputs !== undefined && expectedOutputs.length > numOutputs) {
            numOutputs = expectedOutputs.length;
        } else if ("num_outputs" in attributes) {
            numOutputs = attributes["num_outputs"] as number;
        }

        const outputs: ConcreteValueNode[] = [];
        for (let i = 0; i < numOutputs; i++) {
            const expected =
                expectedOutputs !== undefined && i in expectedOutputs
                    ? expectedOutputs[i]
                    : undefined;
            const initial = this.inferInitialOutputSpec(type, inputs, attributes, expected);

            const outTensor = this.addNode(
                uniq(this.graph, `${this.scopeTag}_${type}_out${i}`),
                new TensorNode.Builder(initial.type, initial.shape, "intermediate"),
            ).as(TensorNode);

            this.addEdge(op, outTensor, outTensor.literalType, outTensor.shape);
            outputs.push(outTensor);
        }

        inferNodeShape(op, this.graph);

        // Explicitly specified expectedOutputs take ultimate precedence
        if (expectedOutputs !== undefined) {
            for (let i = 0; i < expectedOutputs.length; i++) {
                const expectedShape = expectedOutputs[i].shape;
                if (i in outputs && expectedShape !== undefined && expectedShape.length > 0) {
                    outputs[i].setShape(expectedShape);
                }
                if (i in outputs && expectedOutputs[i].type !== DataType.UNDEFINED) {
                    outputs[i].setLiteralType(expectedOutputs[i].type);
                }

                if (i in outputs) {
                    const edge = this.graph.getEdge(op.id, outputs[i].id);
                    if (edge) {
                        this.removeEdge(edge);
                        this.addEdge(op, outputs[i], outputs[i].literalType, outputs[i].shape);
                    }
                }
            }
        }

        return outputs;
    }

    /**
     * Creates an OperationNode and its outputs using exactly specified IDs.
     * Crucial for frontend parsing (`OnnxGraphLoader`) and deterministic pass generation.
     */
    public createOpWithExact(
        opId: string,
        type: string,
        inputs: ValueNode[],
        outputIds: string[],
        attributes: AttributeMap = {},
        expectedOutputs?: { type: DataType; shape?: KnownShape }[],
        regions?: OnnxGraph.Class[],
    ): ConcreteValueNode[] {
        const op = this.addNode(
            opId,
            new OperationNode.Builder(type, inputs, attributes, regions),
        ).as(OperationNode);

        this.wireInputEdges(op, inputs);

        const outputs: ConcreteValueNode[] = [];
        for (let i = 0; i < outputIds.length; i++) {
            const expected =
                expectedOutputs !== undefined && i in expectedOutputs
                    ? expectedOutputs[i]
                    : undefined;
            const initial = this.inferInitialOutputSpec(type, inputs, attributes, expected);

            let outTensor: TensorNode.Class;
            if (this.graph.hasNode(outputIds[i])) {
                outTensor = this.graph.getNodeById(outputIds[i])!.as(TensorNode);
                if (initial.type !== DataType.UNDEFINED) outTensor.setLiteralType(initial.type);
                if (initial.shape.length > 0) outTensor.setShape(initial.shape);
            } else {
                outTensor = this.addNode(
                    outputIds[i],
                    new TensorNode.Builder(initial.type, initial.shape, "intermediate"),
                ).as(TensorNode);
            }

            this.addEdge(op, outTensor, outTensor.literalType, outTensor.shape);
            outputs.push(outTensor);
        }

        inferNodeShape(op, this.graph);
        return outputs;
    }

    /**
     * Creates a ConstantNode safely, ensuring unique IDs.
     */
    public createConstant(id: string, proto: TensorProto): ConstantNode.Class {
        return this.addNode(
            uniq(this.graph, `${this.scopeTag}_${id}`),
            new ConstantNode.Builder(proto),
        ).as(ConstantNode);
    }

    /**
     * Creates a TensorNode safely, ensuring unique IDs and automatic wiring.
     */
    public createTensor(
        id: string,
        literalType: DataType,
        shape: KnownShape,
        type: TensorNode.TensorKind = "intermediate",
    ): TensorNode.Class {
        return this.addNode(
            uniq(this.graph, `${this.scopeTag}_${id}`),
            new TensorNode.Builder(literalType, shape, type),
        ).as(TensorNode);
    }

    /**
     * Rewires all downstream consumers of `oldNode` to point to `newNode`.
     * Automatically deletes `oldNode` if it becomes orphaned and is purely intermediate.
     */
    public replaceAllUsesWith(oldNode: ValueNode, newNode: ValueNode): void {
        let isGraphOutput = false;
        if (oldNode.is(TensorNode) && oldNode.as(TensorNode).type === "output") {
            isGraphOutput = true;
        }

        // 1. Recursive helper to update uses inside the current graph AND subgraphs
        const ensureReplacementInGraph = (g: OnnxGraph.Class): ValueNode => {
            if (g === this.graph) return newNode;
            const origName = newNode.is(RegionArgumentNode)
                ? newNode.as(RegionArgumentNode).originalName
                : newNode.id;
            if (g.hasNode(newNode.id)) {
                const existing = g.getNodeById(newNode.id)!;
                if (
                    existing.is(RegionArgumentNode) &&
                    existing.as(RegionArgumentNode).originalName === origName
                ) {
                    return existing.as(RegionArgumentNode);
                }
                if (newNode.is(ConstantNode) && existing.is(ConstantNode)) {
                    return existing.as(ConstantNode);
                }
            }
            if (newNode.is(ConstantNode)) {
                const c = newNode.as(ConstantNode);
                return new GraphBuilder(g, this.scopeTag).createConstant(
                    uniq(g, c.id),
                    c.constantValue,
                );
            }
            const argId = g.hasNode(newNode.id) ? uniq(g, `${newNode.id}_cap`) : newNode.id;
            return g
                .addNode(argId)
                .init(
                    new RegionArgumentNode.Builder(0, origName, newNode.literalType, newNode.shape),
                )
                .as(RegionArgumentNode);
        };

        const updateUsesInGraph = (g: OnnxGraph.Class) => {
            const matchesOld = (input: ValueNode): boolean =>
                input.id === oldNode.id ||
                (input.is(RegionArgumentNode) &&
                    input.as(RegionArgumentNode).originalName === oldNode.id);

            let replacementInG: ValueNode | undefined = g === this.graph ? newNode : undefined;

            const directSources = getDirectReplaceSources(g, this.graph, oldNode);

            const candidateOps = new Set<OperationNode.Class>();
            for (const src of directSources) {
                for (const op of src.outgoers.targets.filterIs(OperationNode)) {
                    candidateOps.add(op);
                }
            }

            for (const op of candidateOps) {
                const currentInputs = op.getInputs() ?? [];
                const matchedOldNodes = currentInputs.filter(matchesOld);

                if (matchedOldNodes.length > 0) {
                    if (!replacementInG) {
                        replacementInG = ensureReplacementInGraph(g);
                        subgraphMetaCache.delete(g);
                    }
                    const targetRep = replacementInG;
                    const updatedInputs = currentInputs.map((input) =>
                        matchesOld(input) ? targetRep : input,
                    );

                    op.setInputs(updatedInputs);

                    // Disconnect old edges in g and wire replacement node
                    for (const oldInp of matchedOldNodes) {
                        const existingEdge = g.getEdge(oldInp.id, op.id);
                        if (existingEdge) {
                            if (g === this.graph) this.removeEdge(existingEdge);
                            else existingEdge.remove();
                        }
                    }

                    if (!g.getEdge(targetRep.id, op.id)) {
                        if (g === this.graph) {
                            this.addEdge(targetRep, op, targetRep.literalType, targetRep.shape);
                        } else {
                            g.addEdge(targetRep, op).init(
                                new OnnxEdge.Builder(targetRep.literalType, targetRep.shape),
                            );
                        }
                    }
                }
            }

            // Recursively update control-flow subgraphs only when they can reference oldNode or contain deeper regions
            for (const op of g.getOperationNodes()) {
                for (const sub of op.regions) {
                    const subGraph = sub as OnnxGraph.Class;
                    const subMeta = getSubgraphMeta(subGraph);
                    if (
                        !subMeta.hasNestedRegions &&
                        !subMeta.hasRenamedArgs &&
                        !subGraph.hasNode(oldNode.id)
                    ) {
                        continue;
                    }
                    updateUsesInGraph(subGraph);
                }
            }

            // Clean up orphaned RegionArgumentNode in subgraphs after re-capturing to newNode
            if (g !== this.graph && replacementInG !== undefined) {
                for (const src of directSources) {
                    if (src.is(RegionArgumentNode) && src.outgoers.length === 0) {
                        src.remove();
                        subgraphMetaCache.delete(g);
                    }
                }
            }
        };

        // Start recursive replacement
        updateUsesInGraph(this.graph);

        // Disconnect any remaining outgoers manually
        oldNode.outgoers.toArray().forEach((edge) => this.removeEdge(edge));

        // 2. Clean up or redirect
        if (isGraphOutput) {
            // We cannot delete a graph output. We must route the new node into it!
            const producers = newNode.incomers.sources.filterIs(OperationNode).toArray();
            if (producers.length === 1 && newNode.outgoers.length === 0) {
                const producer = producers[0];
                newNode.incomers.forEach((e) => this.removeEdge(e));
                this.addEdge(producer, oldNode, oldNode.literalType, oldNode.shape);
                this.removeNode(newNode);
            } else {
                // Fallback to Identity
                const idOp = this.addNode(
                    uniq(this.graph, `${this.scopeTag}_Identity_to_output_${oldNode.id}`),
                    new OperationNode.Builder("Identity", [newNode]),
                ).as(OperationNode);

                this.addEdge(newNode, idOp, newNode.literalType, newNode.shape);
                this.addEdge(idOp, oldNode, oldNode.literalType, oldNode.shape);
            }
        } else {
            // Clean up orphaned intermediate node
            if (oldNode.outgoers.length === 0) {
                if (oldNode.is(TensorNode) && oldNode.as(TensorNode).type === "intermediate") {
                    this.removeNode(oldNode);
                } else if (oldNode.is(ConstantNode)) {
                    this.removeNode(oldNode);
                }
            }
        }
    }
}
