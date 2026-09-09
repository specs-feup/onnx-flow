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
import OnnxEdge from "./OnnxEdge.js";
import { uniq } from "./Utils.js";
import { OpRegistry } from "./Schema/OpRegistry.js";
import { inferNodeShape } from "./InferShapes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type BaseEdge from "@specs-feup/flow/graph/BaseEdge";

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
        node.remove();
    }

    /**
     * Centralized method to remove an edge.
     */
    public removeEdge(edge: OnnxEdge.Class | BaseEdge.Class): void {
        edge.remove();
    }

    /**
     * Creates an OperationNode, wires its inputs, and provisions its output tensors.
     */
    public createOp(
        type: string,
        inputs: ValueNode[],
        attributes: AttributeMap = {},
        expectedOutputs?: { type: DataType; shape: KnownShape }[],
        regions?: OnnxGraph.Class[],
    ): ConcreteValueNode[] {
        // 1. Build Operation Node
        const op = this.addNode(
            uniq(this.graph, `${this.scopeTag}_${type}`),
            new OperationNode.Builder(type, inputs, attributes, regions),
        ).as(OperationNode);

        // 2. Wire incoming edges using the resolved proxy nodes
        for (const input of inputs) {
            if (this.graph.hasNode(input.id)) {
                this.addEdge(input, op, input.literalType, input.shape);
            }
        }

        // 3. Provision Output Tensors
        // Look up the schema to know how many outputs to generate (default to 1)
        const schema = OpRegistry.getInstance().get(type, 19);
        let numOutputs = schema ? schema.outputs.length : 1;

        // Override output count for variadic ops like Split
        if (expectedOutputs !== undefined && expectedOutputs.length > numOutputs) {
            numOutputs = expectedOutputs.length;
        } else if ("num_outputs" in attributes) {
            numOutputs = attributes["num_outputs"] as number;
        }

        const outputs: ConcreteValueNode[] = [];
        for (let i = 0; i < numOutputs; i++) {
            // Basic inference for types/shapes to keep the graph valid before a full pass
            let inferredType = DataType.UNDEFINED;
            let inferredShape: KnownShape = [];

            if (expectedOutputs !== undefined && i in expectedOutputs) {
                inferredType = expectedOutputs[i].type;
                inferredShape = expectedOutputs[i].shape;
            } else if (inputs.length > 0) {
                inferredType = inputs[0].literalType;

                // Inherit bool for logical ops
                if (
                    [
                        "Equal",
                        "Greater",
                        "Less",
                        "And",
                        "Or",
                        "Not",
                        "GreaterOrEqual",
                        "LessOrEqual",
                    ].includes(type)
                ) {
                    inferredType = DataType.BOOL;
                } else if (["Shape", "Size"].includes(type)) {
                    inferredType = DataType.INT64;
                    inferredShape = [(inputs[0].shape.length as number | undefined) ?? 1];
                } else if (type === "Cast" && "to" in attributes) {
                    inferredType = attributes["to"] as DataType;
                }
            }

            const outTensor = this.addNode(
                uniq(this.graph, `${this.scopeTag}_${type}_out${i}`),
                new TensorNode.Builder(inferredType, inferredShape, "intermediate"),
            ).as(TensorNode);

            this.addEdge(op, outTensor, outTensor.literalType, outTensor.shape);
            outputs.push(outTensor);
        }

        // 4. Automatically infer shapes for the newly created operation
        inferNodeShape(op, this.graph);

        // 5. Explicitly specified expectedOutputs take ultimate precedence
        if (expectedOutputs) {
            for (let i = 0; i < expectedOutputs.length; i++) {
                if (i in outputs && expectedOutputs[i].shape.length > 0) {
                    outputs[i].setShape(expectedOutputs[i].shape);
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
     * Crucial for frontend parsing (initGraph) and deterministic pass generation.
     */
    public createOpWithExact(
        opId: string,
        type: string,
        inputs: ValueNode[],
        outputIds: string[],
        attributes: AttributeMap = {},
        expectedOutputs?: { type: DataType; shape: KnownShape }[],
        regions?: OnnxGraph.Class[],
    ): ConcreteValueNode[] {
        // 1. Build Operation Node with exact ID
        const op = this.addNode(
            opId,
            new OperationNode.Builder(type, inputs, attributes, regions),
        ).as(OperationNode);

        // 2. Wire incoming edges
        for (const input of inputs) {
            if (this.graph.hasNode(input.id)) {
                this.addEdge(input, op, input.literalType, input.shape);
            }
        }

        // 3. Provision Output Tensors using exact IDs
        const outputs: ConcreteValueNode[] = [];
        for (let i = 0; i < outputIds.length; i++) {
            let inferredType = DataType.UNDEFINED;
            let inferredShape: KnownShape = [];

            if (expectedOutputs !== undefined && i in expectedOutputs) {
                inferredType = expectedOutputs[i].type;
                inferredShape = expectedOutputs[i].shape;
            } else if (inputs.length > 0) {
                inferredType = inputs[0].literalType;

                // Inherit bool for logical ops
                if (
                    [
                        "Equal",
                        "Greater",
                        "Less",
                        "And",
                        "Or",
                        "Not",
                        "GreaterOrEqual",
                        "LessOrEqual",
                    ].includes(type)
                ) {
                    inferredType = DataType.BOOL;
                } else if (["Shape", "Size"].includes(type)) {
                    inferredType = DataType.INT64;
                    inferredShape = [(inputs[0].shape.length as number | undefined) ?? 1];
                } else if (type === "Cast" && "to" in attributes) {
                    inferredType = attributes["to"] as DataType;
                }
            }

            let outTensor: TensorNode.Class;
            if (this.graph.hasNode(outputIds[i])) {
                outTensor = this.graph.getNodeById(outputIds[i])!.as(TensorNode);
                if (inferredType !== DataType.UNDEFINED) outTensor.setLiteralType(inferredType);
                if (inferredShape.length > 0) outTensor.setShape(inferredShape);
            } else {
                // Create new node
                outTensor = this.addNode(
                    outputIds[i],
                    new TensorNode.Builder(inferredType, inferredShape, "intermediate"),
                ).as(TensorNode);
            }

            this.addEdge(op, outTensor, outTensor.literalType, outTensor.shape);
            outputs.push(outTensor);
        }

        // 4. Run shape inference
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
     * Rewires all downstream consumers of `oldNode` to point to `newNode`.
     * Automatically deletes `oldNode` if it becomes orphaned and is purely intermediate.
     */
    public replaceAllUsesWith(oldNode: ValueNode, newNode: ValueNode): void {
        let isGraphOutput = false;
        if (oldNode.is(TensorNode) && oldNode.as(TensorNode).type === "output") {
            isGraphOutput = true;
        }

        // 1. Recursive helper to update uses inside the current graph AND subgraphs
        const updateUsesInGraph = (g: OnnxGraph.Class) => {
            for (const op of g.getOperationNodes().toArray()) {
                const currentInputs = op.getInputs() ?? [];

                // Use .some() to cleanly check if we need to make a change
                const changed = currentInputs.some((input) => input.id === oldNode.id);

                if (changed) {
                    // Only run .map() if we know a change is happening
                    const updatedInputs = currentInputs.map((input) =>
                        input.id === oldNode.id ? newNode : input,
                    );

                    op.setInputs(updatedInputs);

                    // Disconnect old edge if it exists at this graph level
                    const existingEdge = g.getEdge(oldNode.id, op.id);
                    if (existingEdge) this.removeEdge(existingEdge);

                    // Connect new edge (only if newNode is visible/accessible in this scope)
                    if (g.hasNode(newNode.id) || g === this.graph) {
                        this.addEdge(newNode, op, newNode.literalType, newNode.shape);
                    }
                }

                // Recursively update control-flow subgraphs (Loop, If, Scan bodies)
                // Assuming OperationNode stores subgraphs in an accessible array:
                for (const sub of op.regions) {
                    updateUsesInGraph(sub as OnnxGraph.Class);
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
