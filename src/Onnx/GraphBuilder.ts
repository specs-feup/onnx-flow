import type OnnxGraph from "./OnnxGraph.js";
import type {
    AttributeMap,
    ValueNode,
    ConcreteValueNode,
    TensorProto,
    KnownShape,
} from "./OnnxTypes.js";
import { DataType } from "./OnnxTypes.js";
import OperationNode from "./OperationNode.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import OnnxEdge from "./OnnxEdge.js";
import { uniq } from "./Utils.js";
import { OpRegistry } from "./Schema/OpRegistry.js";
import { inferNodeShape } from "./InferShapes.js";

export class GraphBuilder {
    constructor(
        public graph: OnnxGraph.Class,
        private scopeTag: string = "",
    ) {}

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
        const op = this.graph
            .addNode(uniq(this.graph, `${this.scopeTag}_${type}`))
            .init(new OperationNode.Builder(type, inputs, attributes, regions))
            .as(OperationNode);

        // 2. Wire incoming edges using the resolved proxy nodes
        for (const input of inputs) {
            if (this.graph.hasNode(input.id)) {
                this.graph
                    .addEdge(input, op)
                    .init(new OnnxEdge.Builder(input.literalType, input.shape))
                    .as(OnnxEdge);
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

            const outTensor = this.graph
                .addNode(uniq(this.graph, `${this.scopeTag}_${type}_out${i}`))
                .init(new TensorNode.Builder(inferredType, inferredShape, "intermediate"))
                .as(TensorNode);

            this.graph
                .addEdge(op, outTensor)
                .init(new OnnxEdge.Builder(outTensor.literalType, outTensor.shape))
                .as(OnnxEdge);
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
                        edge.remove();
                        this.graph
                            .addEdge(op, outputs[i])
                            .init(new OnnxEdge.Builder(outputs[i].literalType, outputs[i].shape))
                            .as(OnnxEdge);
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
        const op = this.graph
            .addNode(opId)
            .init(new OperationNode.Builder(type, inputs, attributes, regions))
            .as(OperationNode);

        // 2. Wire incoming edges
        for (const input of inputs) {
            if (this.graph.hasNode(input.id)) {
                this.graph
                    .addEdge(input, op)
                    .init(new OnnxEdge.Builder(input.literalType, input.shape))
                    .as(OnnxEdge);
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
                outTensor = this.graph
                    .addNode(outputIds[i])
                    .init(new TensorNode.Builder(inferredType, inferredShape, "intermediate"))
                    .as(TensorNode);
            }

            this.graph
                .addEdge(op, outTensor)
                .init(new OnnxEdge.Builder(outTensor.literalType, outTensor.shape))
                .as(OnnxEdge);
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
        return this.graph
            .addNode(uniq(this.graph, `${this.scopeTag}_${id}`))
            .init(new ConstantNode.Builder(proto))
            .as(ConstantNode);
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
                    if (existingEdge) existingEdge.remove();

                    // Connect new edge (only if newNode is visible/accessible in this scope)
                    if (g.hasNode(newNode.id) || g === this.graph) {
                        g.addEdge(newNode, op)
                            .init(new OnnxEdge.Builder(newNode.literalType, newNode.shape))
                            .as(OnnxEdge);
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
        oldNode.outgoers.toArray().forEach((edge) => edge.remove());

        // 2. Clean up or redirect
        if (isGraphOutput) {
            // We cannot delete a graph output. We must route the new node into it!
            const producers = newNode.incomers.sources.filterIs(OperationNode).toArray();
            if (producers.length === 1 && newNode.outgoers.length === 0) {
                const producer = producers[0];
                newNode.incomers.forEach((e) => e.remove());
                this.graph
                    .addEdge(producer, oldNode)
                    .init(new OnnxEdge.Builder(oldNode.literalType, oldNode.shape))
                    .as(OnnxEdge);
                newNode.remove();
            } else {
                // Fallback to Identity
                const idOp = this.graph
                    .addNode(uniq(this.graph, `${this.scopeTag}_Identity_to_output_${oldNode.id}`))
                    .init(new OperationNode.Builder("Identity", [newNode]))
                    .as(OperationNode);

                this.graph
                    .addEdge(newNode, idOp)
                    .init(new OnnxEdge.Builder(newNode.literalType, newNode.shape))
                    .as(OnnxEdge);
                this.graph
                    .addEdge(idOp, oldNode)
                    .init(new OnnxEdge.Builder(oldNode.literalType, oldNode.shape))
                    .as(OnnxEdge);
            }
        } else {
            // Clean up orphaned intermediate node
            if (oldNode.outgoers.length === 0) {
                if (oldNode.is(TensorNode) && oldNode.as(TensorNode).type === "intermediate") {
                    oldNode.remove();
                } else if (oldNode.is(ConstantNode)) {
                    oldNode.remove();
                }
            }
        }
    }
}
