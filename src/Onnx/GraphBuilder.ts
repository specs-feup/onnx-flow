import OnnxGraph from "./OnnxGraph.js";
import type {
    AttributeMap,
    StaticShape,
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
import { bool, int64Vec, scalarInt64, uniq, UNKOWN_SHAPE, zeroTensor } from "./Utils.js";
import { OpRegistry } from "./Schema/OpRegistry.js";
import Graph from "@specs-feup/flow/graph/Graph";
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
    ): ConcreteValueNode[] {
        // 1. Build Operation Node
        const op = this.graph
            .addNode(uniq(this.graph, `${this.scopeTag}_${type}`))
            .init(new OperationNode.Builder(type, inputs, attributes))
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
        const numOutputs = schema ? schema.outputs.length : 1;

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
                    inferredShape = [inputs[0].shape.length];
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
                if (
                    i in outputs &&
                    expectedOutputs[i].shape !== undefined &&
                    expectedOutputs[i].shape.length > 0
                ) {
                    outputs[i].setShape(expectedOutputs[i].shape);
                }
                if (i in outputs && expectedOutputs[i].type !== DataType.UNDEFINED) {
                    outputs[i].setLiteralType(expectedOutputs[i].type);
                }
            }
        }

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
                let changed = false;

                const updatedInputs = currentInputs.map((input) => {
                    if (input.id === oldNode.id) {
                        changed = true;
                        return newNode;
                    }
                    return input;
                });

                if (changed) {
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
                for (const sub of (op as any).subgraphs || []) {
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

    /**
     * Creates a fully wired ONNX Loop node and its internal subgraph.
     * Supports both static and dynamic trip counts and carry shapes.
     * * @param tripCount The maximum number of iterations (number or ValueNode).
     * @param elemTy The data type of the loop-carried state.
     * @param carryLen The shape/length of the carried state (StaticShape or ValueNode).
     */
    public createForLoopRegion(
        outerBuilder: GraphBuilder,
        totalIters: number | ValueNode,
        elemTy: DataType,
        carryLen: StaticShape | ValueNode,
        tag: string = "loop",
    ): {
        loopOp: OperationNode.Class;
        innerBuilder: GraphBuilder;
        trip: TensorNode.Class;
        condIn: TensorNode.Class;
        vInitial: TensorNode.Class;
        loopOutput: TensorNode.Class;
        finalize: (nextStates: ConcreteValueNode[]) => void;
    } {
        // 1. Resolve Outer Inputs
        // Trip Count (Input 0)
        const tripInp =
            typeof totalIters === "number"
                ? outerBuilder.createConstant(`trip_count_${tag}`, scalarInt64(totalIters))
                : totalIters;

        // Condition (Input 1) - Usually hardcoded to true for lowering standard ops
        const condInInp = outerBuilder.createConstant(`cond_${tag}`, bool(true));

        // Initial Carry (Input 2)
        let vInitialInp: ValueNode;
        let internalCarryShape: KnownShape;

        if (Array.isArray(carryLen)) {
            // Static path - ALWAYS flatten the carry state for loop lowering
            const flatLen = carryLen.reduce((a, b) => a * b, 1);
            vInitialInp = outerBuilder.createConstant(
                `init_carry_${tag}`,
                zeroTensor(elemTy, [flatLen]), // Create a 1D flattened zero tensor
            );
            internalCarryShape = [flatLen]; // The loop internally carries a 1D tensor
        } else {
            const zeroScalar = outerBuilder.createConstant(`zero_${tag}`, zeroTensor(elemTy, []));

            // ALWAYS flatten dynamic carry state to 1D to match loop semantics
            // 1. Calculate flat trip count = ReduceProd(carryLen)
            const axes0 = outerBuilder.createConstant(`ax0_fl_${tag}`, int64Vec([0]));
            const flatTrip = outerBuilder.createOp("ReduceProd", [carryLen, axes0], {
                keepdims: 0,
            })[0];
            const flatTrip1D = outerBuilder.createOp("Unsqueeze", [flatTrip, axes0])[0];

            // 2. Expand to 1D shape and enforce [-1] to prevent InferShapes propagation bugs
            [vInitialInp] = outerBuilder.createOp("Expand", [zeroScalar, flatTrip1D], {}, [
                { type: elemTy, shape: UNKOWN_SHAPE },
            ]);

            internalCarryShape = UNKOWN_SHAPE; // The loop internally carries a 1D tensor
        }

        // 2. The Loop Body (region/inner graph)
        const innerGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerBuilder = new GraphBuilder(innerGraph, tag);

        const trip = innerGraph
            .addNode(uniq(innerGraph, "iter"))
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const condIn = innerGraph
            .addNode(uniq(innerGraph, "cond_in"))
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const vInitial = innerGraph
            .addNode(uniq(innerGraph, "carry"))
            .init(new TensorNode.Builder(elemTy, internalCarryShape, "input"))
            .as(TensorNode);

        const loopInputs = [tripInp, condInInp, vInitialInp];

        // 3. Cond Passthrough
        const idCond = innerGraph
            .addNode(uniq(innerGraph, `id_cond_${tag}`))
            .init(new OperationNode.Builder("Identity", [condIn]))
            .as(OperationNode);
        const condOut = innerGraph
            .addNode(uniq(innerGraph, `cond_out_${tag}`))
            .init(new TensorNode.Builder(DataType.BOOL, [], "output"))
            .as(TensorNode);
        innerGraph
            .addEdge(condIn, idCond)
            .init(new OnnxEdge.Builder(condIn.literalType, condIn.shape));
        innerGraph
            .addEdge(idCond, condOut)
            .init(new OnnxEdge.Builder(condOut.literalType, condOut.shape));

        // 4. The Finalizer (Wires the inner results to the loop boundary)
        const finalize = (nextStates: ConcreteValueNode[]) => {
            nextStates.forEach((state, idx) => {
                const expectedType = idx === 0 ? vInitial.literalType : state.literalType;
                const expectedShape = idx === 0 ? vInitial.shape : state.shape;

                const stateOut = innerGraph
                    .addNode(uniq(innerGraph, `${this.scopeTag}_${idx}_state_out_${tag}`))
                    .init(new TensorNode.Builder(expectedType, expectedShape, "output"))
                    .as(TensorNode);

                const idOp = innerGraph
                    .addNode(uniq(innerGraph, `${this.scopeTag}_id_state_out_${tag}_${idx}`))
                    .init(new OperationNode.Builder("Identity", [state]))
                    .as(OperationNode);

                innerGraph
                    .addEdge(state, idOp)
                    .init(new OnnxEdge.Builder(state.literalType, state.shape));
                innerGraph
                    .addEdge(idOp, stateOut)
                    .init(new OnnxEdge.Builder(expectedType, expectedShape));
            });
        };

        // 5. The Loop Node
        const loopOp = this.graph
            .addNode(uniq(this.graph, tag))
            .init(new OperationNode.Builder("Loop", loopInputs, {}, [innerGraph]))
            .as(OperationNode);

        // 6. Wire Outer Graph Edges
        loopInputs.forEach((input) => {
            if (this.graph.hasNode(input.id))
                this.graph
                    .addEdge(input, loopOp)
                    .init(new OnnxEdge.Builder(input.literalType, input.shape))
                    .as(OnnxEdge);
        });

        // 7. Generate Outputs for the Outer Graph
        const loopOutput = this.graph
            .addNode(uniq(this.graph, `${this.scopeTag}_carry_out`))
            .init(new TensorNode.Builder(vInitial.literalType, vInitial.shape, "intermediate"))
            .as(TensorNode);
        this.graph
            .addEdge(loopOp, loopOutput)
            .init(new OnnxEdge.Builder(vInitial.literalType, vInitial.shape))
            .as(OnnxEdge);

        return { loopOp, innerBuilder, trip, condIn, vInitial, loopOutput, finalize };
    }
}
