import type OnnxGraph from "./OnnxGraph.js";
import type {
    AttributeMap,
    DataType,
    StaticShape,
    ValueNode,
    ConcreteValueNode,
} from "./OnnxTypes.js";
import OperationNode from "./OperationNode.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import OnnxEdge from "./OnnxEdge.js";
import { makeTensorProto, uniq } from "./Utils.js";
import { OpRegistry } from "./Schema/OpRegistry.js";

export class GraphBuilder {
    constructor(private graph: OnnxGraph.Class) {}

    /**
     * Creates an OperationNode, wires its inputs, and provisions its output tensors.
     */
    public createOp(
        type: string,
        inputs: ValueNode[],
        attributes: AttributeMap = {},
    ): ConcreteValueNode[] {
        // 1. Build Operation Node
        const op = this.graph
            .addNode(uniq(this.graph, type))
            .init(new OperationNode.Builder(type, inputs, attributes))
            .as(OperationNode);

        // 2. Wire incoming edges automatically
        for (const input of inputs) {
            this.graph
                .addEdge(input, op)
                .init(new OnnxEdge.Builder(input.literalType, input.shape))
                .as(OnnxEdge);
        }

        // 3. Provision Output Tensors
        // Look up the schema to know how many outputs to generate (default to 1)
        const schema = OpRegistry.getInstance().get(type, 19);
        const numOutputs = schema ? schema.outputs.length : 1;

        const outputs: ConcreteValueNode[] = [];
        for (let i = 0; i < numOutputs; i++) {
            // Note: We leave DataType as UNDEFINED and shape as [] for the InferShapes pass to resolve later
            const outTensor = this.graph
                .addNode(uniq(this.graph, `${type}_out${i}`))
                .init(new TensorNode.Builder(0, [], "intermediate")) // 0 = DataType.UNDEFINED
                .as(TensorNode);

            this.graph
                .addEdge(op, outTensor)
                .init(new OnnxEdge.Builder(outTensor.literalType, outTensor.shape))
                .as(OnnxEdge);
            outputs.push(outTensor);
        }

        return outputs;
    }

    /**
     * Creates a ConstantNode safely, ensuring unique IDs.
     */
    public createConstant(
        name: string,
        dtype: DataType,
        shape: StaticShape,
        data: number[],
    ): ConstantNode.Class {
        const tensorProto = makeTensorProto(dtype, shape, data);
        tensorProto.name = name;

        return this.graph
            .addNode(uniq(this.graph, name))
            .init(new ConstantNode.Builder(tensorProto))
            .as(ConstantNode);
    }

    /**
     * Rewires all downstream consumers of `oldNode` to point to `newNode`.
     * Automatically deletes `oldNode` if it becomes orphaned and is purely intermediate.
     */
    public replaceAllUsesWith(oldNode: ValueNode, newNode: ValueNode): void {
        const outgoers = oldNode.getOutgoers.toArray();
        for (const edge of outgoers) {
            const targetOp = edge.target;

            // Wire newNode to targetOp
            this.graph
                .addEdge(newNode, targetOp)
                .init(new OnnxEdge.Builder(newNode.literalType, newNode.shape))
                .as(OnnxEdge);

            // Update the targetOp's internal inputs array
            if (targetOp.is(OperationNode)) {
                const targetOpNode = targetOp.as(OperationNode);
                const currentInputs = targetOpNode.getInputs() ?? [];
                const updatedInputs = currentInputs.map((input) =>
                    input.id === oldNode.id ? newNode : input,
                );
                targetOpNode.setInputs(updatedInputs);
            }

            // Disconnect old edge
            edge.remove();
        }

        // Clean up orphaned node
        if (oldNode.getOutgoers.length === 0) {
            if (oldNode.is(TensorNode) && oldNode.as(TensorNode).type === "intermediate") {
                oldNode.remove();
            } else if (oldNode.is(ConstantNode)) {
                oldNode.remove();
            }
        }
    }
}
