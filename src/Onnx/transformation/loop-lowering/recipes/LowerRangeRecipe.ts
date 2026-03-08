import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, toStaticShape } from "../../../Utils.js";
import OnnxEdge from "../../../OnnxEdge.js";
import RegionArgumentNode from "../../../RegionArgumentNode.js";
import type OnnxGraph from "../../../OnnxGraph.js";

export class LowerRangeRecipe implements DecompositionRecipe {
    public readonly name = "LowerRange";
    public readonly targetOp = "Range";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Mul",
        "Add",
        "ScatterElements",
        "Sub",
        "Div",
        "Ceil",
        "Max",
        "Cast",
        "Expand",
        "Unsqueeze",
        "Squeeze",
    ];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Range";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs() as ConcreteValueNode[];
        const start = inputs[0];
        const limit = inputs[1];
        const delta = inputs[2];

        const output = op.getOutputs()[0];

        let dtype = (output.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED) {
            dtype = (start.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        }
        if (dtype === DataType.UNDEFINED) {
            dtype = DataType.FLOAT;
        }

        // Check if InferShapes was able to determine a static shape for this Range op
        const outShape = toStaticShape(output.shape);
        const isStatic = outShape.length === 1 && outShape[0] > 0;
        const tripCountVal = isStatic ? outShape[0] : -1;
        const shapeArr = isStatic ? [tripCountVal] : [-1];

        // Ensure start, limit, and delta are 0D scalars to prevent [1, 1] unsqueeze issues
        const forceScalar = (node: ConcreteValueNode, name: string) => {
            if (Array.isArray(node.shape) && node.shape.length === 0) return node;
            const axes = builder.createConstant(
                `sq_axes_${name}_${op.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            return builder.createOp("Squeeze", [node, axes], {}, [
                { type: node.literalType as DataType, shape: [] },
            ])[0];
        };

        const startS = forceScalar(start, "start");
        const limitS = forceScalar(limit, "limit");
        const deltaS = forceScalar(delta, "delta");

        // ==========================================
        // 1. Compute dynamic trip count in the graph (used for the Expand/Loop node inputs)
        // trip_count = max(ceil((limit - start) / delta), 0)
        // ==========================================
        const subLimStart = builder.createOp("Sub", [limitS, startS])[0];
        const subF = builder.createOp("Cast", [subLimStart], { to: DataType.FLOAT })[0];
        const deltaF = builder.createOp("Cast", [deltaS], { to: DataType.FLOAT })[0];
        const divF = builder.createOp("Div", [subF, deltaF])[0];
        const ceilF = builder.createOp("Ceil", [divF])[0];

        const zeroF = builder.createConstant(
            `range_zeroF_${op.id}`,
            makeTensorProto(DataType.FLOAT, [], [0]),
        );
        const maxF = builder.createOp("Max", [ceilF, zeroF])[0];
        const tripCountScalar = builder.createOp("Cast", [maxF], { to: DataType.INT64 }, [
            { type: DataType.INT64, shape: [] },
        ])[0];

        // ==========================================
        // 2. Initialize Carry State using Expand
        // ==========================================
        const axes0 = builder.createConstant(
            `axes0_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const tripCount1D = builder.createOp("Unsqueeze", [tripCountScalar, axes0], {}, [
            { type: DataType.INT64, shape: [1] },
        ])[0];

        const zeroVal = builder.createConstant(
            `range_zeroVal_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );

        // Expand(0, [trip_count]) -> array of zeros of size trip_count
        const vInitial = builder.createOp("Expand", [zeroVal, tripCount1D])[0];

        // Pass the statically determined shape (or [-1]) instead of hardcoding [-1]
        vInitial.setShape(shapeArr);
        vInitial.setLiteralType(dtype);

        // ==========================================
        // 3. Generate Loop
        // ==========================================
        const { loopOp, innerBuilder, trip, loopOutput, finalize } = builder.createForLoopRegion(
            builder,
            tripCountVal, // Use static trip count if available
            dtype,
            shapeArr, // Use static shape if available
            `RangeLoop_${op.id}`,
        );

        // Access the underlying outer graph to safely manipulate edges
        const outerGraph = builder.graph as OnnxGraph.Class;
        const oldInputs = loopOp.getInputs()!;

        loopOp.incomers.toArray().forEach((e) => {
            if (oldInputs.some((inp) => inp.id === e.source.id)) {
                const tensorNode = e.source;
                e.remove();

                // Remove the dangling Constant Nodes to resolve ORT warnings
                if (tensorNode.outgoers.length === 0) {
                    const producerEdge = tensorNode.incomers.first();
                    tensorNode.remove();
                    if (producerEdge) {
                        const producer = producerEdge.source;
                        if (producer.outgoers.length === 0) producer.remove();
                    }
                }
            }
        });

        const trueCond = builder.createConstant(
            `cond_true_${op.id}`,
            makeTensorProto(DataType.BOOL, [], [1]),
        );

        loopOp.setInputs([tripCountScalar, trueCond, vInitial]);

        outerGraph
            .addEdge(tripCountScalar, loopOp)
            .init(new OnnxEdge.Builder(tripCountScalar.literalType, tripCountScalar.shape))
            .as(OnnxEdge);
        outerGraph
            .addEdge(trueCond, loopOp)
            .init(new OnnxEdge.Builder(trueCond.literalType, trueCond.shape))
            .as(OnnxEdge);
        outerGraph
            .addEdge(vInitial, loopOp)
            .init(new OnnxEdge.Builder(vInitial.literalType, vInitial.shape))
            .as(OnnxEdge);

        // ==========================================
        // 4. INSIDE LOOP: Calculate y = start + (iter * delta)
        // ==========================================
        const innerGraph = innerBuilder.graph as OnnxGraph.Class;

        const captureNode = (outerNode: ConcreteValueNode) => {
            return innerGraph
                .addNode(outerNode.id)
                .init(
                    new RegionArgumentNode.Builder(
                        0,
                        outerNode.id,
                        outerNode.literalType,
                        outerNode.shape,
                    ),
                )
                .as(RegionArgumentNode);
        };

        const innerStart = captureNode(startS);
        const innerDelta = captureNode(deltaS);

        const flatAxes = innerBuilder.createConstant(
            `axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const iterCast = innerBuilder.createOp("Cast", [trip], { to: dtype })[0];

        const iterStep = innerBuilder.createOp("Mul", [iterCast, innerDelta])[0];
        const currentVal = innerBuilder.createOp("Add", [innerStart, iterStep])[0];

        const iterIdx = innerBuilder.createOp("Unsqueeze", [trip, flatAxes])[0];
        const updateVal = innerBuilder.createOp("Unsqueeze", [currentVal, flatAxes])[0];

        const innerCarry = innerGraph
            .getInputTensorNodes()
            .toArray()
            .find((n) => n.id.includes("carry"))!;
        const nextCarry = innerBuilder.createOp(
            "ScatterElements",
            [innerCarry, iterIdx, updateVal],
            { axis: 0 },
        )[0];

        finalize([nextCarry]);

        builder.replaceAllUsesWith(output, loopOutput);
        op.remove();
    }
}
