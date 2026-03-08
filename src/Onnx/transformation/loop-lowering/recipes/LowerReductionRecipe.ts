import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import type { ConcreteValueNode, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntsAttr,
    makeTensorProto,
    toStaticShape,
    readConstIntegerVectorFromTensorNode,
} from "../../../Utils.js";
import ConstantNode from "../../../ConstantNode.js";

export class LowerReductionRecipe implements DecompositionRecipe {
    public readonly name = "LowerReduction";
    public readonly targetOp = OpCategory.Reduction;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = ["Loop", "Gather", "Add", "Where", "ScatterElements"];

    canApply(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.Reduction) return false;
        return (op.getInputs() ?? []).some((input) => input.shape.length > 0);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs() as ConcreteValueNode[];
        const input = inputs[0];
        const output = op.getOutputs()[0];
        const inShape = toStaticShape(input.shape);
        let outShape = toStaticShape(output.shape);
        const rank = inShape.length;

        let dtype = (output.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED)
            dtype = (input.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED) dtype = DataType.FLOAT;

        // Parse attributes and inputs for axes
        const keepdims = getIntsAttr(op, "keepdims", [1])[0] !== 0;
        let axes: number[] = [];

        if (inputs.length > 1) {
            if (inputs[1].is(ConstantNode)) {
                axes = readConstIntegerVectorFromTensorNode(inputs[1]) ?? [];
            } else {
                console.warn(
                    `[LowerReduction] Dynamic axes not supported, defaulting to all axes.`,
                );
            }
        } else {
            axes = getIntsAttr(op, "axes", []);
        }

        if (axes.length === 0) {
            axes = Array.from({ length: rank }, (_, i) => i);
        } else {
            axes = axes.map((a) => (a < 0 ? a + rank : a));
        }

        const accShape = inShape.slice();
        for (const ax of axes) {
            accShape[ax] = 1;
        }

        if (outShape.length === 0) {
            outShape = keepdims ? accShape : inShape.filter((_, i) => !axes.includes(i));
        }

        const totalElements = inShape.reduce((a, b) => a * b, 1);
        const carryLen = accShape.reduce((a, b) => a * b, 1);

        // Ensure we always pass at least [1] for the carry state to avoid scalar reshape bugs
        const carryShape = [carryLen];

        const { innerBuilder, trip, vInitial, loopOutput, finalize } = builder.createForLoopRegion(
            builder,
            totalElements,
            dtype,
            carryShape,
            `ReduceLoop_${op.id}`,
        );

        const flatAxes = innerBuilder.createConstant(
            `axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );

        // 1. Get input value
        const flatShape = innerBuilder.createConstant(
            `flatShape_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [-1]),
        );
        const flatInput = innerBuilder.createOp("Reshape", [input, flatShape])[0];
        const iterUnsq = innerBuilder.createOp("Unsqueeze", [trip, flatAxes])[0];
        const gathered = innerBuilder.createOp("Gather", [flatInput, iterUnsq], { axis: 0 })[0];
        const val = innerBuilder.createOp("Squeeze", [gathered, flatAxes])[0];

        // 2. Decode `iter` into input coordinates
        let rem: ValueNode = trip;
        const inCoords: ConcreteValueNode[] = [];
        for (let i = rank - 1; i >= 0; i--) {
            const dimConst = innerBuilder.createConstant(
                `dim_${op.id}_${i}`,
                makeTensorProto(DataType.INT64, [], [inShape[i]]),
            );
            const coord = innerBuilder.createOp("Mod", [rem, dimConst])[0];
            inCoords.unshift(coord);
            rem = innerBuilder.createOp("Div", [rem, dimConst])[0];
        }

        // 3. Map input coordinates to output coordinates (reduced axes mapped to 0)
        const outCoords: ConcreteValueNode[] = [];
        const zeroConst = innerBuilder.createConstant(
            `zero_${op.id}`,
            makeTensorProto(DataType.INT64, [], [0]),
        );
        for (let i = 0; i < rank; i++) {
            if (axes.includes(i)) {
                outCoords.push(zeroConst);
            } else {
                outCoords.push(inCoords[i]);
            }
        }

        // 4. Flatten output coordinates into a 1D carry index
        let flatOutIdx: ConcreteValueNode = zeroConst;
        let stride = 1;
        for (let i = rank - 1; i >= 0; i--) {
            if (stride > 1) {
                const strideConst = innerBuilder.createConstant(
                    `stride_${op.id}_${i}`,
                    makeTensorProto(DataType.INT64, [], [stride]),
                );
                const offset = innerBuilder.createOp("Mul", [outCoords[i], strideConst])[0];
                flatOutIdx = innerBuilder.createOp("Add", [flatOutIdx, offset])[0];
            } else {
                flatOutIdx = innerBuilder.createOp("Add", [flatOutIdx, outCoords[i]])[0];
            }
            stride *= accShape[i];
        }

        // 5. Gather current accumulated value from flat carry
        const outIdxUnsq = innerBuilder.createOp("Unsqueeze", [flatOutIdx, flatAxes])[0];
        const currentAccGathered = innerBuilder.createOp("Gather", [vInitial, outIdxUnsq], {
            axis: 0,
        })[0];
        const currentAcc = innerBuilder.createOp("Squeeze", [currentAccGathered, flatAxes])[0];

        // 6. Perform Math & First Hit setup
        let nextCarryMath: ConcreteValueNode;
        let firstVal: ConcreteValueNode = val;

        if (op.type === "ReduceMax") {
            nextCarryMath = innerBuilder.createOp("Max", [currentAcc, val])[0];
        } else if (op.type === "ReduceMin") {
            nextCarryMath = innerBuilder.createOp("Min", [currentAcc, val])[0];
        } else if (op.type === "ReduceProd") {
            nextCarryMath = innerBuilder.createOp("Mul", [currentAcc, val])[0];
        } else if (
            op.type === "ReduceSum" ||
            op.type === "ReduceMean" ||
            op.type === "ReduceLogSum"
        ) {
            nextCarryMath = innerBuilder.createOp("Add", [currentAcc, val])[0];
            firstVal = val;
        } else if (op.type === "ReduceSumSquare" || op.type === "ReduceL2") {
            const sq = innerBuilder.createOp("Mul", [val, val])[0];
            nextCarryMath = innerBuilder.createOp("Add", [currentAcc, sq])[0];
            firstVal = sq;
        } else if (op.type === "ReduceL1") {
            const absVal = innerBuilder.createOp("Abs", [val])[0];
            nextCarryMath = innerBuilder.createOp("Add", [currentAcc, absVal])[0];
            firstVal = absVal;
        } else if (op.type === "ReduceLogSumExp") {
            const expVal = innerBuilder.createOp("Exp", [val])[0];
            nextCarryMath = innerBuilder.createOp("Add", [currentAcc, expVal])[0];
            firstVal = expVal;
        } else {
            nextCarryMath = innerBuilder.createOp("Add", [currentAcc, val])[0];
        }

        // 7. Avoid multiplying/min'ing with initial zero on the first hit for each bin
        let isFirstHit: ValueNode | undefined = undefined;
        for (const ax of axes) {
            const isZero = innerBuilder.createOp("Equal", [inCoords[ax], zeroConst])[0];
            if (!isFirstHit) isFirstHit = isZero;
            else isFirstHit = innerBuilder.createOp("And", [isFirstHit, isZero])[0];
        }

        let nextCarryElement = nextCarryMath;
        if (isFirstHit) {
            nextCarryElement = innerBuilder.createOp("Where", [
                isFirstHit,
                firstVal,
                nextCarryMath,
            ])[0];
        }

        // 8. Scatter updated scalar into flat state
        const updateVal = innerBuilder.createOp("Unsqueeze", [nextCarryElement, flatAxes])[0];
        const nextCarry = innerBuilder.createOp(
            "ScatterElements",
            [vInitial, outIdxUnsq, updateVal],
            { axis: 0 },
        )[0];

        finalize([nextCarry]);

        // 9. Reshape the 1D loop output buffer back to requested outShape
        const outShapeConst = builder.createConstant(
            `outShape_${op.id}`,
            makeTensorProto(DataType.INT64, [outShape.length], outShape),
        );
        const finalShapeAttr = outShape.length > 0 ? outShape : [1];
        let processedOutput = builder.createOp("Reshape", [loopOutput, outShapeConst], {}, [
            { type: dtype, shape: finalShapeAttr },
        ])[0];

        // Ensure purely scalar outputs are appropriately formatted
        if (outShape.length === 0) {
            const sqAxes = builder.createConstant(
                `sqAxes_${op.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            processedOutput = builder.createOp("Squeeze", [processedOutput, sqAxes], {}, [
                { type: dtype, shape: [] },
            ])[0];
        }

        // 10. Post-processing for Mean, L2, LogSum, LogSumExp
        if (op.type === "ReduceMean") {
            const count = totalElements / carryLen;
            const countConst = builder.createConstant(
                `count_${op.id}`,
                makeTensorProto(dtype, [], [count]),
            );
            processedOutput = builder.createOp("Div", [processedOutput, countConst])[0];
        } else if (op.type === "ReduceL2") {
            processedOutput = builder.createOp("Sqrt", [processedOutput])[0];
        } else if (op.type === "ReduceLogSum" || op.type === "ReduceLogSumExp") {
            processedOutput = builder.createOp("Log", [processedOutput])[0];
        }

        builder.replaceAllUsesWith(output, processedOutput);
        op.remove();
    }
}
