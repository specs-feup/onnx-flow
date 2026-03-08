import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asStaticDims,
    getIntsAttr,
    readConstIntegerVectorFromTensorNode,
    scalarInt64,
    computeStrides,
    int64Vec,
} from "../../../Utils.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import {
    resolveRecipeInput,
    squeezeIfLen1,
    ensureFlatInput,
    decodeMixedRadix,
    buildLinearIndex,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import ConstantNode from "../../../ConstantNode.js";

export class LowerReductionRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        return schema?.category === OpCategory.Reduction;
    }

    getLoopBounds(op: OperationNode.Class, outShape: KnownShape) {
        const inputs = op.getInputs()!;
        const inShape = asStaticDims(inputs[0].shape);
        const rank = inShape.length;

        let axes: number[] = [];
        if (inputs.length > 1 && inputs[1].is(ConstantNode)) {
            axes = readConstIntegerVectorFromTensorNode(inputs[1].as(ConstantNode)) ?? [];
        } else {
            axes = getIntsAttr(op, "axes", []);
        }

        if (axes.length === 0) axes = Array.from({ length: rank }, (_, i) => i);
        else axes = axes.map((a) => (a < 0 ? a + rank : a));

        const accShape = inShape.slice();
        for (const ax of axes) accShape[ax] = 1;

        const totalIters = inShape.reduce((a, b) => a * b, 1);
        const carryLen = accShape.reduce((a, b) => a * b, 1);

        // Flatten carry to 1D for ScatterElements compatibility in ORT
        return { totalIters, carryShape: [carryLen] };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `red_${op.id}`);
        const inputs = op.getInputs()! as ConcreteValueNode[];
        const inShape = asStaticDims(inputs[0].shape);
        const dtype = (op.getOutputs()[0].literalType as DataType) ?? DataType.FLOAT;

        // 1. Resolve and Gather input element
        const tInner = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            true,
            false,
        );
        const flatInput = ensureFlatInput(builder, tInner);
        const [iterUnsq] = builder.createOp("Unsqueeze", [iter, axes]);
        const [gatheredOut] = builder.createOp("Gather", [flatInput, iterUnsq], { axis: 0 });
        let val = squeezeIfLen1(builder, gatheredOut, axes, `sq_in`) as ConcreteValueNode;

        // 2. Pre-processing for specific reductions
        if (op.type === "ReduceLogSumExp") [val] = builder.createOp("Exp", [val]);
        else if (op.type === "ReduceSumSquare" || op.type === "ReduceL2")
            [val] = builder.createOp("Mul", [val, val]);
        else if (op.type === "ReduceL1") [val] = builder.createOp("Abs", [val]);

        // 3. Coordinate decoding for indexing
        const inCoords = decodeMixedRadix(builder, iter, inShape, `coords`);

        let axesList: number[] = [];
        if (inputs.length > 1 && inputs[1].is(ConstantNode)) {
            axesList = readConstIntegerVectorFromTensorNode(inputs[1].as(ConstantNode)) ?? [];
        } else {
            axesList = getIntsAttr(op, "axes", []);
        }
        if (axesList.length === 0) {
            axesList = Array.from({ length: inShape.length }, (_, i) => i);
        } else {
            axesList = axesList.map((a) => (a < 0 ? a + inShape.length : a));
        }

        const accShapeRaw = inShape.slice();
        for (const ax of axesList) accShapeRaw[ax] = 1;

        const zeroConst = builder.createConstant(`zero`, scalarInt64(0));
        const outCoords = inCoords.map((c, i) => (axesList.includes(i) ? zeroConst : c));
        const flatOutIdx = buildLinearIndex(
            builder,
            outCoords,
            computeStrides(accShapeRaw),
            `out_idx`,
        );
        const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        // 4. Accumulate
        const [gAccOut] = builder.createOp("Gather", [carryNode, flatOutIdxUnsq], { axis: 0 });
        const currentAcc = squeezeIfLen1(builder, gAccOut, axes, `sq_acc`) as ConcreteValueNode;

        let mathOpType = "Add";
        if (op.type === "ReduceMax") mathOpType = "Max";
        else if (op.type === "ReduceMin") mathOpType = "Min";
        else if (op.type === "ReduceProd") mathOpType = "Mul";

        const [mathOut] = builder.createOp(mathOpType, [currentAcc, val]);

        // 5. First-hit logic (Initializes slots with the first element they encounter)
        let isFirstHit: ConcreteValueNode | undefined;
        for (const ax of axesList) {
            const [isZero] = builder.createOp("Equal", [inCoords[ax], zeroConst]);
            isFirstHit = isFirstHit ? builder.createOp("And", [isFirstHit, isZero])[0] : isZero;
        }

        const [finalValue] = builder.createOp("Where", [isFirstHit!, val, mathOut]);
        const [updateVal] = builder.createOp("Unsqueeze", [finalValue, axes]);
        const [nextCarry] = builder.createOp(
            "ScatterElements",
            [carryNode, flatOutIdxUnsq, updateVal],
            { axis: 0 },
        );

        return { resultNode: finalValue, nextCarry };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        if (op.type === "ReduceL2") return builder.createOp("Sqrt", [loopOut])[0];
        if (op.type === "ReduceLogSum" || op.type === "ReduceLogSumExp")
            return builder.createOp("Log", [loopOut])[0];

        if (op.type === "ReduceMean") {
            const bounds = this.getLoopBounds(op, []);
            const count = bounds.totalIters / (bounds.carryShape[0] as number);
            const countConst = builder.createConstant(`mean_count`, {
                dataType: loopOut.literalType || DataType.FLOAT,
                dims: [],
                floatData: [count],
                int64Data: [BigInt(count)],
            });
            return builder.createOp("Div", [loopOut, countConst])[0];
        }
        return loopOut;
    }
}
