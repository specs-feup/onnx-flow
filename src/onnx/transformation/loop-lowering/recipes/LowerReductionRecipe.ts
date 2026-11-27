import type OnnxGraph from "../../../ir/OnnxGraph.js";
import type OperationNode from "../../../ir/OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import {
    asStaticDims,
    getIntAttr,
    getIntsAttr,
    readConstIntegerVectorFromTensorNode,
    scalarInt64,
    computeStrides,
    int64Vec,
    makeTensorProto,
    toStaticShape,
} from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import { OpRegistry } from "../../../schema/OpRegistry.js";
import { OpCategory } from "../../../schema/OpSchema.js";
import {
    resolveRecipeInput,
    ensureFlatInput,
    gatherScalar,
    decodeMixedRadix,
    buildLinearIndex,
    buildAguPointer,
    getOrCreateIterationSpaceCounters,
    extractMixedDims,
    packShapeVector,
    productMixed,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../ir/GraphBuilder.js";
import ConstantNode from "../../../ir/ConstantNode.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerReductionRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerReduction";
    public readonly targetOp = OpCategory.Reduction;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["reduction", "self-carry"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Size",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "And",
        "Where",
        "Max",
        "Min",
        "Abs",
        "Exp",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type === "ArgMax" || op.type === "ArgMin" || op.type === "TopK") return false;
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.Reduction) return false;
        const inputs = op.getInputs();
        if (!inputs || inputs.length === 0) return false;
        const dims = toStaticShape(inputs[0].shape);
        if (dims.length === 0 || dims.includes(-1)) return false;
        if (getIntAttr(op, "noop_with_empty_axes", 0) === 1) {
            const axesInput = inputs.length > 1 && inputs[1].id !== "" ? inputs[1] : undefined;
            const constAxes =
                axesInput?.is(ConstantNode) === true
                    ? (readConstIntegerVectorFromTensorNode(axesInput.as(ConstantNode)) ?? [])
                    : getIntsAttr(op, "axes", []);
            if (constAxes.length === 0 && (!axesInput || axesInput.is(ConstantNode))) {
                return false;
            }
        }
        return true;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: number[] | ConcreteValueNode;
    } {
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

        // --- DYNAMIC SUPPORT ---
        if (inputs[0].shape.includes(-1) || rank === 0) {
            const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `bounds_red_${op.id}`);
            const inputNode = inputs[0];

            if (rank === 0) {
                return { totalIters: 1, carryShape: [1], targetShape: [] };
            }

            const axesConst0 = builder.createConstant(`axes_fallback_${op.id}_0`, int64Vec([0]));
            const inDims = extractMixedDims(
                builder,
                inputNode,
                toStaticShape(inputNode.shape),
                `rb_${op.id}`,
            );
            const totalIters = productMixed(builder, inDims, `rb_tot_${op.id}`);
            const keepdims = getIntAttr(op, "keepdims", 1);
            const axesSet = new Set(axes);
            const outDims =
                keepdims === 1
                    ? inDims.map((d, idx) => (axesSet.has(idx) ? 1 : d))
                    : inDims.filter((_, idx) => !axesSet.has(idx));

            if (outDims.length === 0) {
                const emptyTsh = builder.createConstant(`rb_empty_${op.id}`, int64Vec([]));
                return { totalIters, carryShape: [1], targetShape: emptyTsh };
            }

            const targetShapeNode = packShapeVector(builder, outDims, `rb_tsh_${op.id}`);
            const carryLen = productMixed(builder, outDims, `rb_cl_${op.id}`);
            const [carryShape] = builder.createOp("Unsqueeze", [carryLen, axesConst0], {}, [
                { type: DataType.INT64, shape: [1] },
            ]);

            return { totalIters, carryShape, targetShape: targetShapeNode };
        }

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
        targetShapeNode: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `red_${op.id}`);
        const inputs = op.getInputs()! as ConcreteValueNode[];
        const inShape = asStaticDims(inputs[0].shape);

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
            targetShapeNode,
            loopCtx,
        );
        const flatInput = ensureFlatInput(builder, tInner);

        let readIdx: ConcreteValueNode = iter;
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            readIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                inShape,
                computeStrides(inShape),
                `red_in_${op.id}`,
            );
        }

        let val = gatherScalar(builder, flatInput, readIdx);

        // 2. Pre-processing for specific reductions
        if (op.type === "ReduceLogSumExp") [val] = builder.createOp("Exp", [val]);
        else if (op.type === "ReduceSumSquare" || op.type === "ReduceL2")
            [val] = builder.createOp("Mul", [val, val]);
        else if (op.type === "ReduceL1") [val] = builder.createOp("Abs", [val]);

        // 3. Determine reduced axes
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
        const accStrides = computeStrides(accShapeRaw);
        const zeroConst = builder.createConstant(`zero`, scalarInt64(0));

        let inCoords: ConcreteValueNode[];
        let flatOutIdx: ConcreteValueNode;

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                inShape,
                `red_space_${op.id}`,
            );
            inCoords = counters;
            // Reduced axes have stride 0 in the output accumulator
            const outAguStrides = accStrides.map((s, i) => (axesList.includes(i) ? 0 : s));
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                inShape,
                outAguStrides,
                `red_out_${op.id}`,
            );
        } else {
            inCoords = decodeMixedRadix(builder, iter, inShape, `coords`);
            const outCoords = inCoords.map((c, i) => (axesList.includes(i) ? zeroConst : c));
            flatOutIdx = buildLinearIndex(builder, outCoords, accStrides, `out_idx`);
        }

        const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        // 4. Accumulate (0D scalar register carry for contiguous trailing suffix reductions)
        const dtype = inputs[0].literalType;
        const rank = inShape.length;
        const sortedAxes = Array.from(new Set(axesList)).sort((a, b) => a - b);
        const isContiguousSuffix =
            sortedAxes.length > 0 &&
            sortedAxes.every((ax, idx) => ax === rank - sortedAxes.length + idx);
        const canScalarAcc =
            loopCtx !== undefined &&
            isContiguousSuffix &&
            (dtype === DataType.FLOAT ||
                dtype === DataType.DOUBLE ||
                dtype === DataType.INT32 ||
                dtype === DataType.INT64);

        let accIn: ReturnType<LoopCarryContext["addCarry"]> | undefined;
        let currentAcc: ConcreteValueNode;
        if (canScalarAcc) {
            const initAcc = loopCtx!.outerBuilder.createConstant(
                `red_init_acc_${op.id}`,
                makeTensorProto(dtype, [], [0]),
            );
            accIn = loopCtx!.addCarry(`red_acc_${op.id}`, initAcc, dtype, []);
            currentAcc = accIn;
        } else {
            currentAcc = gatherScalar(builder, carryNode, flatOutIdx);
        }

        let mathOpType = "Add";
        if (op.type === "ReduceMax") mathOpType = "Max";
        else if (op.type === "ReduceMin") mathOpType = "Min";
        else if (op.type === "ReduceProd") mathOpType = "Mul";
        const [mathOut] = builder.createOp(mathOpType, [currentAcc, val]);

        // 5. First-hit logic
        let isFirstHit: ConcreteValueNode | undefined;
        for (const ax of axesList) {
            const [isZero] = builder.createOp("Equal", [inCoords[ax], zeroConst]);
            isFirstHit = isFirstHit ? builder.createOp("And", [isFirstHit, isZero])[0] : isZero;
        }

        const [finalValue] = builder.createOp("Where", [isFirstHit!, val, mathOut]);
        if (accIn !== undefined) {
            loopCtx!.setNextState(accIn, finalValue);
        }
        const [updateVal] = builder.createOp("Unsqueeze", [finalValue, axes]);
        const [nextCarry] = builder.createOp(
            "ScatterElements",
            [carryNode, flatOutIdxUnsq, updateVal],
            { axis: 0 },
        );

        return { resultNode: finalValue, nextCarry };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        const outDtype = loopOut.literalType || DataType.FLOAT;
        const outShape = toStaticShape(loopOut.shape) as KnownShape;
        const outSpec = [{ type: outDtype, shape: outShape }];

        if (op.type === "ReduceL2") return builder.createOp("Sqrt", [loopOut], {}, outSpec)[0];
        if (op.type === "ReduceLogSum" || op.type === "ReduceLogSumExp")
            return builder.createOp("Log", [loopOut], {}, outSpec)[0];

        if (op.type === "ReduceMean") {
            const bounds = this.getLoopBounds(op, []);

            // --- DYNAMIC SUPPORT ---
            const isDynamic =
                typeof bounds.totalIters !== "number" || !Array.isArray(bounds.carryShape);

            if (isDynamic) {
                const iterNode =
                    typeof bounds.totalIters === "number"
                        ? builder.createConstant(`mean_iters`, scalarInt64(bounds.totalIters))
                        : bounds.totalIters;

                const carryLenNode = Array.isArray(bounds.carryShape)
                    ? builder.createConstant(
                          `mean_carry`,
                          scalarInt64(bounds.carryShape[0] as number),
                      )
                    : builder.createOp("Squeeze", [
                          bounds.carryShape,
                          builder.createConstant("sq_ax", int64Vec([0])),
                      ])[0];

                const [countRaw] = builder.createOp("Div", [iterNode, carryLenNode]);
                const [countCast] = builder.createOp("Cast", [countRaw], {
                    to: outDtype,
                });

                return builder.createOp("Div", [loopOut, countCast], {}, outSpec)[0];
            }

            const boundsCarryShape = bounds.carryShape as KnownShape;
            const count = (bounds.totalIters as number) / (boundsCarryShape[0] as number);
            const isFloat =
                outDtype === DataType.FLOAT ||
                outDtype === DataType.DOUBLE ||
                outDtype === DataType.FLOAT16 ||
                outDtype === DataType.BFLOAT16;
            if (isFloat && Number.isFinite(count) && count !== 0) {
                const invConst = builder.createConstant(
                    `mean_inv_count`,
                    makeTensorProto(outDtype, [], [1 / count]),
                );
                return builder.createOp("Mul", [loopOut, invConst], {}, outSpec)[0];
            }
            const countConst = builder.createConstant(`mean_count`, {
                dataType: outDtype,
                dims: [],
                floatData: [count],
                int64Data: [BigInt(count)],
            });
            return builder.createOp("Div", [loopOut, countConst], {}, outSpec)[0];
        }
        return loopOut;
    }
}
