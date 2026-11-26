import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    buildMixedStrides,
    decodeMixedRadix,
    ensureFlatInput,
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerReverseSequenceRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerReverseSequence";
    public readonly targetOp = "ReverseSequence";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["indexing", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "ReduceProd",
        "Reshape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Less",
        "Equal",
        "And",
        "Where",
        "Cast",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "ReverseSequence") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const dims = toStaticShape(ins[0].shape);
        return dims.length >= 2;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        const X = op.getInputs()![0];
        const dims = toStaticShape(X.shape);
        if (!dims.includes(-1)) {
            const total = dims.reduce((a, b) => a * b, 1);
            return { totalIters: total, carryShape: [total] };
        }
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `rs_bounds_${op.id}`);
        const ax0 = builder.createConstant(`rsb_ax0_${op.id}`, int64Vec([0]));
        const [targetShape] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [dims.length] },
        ]);
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, ax0]);
        return { totalIters, carryShape, targetShape };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        _carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `revseq_${op.id}`);
        const ins = op.getInputs()!;
        const rawDims = toStaticShape(ins[0].shape);
        const rank = rawDims.length;

        const batchAxis = normalizeAxis(getIntAttr(op, "batch_axis", 1), rank);
        const timeAxis = normalizeAxis(getIntAttr(op, "time_axis", 0), rank);

        const rawIn = resolveRecipeInput(
            builder,
            ins[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const rawLens = resolveRecipeInput(
            builder,
            ins[1],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const dims = extractMixedDims(builder, rawIn, rawDims, `rs_${op.id}`);
        const strides = buildMixedStrides(builder, dims);
        const timeStride = strides[timeAxis];
        const flatIn = ensureFlatInput(builder, rawIn);
        const flatLens = ensureFlatInput(builder, rawLens);

        let batchCoord: ConcreteValueNode;
        let timeCoord: ConcreteValueNode;
        let basePtr: ConcreteValueNode;

        const baseStrides: (number | ConcreteValueNode)[] = [...strides];
        baseStrides[timeAxis] = 0;

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                dims,
                `rs_cnt_${op.id}`,
            );
            batchCoord = counters[batchAxis];
            timeCoord = counters[timeAxis];
            basePtr = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                dims,
                baseStrides,
                `rs_base_${op.id}`,
            );
        } else {
            const coords = decodeMixedRadix(builder, iter, dims, `rs_dec_${op.id}`);
            batchCoord = coords[batchAxis];
            timeCoord = coords[timeAxis];
            basePtr = buildLinearIndex(builder, coords, baseStrides, `rs_lin_${op.id}`);
        }

        // seqLen = sequence_lens[batchCoord]
        let seqLen = gatherScalar(builder, flatLens, batchCoord);
        if (ins[1].literalType !== DataType.INT64) {
            [seqLen] = builder.createOp("Cast", [seqLen], { to: DataType.INT64 });
        }

        // srcTime = (timeCoord < seqLen) ? (seqLen - 1 - timeCoord) : timeCoord
        const oneI64 = builder.createConstant(`rs_one_${op.id}`, scalarInt64(1));
        const [inRange] = builder.createOp("Less", [timeCoord, seqLen]);
        const [lenMinus1] = builder.createOp("Sub", [seqLen, oneI64]);
        const [revTime] = builder.createOp("Sub", [lenMinus1, timeCoord]);
        const [srcTime] = builder.createOp("Where", [inRange, revTime, timeCoord]);

        // flatSrcIdx = basePtr + srcTime * timeStride
        let timeOffset: ConcreteValueNode = srcTime;
        if (typeof timeStride !== "number") {
            [timeOffset] = builder.createOp("Mul", [srcTime, timeStride]);
        } else if (timeStride !== 1) {
            const tsConst = builder.createConstant(`rs_tstr_${op.id}`, scalarInt64(timeStride));
            [timeOffset] = builder.createOp("Mul", [srcTime, tsConst]);
        }
        const [flatSrcIdx] = builder.createOp("Add", [basePtr, timeOffset]);

        return gatherScalar(builder, flatIn, flatSrcIdx);
    }
}
