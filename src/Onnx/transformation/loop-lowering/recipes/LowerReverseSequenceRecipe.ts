import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    computeStrides,
    getIntAttr,
    normalizeAxis,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
    squeezeIfLen1,
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
    public readonly producedOps = [
        "Loop",
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
        return dims.length >= 2 && !dims.includes(-1);
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
        const dims = toStaticShape(ins[0].shape);
        const rank = dims.length;

        const batchAxis = normalizeAxis(getIntAttr(op, "batch_axis", 1), rank);
        const timeAxis = normalizeAxis(getIntAttr(op, "time_axis", 0), rank);
        const strides = computeStrides(dims);
        const timeStride = strides[timeAxis];

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
        const flatIn = ensureFlatInput(builder, rawIn);
        const flatLens = ensureFlatInput(builder, rawLens);

        let batchCoord: ConcreteValueNode;
        let timeCoord: ConcreteValueNode;
        let basePtr: ConcreteValueNode;

        const baseStrides = [...strides];
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
        const [bUnsq] = builder.createOp("Unsqueeze", [batchCoord, axes]);
        const [lenRaw] = builder.createOp("Gather", [flatLens, bUnsq], { axis: 0 });
        let seqLen = squeezeIfLen1(
            builder,
            lenRaw,
            axes,
            `rs_sq_len_${op.id}`,
        ) as ConcreteValueNode;
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
        if (timeStride !== 1) {
            const tsConst = builder.createConstant(`rs_tstr_${op.id}`, scalarInt64(timeStride));
            [timeOffset] = builder.createOp("Mul", [srcTime, tsConst]);
        }
        const [flatSrcIdx] = builder.createOp("Add", [basePtr, timeOffset]);

        const [srcUnsq] = builder.createOp("Unsqueeze", [flatSrcIdx, axes]);
        const [gathered] = builder.createOp("Gather", [flatIn, srcUnsq], { axis: 0 });
        return squeezeIfLen1(builder, gathered, axes, `rs_sq_out_${op.id}`);
    }
}
