import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { DataType } from "../../../OnnxTypes.js";
import { type ValueNode, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import {
    computeStrides,
    getIntAttr,
    makeTensorProto,
    normalizeAxis,
    readScalarFromTensorNode,
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

export class LowerCumSumRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerCumSum";
    public readonly targetOp = "CumSum";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Equal",
        "And",
        "Where",
        "Mul",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "CumSum") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const dims = toStaticShape(ins[0].shape);
        const axisVal = readScalarFromTensorNode(ins[1]);
        return dims.length > 0 && !dims.includes(-1) && axisVal !== undefined;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const dims = toStaticShape(op.getInputs()![0].shape);
        const total = dims.reduce((a, b) => a * b, 1);
        return { totalIters: total, carryShape: dims };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `cumsum_${op.id}`);
        const ins = op.getInputs()!;
        const X = ins[0];
        const dims = toStaticShape(X.shape);
        const dtype = X.literalType as DataType;

        const axis = normalizeAxis(Math.trunc(readScalarFromTensorNode(ins[1])!), dims.length);
        const exclusive = getIntAttr(op, "exclusive", 0) !== 0;
        const reverse = getIntAttr(op, "reverse", 0) !== 0;

        const strides = computeStrides(dims);
        const axisStride = strides[axis];
        const axisDim = dims[axis];

        const rawInnerX = resolveRecipeInput(
            builder,
            X,
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const flatX = ensureFlatInput(builder, rawInnerX);

        let flatOutIdx: ConcreteValueNode;
        let axisCoord: ConcreteValueNode;

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                dims,
                `cs_cnt_${op.id}`,
            );
            axisCoord = counters[axis];

            const ptrStrides = [...strides];
            let initialOffset = 0;
            if (reverse) {
                ptrStrides[axis] = -axisStride;
                initialOffset = (axisDim - 1) * axisStride;
            }
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                dims,
                ptrStrides,
                `cs_ptr_${op.id}`,
                initialOffset,
            );
        } else {
            const coords = decodeMixedRadix(builder, iter, dims, `cs_dec_${op.id}`);
            axisCoord = coords[axis];
            const mappedCoords = [...coords];
            if (reverse) {
                const lastIdx = builder.createConstant(
                    `cs_last_${op.id}`,
                    scalarInt64(axisDim - 1),
                );
                [mappedCoords[axis]] = builder.createOp("Sub", [lastIdx, axisCoord]);
            }
            flatOutIdx = buildLinearIndex(builder, mappedCoords, strides, `cs_lin_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`cs_z64_${op.id}`, scalarInt64(0));
        const strideConst = builder.createConstant(`cs_str_${op.id}`, scalarInt64(axisStride));
        const zeroVal = builder.createConstant(`cs_zval_${op.id}`, makeTensorProto(dtype, [], [0]));

        const [isFirst] = builder.createOp("Equal", [axisCoord, zeroI64]);

        // Previous element index along the cumsum axis (clamped to 0 when isFirst)
        const [rawPrevIdx] = reverse
            ? builder.createOp("Add", [flatOutIdx, strideConst])
            : builder.createOp("Sub", [flatOutIdx, strideConst]);
        const [safePrevIdx] = builder.createOp("Where", [isFirst, zeroI64, rawPrevIdx]);

        const [prevIdxUnsq] = builder.createOp("Unsqueeze", [safePrevIdx, axes]);
        const [prevCarryRaw] = builder.createOp("Gather", [carryNode, prevIdxUnsq], { axis: 0 });
        const prevCarry = squeezeIfLen1(builder, prevCarryRaw, axes, `cs_sq_prev_${op.id}`);

        const [outIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        let outVal: ConcreteValueNode;
        if (exclusive) {
            // exclusive: out[0] = 0; out[i] = out[i-1] + x[i-1]
            const [xPrevRaw] = builder.createOp("Gather", [flatX, prevIdxUnsq], { axis: 0 });
            const xPrev = squeezeIfLen1(builder, xPrevRaw, axes, `cs_sq_xp_${op.id}`);
            const [sum] = builder.createOp("Add", [prevCarry, xPrev]);
            [outVal] = builder.createOp("Where", [isFirst, zeroVal, sum]);
        } else {
            // inclusive: out[0] = x[0]; out[i] = out[i-1] + x[i]
            const [xCurRaw] = builder.createOp("Gather", [flatX, outIdxUnsq], { axis: 0 });
            const xCur = squeezeIfLen1(builder, xCurRaw, axes, `cs_sq_xc_${op.id}`);
            const [prevEff] = builder.createOp("Where", [isFirst, zeroVal, prevCarry]);
            [outVal] = builder.createOp("Add", [prevEff, xCur]);
        }

        const [outValUnsq] = builder.createOp("Unsqueeze", [outVal, axes]);
        const [nextCarry] = builder.createOp(
            "ScatterElements",
            [carryNode, outIdxUnsq, outValUnsq],
            { axis: 0 },
        );

        return { resultNode: outVal, nextCarry };
    }
}
