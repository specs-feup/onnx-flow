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
    getIntsAttr,
    getStringAttr,
    inferPoolDim,
    makeTensorProto,
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

export class LowerMaxPoolRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerMaxPool";
    public readonly targetOp = "MaxPool";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["pooling", "self-carry"] as const;
    public readonly producedOps = [
        "Loop",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Equal",
        "Less",
        "Greater",
        "And",
        "Not",
        "Where",
        "Max",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "MaxPool") return false;
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") return false;
        if (op.getOutputs().length > 1) return false; // optional Indices output not lowered here
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 3) return false;
        const dims = toStaticShape(X.shape);
        const kernel = getIntsAttr(op, "kernel_shape", []);
        return !dims.includes(-1) && kernel.length === dims.length - 2;
    }

    private computeOutSpatial(op: OperationNode.Class, inDims: number[]): number[] {
        const spatialRank = inDims.length - 2;
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const pads = getIntsAttr(op, "pads", new Array<number>(spatialRank * 2).fill(0));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const ceilMode = getIntAttr(op, "ceil_mode", 0);

        const outSpatial: number[] = [];
        for (let d = 0; d < spatialRank; d++) {
            outSpatial.push(
                inferPoolDim(
                    inDims[2 + d],
                    kernel[d],
                    strides[d] ?? 1,
                    pads[d] ?? 0,
                    pads[spatialRank + d] ?? 0,
                    dilations[d] ?? 1,
                    ceilMode,
                ),
            );
        }
        return outSpatial;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const inDims = toStaticShape(op.getInputs()![0].shape);
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const outSpatial = this.computeOutSpatial(op, inDims);
        const outTotal = inDims[0] * inDims[1] * outSpatial.reduce((a, b) => a * b, 1);
        const kTotal = kernel.reduce((a, b) => a * b, 1);
        return { totalIters: outTotal * kTotal, carryShape: [inDims[0], inDims[1], ...outSpatial] };
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
        const builder = new GraphBuilder(body, `maxpool_${op.id}`);
        const X = op.getInputs()![0];
        const inDims = toStaticShape(X.shape);
        const dtype = X.literalType as DataType;
        const spatialRank = inDims.length - 2;

        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const pads = getIntsAttr(op, "pads", new Array<number>(spatialRank * 2).fill(0));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const outSpatial = this.computeOutSpatial(op, inDims);

        const outDims = [inDims[0], inDims[1], ...outSpatial];
        const iterDims = [...outDims, ...kernel];
        const inStrides = computeStrides(inDims);
        const outStrides = computeStrides(outDims);

        const rawX = resolveRecipeInput(
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
        const flatX = ensureFlatInput(builder, rawX);

        let coords: ConcreteValueNode[];
        let flatOutIdx: ConcreteValueNode;
        let ncBasePtr: ConcreteValueNode;

        const outPtrStrides = [...outStrides, ...new Array<number>(spatialRank).fill(0)];
        const ncPtrStrides = [
            inStrides[0],
            inStrides[1],
            ...new Array<number>(spatialRank * 2).fill(0),
        ];

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                `mp_cnt_${op.id}`,
            );
            coords = counters;
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                outPtrStrides,
                `mp_out_${op.id}`,
            );
            ncBasePtr = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                ncPtrStrides,
                `mp_nc_${op.id}`,
            );
        } else {
            coords = decodeMixedRadix(builder, iter, iterDims, `mp_dec_${op.id}`);
            flatOutIdx = buildLinearIndex(builder, coords, outPtrStrides, `mp_olin_${op.id}`);
            ncBasePtr = buildLinearIndex(builder, coords, ncPtrStrides, `mp_nclin_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`mp_z64_${op.id}`, scalarInt64(0));
        const negInfVal =
            dtype === DataType.FLOAT || dtype === DataType.DOUBLE
                ? -3.4028234663852886e38
                : -2147483648;
        const negInf = builder.createConstant(
            `mp_neginf_${op.id}`,
            makeTensorProto(dtype, [], [negInfVal]),
        );

        // Compute spatial input coordinates, bounds mask, and first-kernel-tap mask
        let flatInIdx: ConcreteValueNode = ncBasePtr;
        let isValid: ConcreteValueNode | null = null;
        let isFirstTap: ConcreteValueNode | null = null;

        for (let d = 0; d < spatialRank; d++) {
            const oCoord = coords[2 + d];
            const kCoord = coords[2 + spatialRank + d];

            const [kIsZero] = builder.createOp("Equal", [kCoord, zeroI64]);
            isFirstTap = isFirstTap ? builder.createOp("And", [isFirstTap, kIsZero])[0] : kIsZero;

            const sVal = strides[d] ?? 1;
            const pVal = pads[d] ?? 0;
            const dilVal = dilations[d] ?? 1;

            let inCoord: ConcreteValueNode =
                sVal === 1
                    ? oCoord
                    : builder.createOp("Mul", [
                          oCoord,
                          builder.createConstant(`mp_st_${d}_${op.id}`, scalarInt64(sVal)),
                      ])[0];

            const kOff: ConcreteValueNode =
                dilVal === 1
                    ? kCoord
                    : builder.createOp("Mul", [
                          kCoord,
                          builder.createConstant(`mp_dil_${d}_${op.id}`, scalarInt64(dilVal)),
                      ])[0];

            [inCoord] = builder.createOp("Add", [inCoord, kOff]);
            if (pVal !== 0) {
                [inCoord] = builder.createOp("Sub", [
                    inCoord,
                    builder.createConstant(`mp_pad_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }

            const dimConst = builder.createConstant(
                `mp_dim_${d}_${op.id}`,
                scalarInt64(inDims[2 + d]),
            );
            const [ltZero] = builder.createOp("Less", [inCoord, zeroI64]);
            const [geZero] = builder.createOp("Not", [ltZero]);
            const [ltDim] = builder.createOp("Less", [inCoord, dimConst]);
            const [dimValid] = builder.createOp("And", [geZero, ltDim]);
            isValid = isValid ? builder.createOp("And", [isValid, dimValid])[0] : dimValid;

            const [safeCoord] = builder.createOp("Where", [dimValid, inCoord, zeroI64]);
            const str = inStrides[2 + d];
            const term =
                str === 1
                    ? safeCoord
                    : builder.createOp("Mul", [
                          safeCoord,
                          builder.createConstant(`mp_istr_${d}_${op.id}`, scalarInt64(str)),
                      ])[0];
            [flatInIdx] = builder.createOp("Add", [flatInIdx, term]);
        }

        const [inUnsq] = builder.createOp("Unsqueeze", [flatInIdx, axes]);
        const [rawVal] = builder.createOp("Gather", [flatX, inUnsq], { axis: 0 });
        const xVal = squeezeIfLen1(builder, rawVal, axes, `mp_sqx_${op.id}`) as ConcreteValueNode;
        const [effVal] = builder.createOp("Where", [isValid!, xVal, negInf]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const [accRaw] = builder.createOp("Gather", [carryNode, outUnsq], { axis: 0 });
        const prevAcc = squeezeIfLen1(
            builder,
            accRaw,
            axes,
            `mp_sqa_${op.id}`,
        ) as ConcreteValueNode;

        const [baseAcc] = builder.createOp("Where", [isFirstTap!, negInf, prevAcc]);
        const [newMax] = builder.createOp("Max", [baseAcc, effVal]);

        const [newMaxUnsq] = builder.createOp("Unsqueeze", [newMax, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, newMaxUnsq], {
            axis: 0,
        });

        return { resultNode: newMax, nextCarry };
    }
}
