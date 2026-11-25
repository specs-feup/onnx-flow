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
    resolveAutoPads,
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

export class LowerAveragePoolLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerAveragePoolLoop";
    public readonly targetOp = "AveragePool";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["pooling", "nd", "self-carry"] as const;
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
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "Less",
        "And",
        "Not",
        "Where",
        "Max",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "AveragePool") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 3) return false;
        const dims = toStaticShape(X.shape);
        const kernel = getIntsAttr(op, "kernel_shape", []);
        return !dims.includes(-1) && kernel.length === dims.length - 2;
    }

    private resolvePadsAndOutSpatial(
        op: OperationNode.Class,
        inDims: number[],
    ): { pads: number[]; outSpatial: number[] } {
        const spatialRank = inDims.length - 2;
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const ceilMode = getIntAttr(op, "ceil_mode", 0);
        const explicitPads = getIntsAttr(op, "pads", []);

        const pads = resolveAutoPads(
            autoPad,
            inDims.slice(2),
            kernel,
            strides,
            dilations,
            explicitPads,
        );
        const outSpatial = inDims
            .slice(2)
            .map((inD, d) =>
                inferPoolDim(
                    inD,
                    kernel[d],
                    strides[d] ?? 1,
                    pads[d] ?? 0,
                    pads[spatialRank + d] ?? 0,
                    dilations[d] ?? 1,
                    ceilMode,
                ),
            );

        return { pads, outSpatial };
    }

    private usesConstantDivisor(op: OperationNode.Class, pads: number[]): boolean {
        const countIncludePad = getIntAttr(op, "count_include_pad", 0);
        const ceilMode = getIntAttr(op, "ceil_mode", 0);
        return countIncludePad === 1 || (ceilMode === 0 && pads.every((p) => p === 0));
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const inDims = toStaticShape(op.getInputs()![0].shape);
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const { outSpatial } = this.resolvePadsAndOutSpatial(op, inDims);
        const outDims = [inDims[0], inDims[1], ...outSpatial];
        const outTotal = outDims.reduce((a, b) => a * b, 1);
        const kTotal = kernel.reduce((a, b) => a * b, 1);
        return { totalIters: outTotal * kTotal, carryShape: outDims };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        const inDims = toStaticShape(op.getInputs()![0].shape);
        const { pads } = this.resolvePadsAndOutSpatial(op, inDims);
        if (!this.usesConstantDivisor(op, pads)) {
            return loopOut;
        }
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const kTotal = kernel.reduce((a, b) => a * b, 1);
        const dtype = (loopOut.literalType || DataType.FLOAT) as DataType;
        const divConst = builder.createConstant(
            `ap_ktotal_${op.id}`,
            makeTensorProto(dtype, [], [kTotal]),
        );
        return builder.createOp("Div", [loopOut, divConst])[0];
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
        const builder = new GraphBuilder(body, `avgpool_${op.id}`);
        const X = op.getInputs()![0];
        const inDims = toStaticShape(X.shape);
        const dtype = X.literalType as DataType;
        const spatialRank = inDims.length - 2;

        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const { pads, outSpatial } = this.resolvePadsAndOutSpatial(op, inDims);
        const constDivisor = this.usesConstantDivisor(op, pads);

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
                `ap_cnt_${op.id}`,
            );
            coords = counters;
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                outPtrStrides,
                `ap_out_${op.id}`,
            );
            ncBasePtr = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                ncPtrStrides,
                `ap_nc_${op.id}`,
            );
        } else {
            coords = decodeMixedRadix(builder, iter, iterDims, `ap_dec_${op.id}`);
            flatOutIdx = buildLinearIndex(builder, coords, outPtrStrides, `ap_olin_${op.id}`);
            ncBasePtr = buildLinearIndex(builder, coords, ncPtrStrides, `ap_nclin_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`ap_z64_${op.id}`, scalarInt64(0));
        const zeroVal = builder.createConstant(`ap_zval_${op.id}`, makeTensorProto(dtype, [], [0]));
        const oneVal = builder.createConstant(`ap_oval_${op.id}`, makeTensorProto(dtype, [], [1]));

        let flatInIdx: ConcreteValueNode = ncBasePtr;
        let isValid: ConcreteValueNode | null = null;
        let isFirstTap: ConcreteValueNode | null = null;
        let isLastTap: ConcreteValueNode | null = null;

        for (let d = 0; d < spatialRank; d++) {
            const oCoord = coords[2 + d];
            const kCoord = coords[2 + spatialRank + d];

            const [kIsZero] = builder.createOp("Equal", [kCoord, zeroI64]);
            isFirstTap = isFirstTap ? builder.createOp("And", [isFirstTap, kIsZero])[0] : kIsZero;

            const kLastConst = builder.createConstant(
                `ap_klast_${d}_${op.id}`,
                scalarInt64(kernel[d] - 1),
            );
            const [kIsLast] = builder.createOp("Equal", [kCoord, kLastConst]);
            isLastTap = isLastTap ? builder.createOp("And", [isLastTap, kIsLast])[0] : kIsLast;

            const sVal = strides[d] ?? 1;
            const pVal = pads[d] ?? 0;
            const dilVal = dilations[d] ?? 1;

            let inCoord: ConcreteValueNode =
                sVal === 1
                    ? oCoord
                    : builder.createOp("Mul", [
                          oCoord,
                          builder.createConstant(`ap_st_${d}_${op.id}`, scalarInt64(sVal)),
                      ])[0];

            const kOff: ConcreteValueNode =
                dilVal === 1
                    ? kCoord
                    : builder.createOp("Mul", [
                          kCoord,
                          builder.createConstant(`ap_dil_${d}_${op.id}`, scalarInt64(dilVal)),
                      ])[0];

            [inCoord] = builder.createOp("Add", [inCoord, kOff]);
            if (pVal !== 0) {
                [inCoord] = builder.createOp("Sub", [
                    inCoord,
                    builder.createConstant(`ap_pad_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }

            const dimConst = builder.createConstant(
                `ap_dim_${d}_${op.id}`,
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
                          builder.createConstant(`ap_istr_${d}_${op.id}`, scalarInt64(str)),
                      ])[0];
            [flatInIdx] = builder.createOp("Add", [flatInIdx, term]);
        }

        const [inUnsq] = builder.createOp("Unsqueeze", [flatInIdx, axes]);
        const [rawVal] = builder.createOp("Gather", [flatX, inUnsq], { axis: 0 });
        const xVal = squeezeIfLen1(builder, rawVal, axes, `ap_sqx_${op.id}`) as ConcreteValueNode;
        const [effVal] = builder.createOp("Where", [isValid!, xVal, zeroVal]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const [accRaw] = builder.createOp("Gather", [carryNode, outUnsq], { axis: 0 });
        const prevAcc = squeezeIfLen1(
            builder,
            accRaw,
            axes,
            `ap_sqa_${op.id}`,
        ) as ConcreteValueNode;

        const [newSum] = builder.createOp("Add", [prevAcc, effVal]);
        let storeVal: ConcreteValueNode = newSum;

        if (!constDivisor && loopCtx !== undefined) {
            const initCnt = loopCtx.outerBuilder.createConstant(
                `ap_init_cnt_${op.id}`,
                makeTensorProto(dtype, [], [0]),
            );
            const cntIn = loopCtx.addCarry(`ap_vcnt_${op.id}`, initCnt, dtype, []);
            const [baseCnt] = builder.createOp("Where", [isFirstTap!, zeroVal, cntIn]);
            const [tapCnt] = builder.createOp("Where", [isValid!, oneVal, zeroVal]);
            const [newCnt] = builder.createOp("Add", [baseCnt, tapCnt]);
            loopCtx.setNextState(cntIn, newCnt);

            const [safeCnt] = builder.createOp("Max", [newCnt, oneVal]);
            const [avg] = builder.createOp("Div", [newSum, safeCnt]);
            [storeVal] = builder.createOp("Where", [isLastTap!, avg, newSum]);
        }

        const [storeUnsq] = builder.createOp("Unsqueeze", [storeVal, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, storeUnsq], {
            axis: 0,
        });

        return { resultNode: storeVal, nextCarry };
    }
}
