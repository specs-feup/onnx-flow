import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { DataType } from "../../../OnnxTypes.js";
import { type ValueNode, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    getIntsAttr,
    getStringAttr,
    int64Vec,
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
    buildMixedStrides,
    captureMixedList,
    decodeMixedRadix,
    ensureFlatInput,
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    mulMixed,
    packShapeVector,
    resolvePoolShapeAndPadsMixed,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";
import { isExactPowerOfTwoFloat } from "../../canonicalization/recipes/LowerDivByConstantRecipe.js";

export class LowerAveragePoolLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerAveragePoolLoop";
    public readonly targetOp = "AveragePool";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["pooling", "nd", "self-carry"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Concat",
        "ReduceProd",
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
        const kernel = getIntsAttr(op, "kernel_shape", []);
        return kernel.length === X.shape.length - 2;
    }

    private resolveMixedPadsAndOutSpatial(
        op: OperationNode.Class,
        builder: GraphBuilder,
        inDims: (number | ConcreteValueNode)[],
        tag: string,
    ): { pHeads: (number | ConcreteValueNode)[]; outSpatial: (number | ConcreteValueNode)[] } {
        const spatialRank = inDims.length - 2;
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const ceilMode = getIntAttr(op, "ceil_mode", 0);
        const explicitPads = getIntsAttr(op, "pads", []);

        return resolvePoolShapeAndPadsMixed(
            autoPad,
            inDims.slice(2),
            kernel,
            strides,
            dilations,
            explicitPads,
            ceilMode,
            builder,
            tag,
        );
    }

    private usesConstantDivisor(op: OperationNode.Class, rawInDims: number[]): boolean {
        const countIncludePad = getIntAttr(op, "count_include_pad", 0);
        if (countIncludePad === 1) return true;
        const ceilMode = getIntAttr(op, "ceil_mode", 0);
        if (ceilMode !== 0) return false;

        const spatialRank = rawInDims.length - 2;
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const explicitPads = getIntsAttr(op, "pads", []);

        if (!rawInDims.slice(2).includes(-1)) {
            const kernel = getIntsAttr(op, "kernel_shape", []);
            const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
            const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
            const pads = resolveAutoPads(
                autoPad,
                rawInDims.slice(2),
                kernel,
                strides,
                dilations,
                explicitPads,
            );
            return pads.every((p) => p === 0);
        }

        if (autoPad === "VALID") return true;
        if (autoPad === "NOTSET") return explicitPads.every((p) => p === 0);
        return false;
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
        const rawInDims = toStaticShape(X.shape);
        const kernel = getIntsAttr(op, "kernel_shape", []);
        const kTotal = kernel.reduce((a, b) => a * b, 1);

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `ap_bounds_${op.id}`);
        if (!rawInDims.includes(-1)) {
            const { outSpatial } = this.resolveMixedPadsAndOutSpatial(
                op,
                builder,
                rawInDims,
                "aps",
            );
            const outDims = [rawInDims[0], rawInDims[1], ...(outSpatial as number[])];
            const outTotal = outDims.reduce((a, b) => a * b, 1);
            return { totalIters: outTotal * kTotal, carryShape: outDims };
        }

        const inDims = extractMixedDims(builder, X, rawInDims, `apb_${op.id}`);
        const { outSpatial } = this.resolveMixedPadsAndOutSpatial(
            op,
            builder,
            inDims,
            `apb_${op.id}`,
        );
        const outDims = [inDims[0], inDims[1], ...outSpatial];
        const targetShape = packShapeVector(builder, outDims, `apb_tsh_${op.id}`);
        const ax0 = builder.createConstant(`apb_ax0_${op.id}`, int64Vec([0]));
        const [outTotal] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [outTotal, ax0]);
        const totalIters = mulMixed(
            builder,
            outTotal,
            kTotal,
            `apb_it_${op.id}`,
        ) as ConcreteValueNode;
        return { totalIters, carryShape, targetShape };
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
        const rawInDims = toStaticShape(X.shape);
        const dtype = X.literalType as DataType;
        const spatialRank = rawInDims.length - 2;

        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const constDivisor = this.usesConstantDivisor(op, rawInDims);
        const canHoist =
            loopCtx !== undefined &&
            !valueMap.has(X.id) &&
            loopCtx.outerBuilder.graph.hasNode(X.id);

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
        const dimBuilder = canHoist ? loopCtx!.outerBuilder : builder;
        const dimSource = canHoist ? X : rawX;
        const rawInDimsList = extractMixedDims(dimBuilder, dimSource, rawInDims, `ap_id_${op.id}`);
        const { pHeads: rawPHeads, outSpatial: rawOutSpatial } = this.resolveMixedPadsAndOutSpatial(
            op,
            dimBuilder,
            rawInDimsList,
            `ap_sh_${op.id}`,
        );

        const rawOutDims = [rawInDimsList[0], rawInDimsList[1], ...rawOutSpatial];
        const rawInStrides = buildMixedStrides(dimBuilder, rawInDimsList);
        const rawOutStrides = buildMixedStrides(dimBuilder, rawOutDims);

        const inDims = canHoist ? captureMixedList(builder, rawInDimsList) : rawInDimsList;
        const pHeads = canHoist ? captureMixedList(builder, rawPHeads) : rawPHeads;
        const outSpatial = canHoist ? captureMixedList(builder, rawOutSpatial) : rawOutSpatial;
        const outDims = [inDims[0], inDims[1], ...outSpatial];
        const iterDims = [...outDims, ...kernel];
        const inStrides = canHoist ? captureMixedList(builder, rawInStrides) : rawInStrides;
        const outStrides = canHoist ? captureMixedList(builder, rawOutStrides) : rawOutStrides;
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
            const pVal = pHeads[d] ?? 0;
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
            if (typeof pVal !== "number") {
                [inCoord] = builder.createOp("Sub", [inCoord, pVal]);
            } else if (pVal !== 0) {
                [inCoord] = builder.createOp("Sub", [
                    inCoord,
                    builder.createConstant(`ap_pad_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }

            const dimNode =
                typeof inDims[2 + d] === "number"
                    ? builder.createConstant(
                          `ap_dim_${d}_${op.id}`,
                          scalarInt64(inDims[2 + d] as number),
                      )
                    : (inDims[2 + d] as ConcreteValueNode);
            const [ltZero] = builder.createOp("Less", [inCoord, zeroI64]);
            const [geZero] = builder.createOp("Not", [ltZero]);
            const [ltDim] = builder.createOp("Less", [inCoord, dimNode]);
            const [dimValid] = builder.createOp("And", [geZero, ltDim]);
            isValid = isValid ? builder.createOp("And", [isValid, dimValid])[0] : dimValid;

            const [safeCoord] = builder.createOp("Where", [dimValid, inCoord, zeroI64]);
            const str = inStrides[2 + d];
            const term =
                typeof str !== "number"
                    ? builder.createOp("Mul", [safeCoord, str])[0]
                    : str === 1
                      ? safeCoord
                      : builder.createOp("Mul", [
                            safeCoord,
                            builder.createConstant(`ap_istr_${d}_${op.id}`, scalarInt64(str)),
                        ])[0];
            [flatInIdx] = builder.createOp("Add", [flatInIdx, term]);
        }

        const xVal = gatherScalar(builder, flatX, flatInIdx);
        const [effVal] = builder.createOp("Where", [isValid!, xVal, zeroVal]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        let prevAcc: ConcreteValueNode;
        let sumIn: ReturnType<LoopCarryContext["addCarry"]> | undefined;
        if (loopCtx !== undefined) {
            const initAcc = loopCtx.outerBuilder.createConstant(
                `ap_init_acc_${op.id}`,
                makeTensorProto(dtype, [], [0]),
            );
            sumIn = loopCtx.addCarry(`ap_acc_${op.id}`, initAcc, dtype, []);
            prevAcc = sumIn;
        } else {
            prevAcc = gatherScalar(builder, carryNode, flatOutIdx);
        }

        const [baseAcc] = builder.createOp("Where", [isFirstTap!, zeroVal, prevAcc]);
        const [newSum] = builder.createOp("Add", [baseAcc, effVal]);
        if (sumIn !== undefined) {
            loopCtx!.setNextState(sumIn, newSum);
        }
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
        } else if (constDivisor) {
            const kTotal = kernel.reduce((a, b) => a * b, 1);
            if (kTotal > 1) {
                let avg: ConcreteValueNode;
                if (isExactPowerOfTwoFloat(kTotal, dtype)) {
                    const invConst = builder.createConstant(
                        `ap_invk_${op.id}`,
                        makeTensorProto(dtype, [], [1 / kTotal]),
                    );
                    [avg] = builder.createOp("Mul", [newSum, invConst]);
                } else {
                    const kConst = builder.createConstant(
                        `ap_ktotal_${op.id}`,
                        makeTensorProto(dtype, [], [kTotal]),
                    );
                    [avg] = builder.createOp("Div", [newSum, kConst]);
                }
                [storeVal] = builder.createOp("Where", [isLastTap!, avg, newSum]);
            }
        }

        const [storeUnsq] = builder.createOp("Unsqueeze", [storeVal, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, storeUnsq], {
            axis: 0,
        });

        return { resultNode: storeVal, nextCarry };
    }
}
