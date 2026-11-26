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
    getIntsAttr,
    getStringAttr,
    int64Vec,
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

export class LowerMaxPoolRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerMaxPool";
    public readonly targetOp = "MaxPool";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["pooling", "self-carry"] as const;
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
        if (op.getOutputs().length > 1) return false; // optional Indices output not lowered here
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 3) return false;
        const dims = toStaticShape(X.shape);
        const kernel = getIntsAttr(op, "kernel_shape", []);
        return kernel.length === dims.length - 2;
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

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `mp_bounds_${op.id}`);
        if (!rawInDims.includes(-1)) {
            const { outSpatial } = this.resolveMixedPadsAndOutSpatial(
                op,
                builder,
                rawInDims,
                "mps",
            );
            const outDims = [rawInDims[0], rawInDims[1], ...(outSpatial as number[])];
            const outTotal = outDims.reduce((a, b) => a * b, 1);
            return { totalIters: outTotal * kTotal, carryShape: outDims };
        }

        const inDims = extractMixedDims(builder, X, rawInDims, `mpb_${op.id}`);
        const { outSpatial } = this.resolveMixedPadsAndOutSpatial(
            op,
            builder,
            inDims,
            `mpb_${op.id}`,
        );
        const outDims = [inDims[0], inDims[1], ...outSpatial];
        const targetShape = packShapeVector(builder, outDims, `mpb_tsh_${op.id}`);
        const ax0 = builder.createConstant(`mpb_ax0_${op.id}`, int64Vec([0]));
        const [outTotal] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [outTotal, ax0]);
        const totalIters = mulMixed(
            builder,
            outTotal,
            kTotal,
            `mpb_it_${op.id}`,
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
        const builder = new GraphBuilder(body, `maxpool_${op.id}`);
        const X = op.getInputs()![0];
        const rawInDims = toStaticShape(X.shape);
        const dtype = X.literalType as DataType;
        const spatialRank = rawInDims.length - 2;

        const kernel = getIntsAttr(op, "kernel_shape", []);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
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
        const rawInDimsList = extractMixedDims(dimBuilder, dimSource, rawInDims, `mp_id_${op.id}`);
        const { pHeads: rawPHeads, outSpatial: rawOutSpatial } = this.resolveMixedPadsAndOutSpatial(
            op,
            dimBuilder,
            rawInDimsList,
            `mp_sh_${op.id}`,
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
            const pVal = pHeads[d] ?? 0;
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
            if (typeof pVal !== "number") {
                [inCoord] = builder.createOp("Sub", [inCoord, pVal]);
            } else if (pVal !== 0) {
                [inCoord] = builder.createOp("Sub", [
                    inCoord,
                    builder.createConstant(`mp_pad_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }

            const dimNode =
                typeof inDims[2 + d] === "number"
                    ? builder.createConstant(
                          `mp_dim_${d}_${op.id}`,
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
                            builder.createConstant(`mp_istr_${d}_${op.id}`, scalarInt64(str)),
                        ])[0];
            [flatInIdx] = builder.createOp("Add", [flatInIdx, term]);
        }

        const xVal = gatherScalar(builder, flatX, flatInIdx);
        const [effVal] = builder.createOp("Where", [isValid!, xVal, negInf]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        let baseAcc: ConcreteValueNode;
        let accIn: ReturnType<LoopCarryContext["addCarry"]> | undefined;
        if (loopCtx !== undefined) {
            const initAcc = loopCtx.outerBuilder.createConstant(
                `mp_init_acc_${op.id}`,
                makeTensorProto(dtype, [], [negInfVal]),
            );
            accIn = loopCtx.addCarry(`mp_acc_${op.id}`, initAcc, dtype, []);
            [baseAcc] = builder.createOp("Where", [isFirstTap!, negInf, accIn]);
        } else {
            const prevAcc = gatherScalar(builder, carryNode, flatOutIdx);
            [baseAcc] = builder.createOp("Where", [isFirstTap!, negInf, prevAcc]);
        }
        const [newMax] = builder.createOp("Max", [baseAcc, effVal]);
        if (accIn !== undefined) {
            loopCtx!.setNextState(accIn, newMax);
        }

        const [newMaxUnsq] = builder.createOp("Unsqueeze", [newMax, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, newMaxUnsq], {
            axis: 0,
        });

        return { resultNode: newMax, nextCarry };
    }
}
