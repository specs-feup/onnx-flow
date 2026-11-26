import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    int64Vec,
    makeTensorProto,
    readConstIntegerVectorFromTensorNode,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildLinearIndex,
    captureMixedList,
    decodeMixedRadix,
    ensureFlatInput,
    ensureInnerValueNode,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    mulMixed,
    packShapeVector,
    productMixed,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";
import ConstantNode from "../../../ConstantNode.js";
import TensorNode from "../../../TensorNode.js";

const isPresent = (n?: ValueNode): n is ValueNode => n !== undefined && n.id !== "";

const readConstVec = (n?: ValueNode): number[] | undefined => {
    if (!isPresent(n)) return undefined;
    if (n.is(ConstantNode)) return readConstIntegerVectorFromTensorNode(n.as(ConstantNode));
    if (n.is(TensorNode)) return readConstIntegerVectorFromTensorNode(n.as(TensorNode));
    return undefined;
};

export class LowerSliceLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerSliceLoop";
    public readonly targetOp = "Slice";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["slicing", "dynamic-bounds", "fuseable"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Less",
        "Greater",
        "Equal",
        "And",
        "Where",
        "Cast",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Slice") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        if (!Array.isArray(ins[0].shape) || ins[0].shape.length === 0) return false;
        const k = toStaticShape(ins[1].shape)[0];
        return k > 0;
    }

    /**
     * Computes full-rank normalized starts, ends, and steps (all 1D INT64 [rank]).
     * Uses zero Div/Mod so it is safe inside Pointer-mode loop bodies.
     */
    private normalizeSliceParams(
        builder: GraphBuilder,
        data: ValueNode,
        startsIn: ValueNode,
        endsIn: ValueNode,
        axesIn: ValueNode | undefined,
        stepsIn: ValueNode | undefined,
        rank: number,
        k: number,
        tag: string,
    ): {
        shapeData: ConcreteValueNode;
        effStarts: ConcreteValueNode;
        effEnds: ConcreteValueNode;
        fullSteps: ConcreteValueNode;
        posStep: ConcreteValueNode;
    } {
        const toI64 = (v: ValueNode): ConcreteValueNode =>
            v.literalType === DataType.INT64
                ? (v as ConcreteValueNode)
                : builder.createOp("Cast", [v], { to: DataType.INT64 })[0];

        const [shapeData] = builder.createOp("Shape", [data], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);

        const starts64 = toI64(startsIn);
        const ends64 = toI64(endsIn);

        const zeroVec = builder.createConstant(`${tag}_zvec`, int64Vec(new Array(rank).fill(0)));
        const oneVec = builder.createConstant(`${tag}_ovec`, int64Vec(new Array(rank).fill(1)));
        const zeroSc = builder.createConstant(`${tag}_zsc`, scalarInt64(0));
        const oneSc = builder.createConstant(`${tag}_osc`, scalarInt64(1));
        const negOneSc = builder.createConstant(`${tag}_m1sc`, scalarInt64(-1));
        const rankSc = builder.createConstant(`${tag}_rsc`, scalarInt64(rank));

        let normAxes: ConcreteValueNode;
        if (axesIn !== undefined) {
            const axes64 = toI64(axesIn);
            const [isNegAx] = builder.createOp("Less", [axes64, zeroSc]);
            const [wrappedAx] = builder.createOp("Add", [axes64, rankSc]);
            [normAxes] = builder.createOp("Where", [isNegAx, wrappedAx, axes64]);
        } else {
            normAxes = builder.createConstant(
                `${tag}_defax`,
                int64Vec(Array.from({ length: k }, (_, i) => i)),
            );
        }

        const [fullStarts] = builder.createOp(
            "ScatterElements",
            [zeroVec, normAxes, starts64],
            { axis: 0 },
            [{ type: DataType.INT64, shape: [rank] }],
        );
        const [fullEnds] = builder.createOp(
            "ScatterElements",
            [shapeData, normAxes, ends64],
            { axis: 0 },
            [{ type: DataType.INT64, shape: [rank] }],
        );
        const fullSteps: ConcreteValueNode =
            stepsIn !== undefined
                ? builder.createOp(
                      "ScatterElements",
                      [oneVec, normAxes, toI64(stepsIn)],
                      { axis: 0 },
                      [{ type: DataType.INT64, shape: [rank] }],
                  )[0]
                : oneVec;

        // Wrap negative starts and ends against shapeData
        const [sNeg] = builder.createOp("Less", [fullStarts, zeroSc]);
        const [sWrap] = builder.createOp("Where", [
            sNeg,
            builder.createOp("Add", [fullStarts, shapeData])[0],
            fullStarts,
        ]);
        const [eNeg] = builder.createOp("Less", [fullEnds, zeroSc]);
        const [eWrap] = builder.createOp("Where", [
            eNeg,
            builder.createOp("Add", [fullEnds, shapeData])[0],
            fullEnds,
        ]);

        const [posStep] = builder.createOp("Greater", [fullSteps, zeroSc]);
        const [maxIdx] = builder.createOp("Sub", [shapeData, oneSc]);

        const clampMax = (
            a: ConcreteValueNode,
            b: ConcreteValueNode,
            outShape: KnownShape = [rank],
        ): ConcreteValueNode => {
            const [gt] = builder.createOp("Greater", [a, b], {}, [
                { type: DataType.BOOL, shape: outShape },
            ]);
            return builder.createOp("Where", [gt, a, b], {}, [
                { type: DataType.INT64, shape: outShape },
            ])[0];
        };
        const clampMin = (
            a: ConcreteValueNode,
            b: ConcreteValueNode,
            outShape: KnownShape = [rank],
        ): ConcreteValueNode => {
            const [lt] = builder.createOp("Less", [a, b], {}, [
                { type: DataType.BOOL, shape: outShape },
            ]);
            return builder.createOp("Where", [lt, a, b], {}, [
                { type: DataType.INT64, shape: outShape },
            ])[0];
        };

        // Positive step: clamp to [0, dim]
        const sClampPos = clampMin(clampMax(sWrap, zeroSc), shapeData);
        const eClampPos = clampMin(clampMax(eWrap, zeroSc), shapeData);

        // Negative step: clamp to [-1, dim - 1]
        const sClampNeg = clampMin(clampMax(sWrap, negOneSc), maxIdx);
        const eClampNeg = clampMin(clampMax(eWrap, negOneSc), maxIdx);

        const [effStarts] = builder.createOp("Where", [posStep, sClampPos, sClampNeg]);
        const [effEnds] = builder.createOp("Where", [posStep, eClampPos, eClampNeg]);

        return { shapeData, effStarts, effEnds, fullSteps, posStep };
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: ConcreteValueNode;
        carryShape: ConcreteValueNode;
        targetShape: ConcreteValueNode;
    } {
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;
        const k = toStaticShape(ins[1].shape)[0];
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `sl_bounds_${op.id}`);

        const { effStarts, effEnds, fullSteps, posStep } = this.normalizeSliceParams(
            builder,
            ins[0],
            ins[1],
            ins[2],
            isPresent(ins[3]) ? ins[3] : undefined,
            isPresent(ins[4]) ? ins[4] : undefined,
            rank,
            k,
            `slb_${op.id}`,
        );

        const zeroSc = builder.createConstant(`slb_z_${op.id}`, scalarInt64(0));

        const clampZero = (v: ConcreteValueNode): ConcreteValueNode => {
            const [gt] = builder.createOp("Greater", [v, zeroSc], {}, [
                { type: DataType.BOOL, shape: [rank] },
            ]);
            return builder.createOp("Where", [gt, v, zeroSc], {}, [
                { type: DataType.INT64, shape: [rank] },
            ])[0];
        };

        const spanPos = clampZero(builder.createOp("Sub", [effEnds, effStarts])[0]);
        const spanNeg = clampZero(builder.createOp("Sub", [effStarts, effEnds])[0]);

        const rawStepsConst = readConstVec(ins[4]);
        const rawAxesConst = readConstVec(ins[3]);
        const hasKnownPerAxisSteps =
            !isPresent(ins[4]) ||
            (rawStepsConst !== undefined && (!isPresent(ins[3]) || rawAxesConst !== undefined));

        let fullConstSteps: number[] | undefined;
        if (hasKnownPerAxisSteps) {
            fullConstSteps = new Array(rank).fill(1);
            if (rawStepsConst !== undefined) {
                const axesList =
                    rawAxesConst ?? Array.from({ length: rawStepsConst.length }, (_, i) => i);
                for (let i = 0; i < axesList.length; i++) {
                    const ax = axesList[i] < 0 ? axesList[i] + rank : axesList[i];
                    if (ax >= 0 && ax < rank) {
                        fullConstSteps[ax] = Number(rawStepsConst[i]);
                    }
                }
            }
        }

        const isAllUnitSteps =
            fullConstSteps !== undefined && fullConstSteps.every((s) => Math.abs(s) === 1);
        const isAllPow2Steps =
            fullConstSteps !== undefined &&
            fullConstSteps.every((s) => {
                const a = Math.abs(s);
                return a > 0 && (a & (a - 1)) === 0;
            });

        let targetShape: ConcreteValueNode;
        if (isAllUnitSteps) {
            [targetShape] = builder.createOp("Where", [posStep, spanPos, spanNeg], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
        } else if (isAllPow2Steps && fullConstSteps !== undefined) {
            const perAxisDims: ConcreteValueNode[] = [];
            for (let d = 0; d < rank; d++) {
                const st = fullConstSteps[d];
                const absSt = Math.abs(st);
                const spanVec = st > 0 ? spanPos : spanNeg;
                const dIdx = builder.createConstant(`slb_didx_${d}_${op.id}`, scalarInt64(d));
                const spanD = gatherScalar(builder, spanVec, dIdx);
                if (absSt === 1) {
                    perAxisDims.push(spanD);
                } else {
                    const shiftK = Math.round(Math.log2(absSt));
                    const roundUp = builder.createConstant(
                        `slb_ru_${d}_${op.id}`,
                        scalarInt64(absSt - 1),
                    );
                    const [num] = builder.createOp("Add", [spanD, roundUp]);
                    const [numU64] = builder.createOp("Cast", [num], { to: DataType.UINT64 });
                    const shiftC = builder.createConstant(
                        `slb_sh_${d}_${op.id}`,
                        makeTensorProto(DataType.UINT64, [], [shiftK]),
                    );
                    const [shifted] = builder.createOp("BitShift", [numU64, shiftC], {
                        direction: "RIGHT",
                    });
                    const [dimD] = builder.createOp("Cast", [shifted], { to: DataType.INT64 });
                    perAxisDims.push(dimD);
                }
            }
            targetShape = packShapeVector(builder, perAxisDims, `slb_tsh_${op.id}`);
        } else {
            const oneSc = builder.createConstant(`slb_o_${op.id}`, scalarInt64(1));
            const negOneSc = builder.createConstant(`slb_m1_${op.id}`, scalarInt64(-1));
            // Positive step: ceil(max(0, effEnds - effStarts) / fullSteps)
            const [posDenom] = builder.createOp("Where", [posStep, fullSteps, oneSc]);
            const [dimPos] = builder.createOp("Div", [
                builder.createOp("Sub", [
                    builder.createOp("Add", [spanPos, posDenom])[0],
                    oneSc,
                ])[0],
                posDenom,
            ]);

            // Negative step: ceil(max(0, effStarts - effEnds) / (-fullSteps))
            const [negSteps] = builder.createOp("Mul", [fullSteps, negOneSc]);
            const [negDenom] = builder.createOp("Where", [posStep, oneSc, negSteps]);
            const [dimNeg] = builder.createOp("Div", [
                builder.createOp("Sub", [
                    builder.createOp("Add", [spanNeg, negDenom])[0],
                    oneSc,
                ])[0],
                negDenom,
            ]);

            [targetShape] = builder.createOp("Where", [posStep, dimPos, dimNeg], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
        }

        const axes0 = builder.createConstant(`slb_ax0_${op.id}`, int64Vec([0]));
        const targetDimsForProd: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            const dIdx = builder.createConstant(`slb_pidx_${d}_${op.id}`, scalarInt64(d));
            targetDimsForProd.push(gatherScalar(builder, targetShape, dIdx));
        }
        const totalIters = productMixed(builder, targetDimsForProd, `slb_tot_${op.id}`);
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);

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
        const builder = new GraphBuilder(body, `slloop_${op.id}`);
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;
        const k = toStaticShape(ins[1].shape)[0];

        const canHoist =
            loopCtx !== undefined &&
            ins.every(
                (inp) =>
                    !isPresent(inp) ||
                    (!valueMap.has(inp.id) && loopCtx.outerBuilder.graph.hasNode(inp.id)),
            );

        const paramBuilder = canHoist ? loopCtx!.outerBuilder : builder;
        const rawData = canHoist
            ? (ins[0] as ConcreteValueNode)
            : resolveRecipeInput(
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
        const rawStarts = canHoist
            ? (ins[1] as ConcreteValueNode)
            : resolveRecipeInput(
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
        const rawEnds = canHoist
            ? (ins[2] as ConcreteValueNode)
            : resolveRecipeInput(
                  builder,
                  ins[2],
                  valueMap,
                  iter,
                  axes,
                  outShape,
                  false,
                  false,
                  targetShapeNode,
                  loopCtx,
              );
        const rawAxes = isPresent(ins[3])
            ? canHoist
                ? (ins[3] as ConcreteValueNode)
                : resolveRecipeInput(
                      builder,
                      ins[3],
                      valueMap,
                      iter,
                      axes,
                      outShape,
                      false,
                      false,
                      targetShapeNode,
                      loopCtx,
                  )
            : undefined;
        const rawSteps = isPresent(ins[4])
            ? canHoist
                ? (ins[4] as ConcreteValueNode)
                : resolveRecipeInput(
                      builder,
                      ins[4],
                      valueMap,
                      iter,
                      axes,
                      outShape,
                      false,
                      false,
                      targetShapeNode,
                      loopCtx,
                  )
            : undefined;

        const { shapeData, effStarts, fullSteps } = this.normalizeSliceParams(
            paramBuilder,
            rawData,
            rawStarts,
            rawEnds,
            rawAxes,
            rawSteps,
            rank,
            k,
            `sli_${op.id}`,
        );

        const paramTargetShape = canHoist
            ? (targetShapeNode! as ConcreteValueNode)
            : ensureInnerValueNode(paramBuilder, targetShapeNode!);
        const flatDataOuter = ensureFlatInput(paramBuilder, rawData);

        const gatherElem = (
            b: GraphBuilder,
            vec: ValueNode,
            idx: number,
            tag: string,
        ): ConcreteValueNode => {
            const idxC = b.createConstant(`${tag}_${idx}`, scalarInt64(idx));
            return gatherScalar(b, vec, idxC);
        };

        const rawInDims: ConcreteValueNode[] = [];
        const rawOutDims: ConcreteValueNode[] = [];
        const rawStartDims: ConcreteValueNode[] = [];
        const rawStepDims: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            rawInDims.push(gatherElem(paramBuilder, shapeData, d, `sli_id_${op.id}`));
            rawOutDims.push(gatherElem(paramBuilder, paramTargetShape, d, `sli_od_${op.id}`));
            rawStartDims.push(gatherElem(paramBuilder, effStarts, d, `sli_st_${op.id}`));
            rawStepDims.push(gatherElem(paramBuilder, fullSteps, d, `sli_sp_${op.id}`));
        }

        const rawInStrides: (number | ConcreteValueNode)[] = new Array(rank).fill(1);
        let curStride: number | ConcreteValueNode = 1;
        for (let d = rank - 1; d >= 0; d--) {
            rawInStrides[d] = curStride;
            if (d > 0) {
                curStride = mulMixed(
                    paramBuilder,
                    curStride,
                    rawInDims[d],
                    `sli_str_${d}_${op.id}`,
                );
            }
        }

        const flatData = canHoist ? ensureInnerValueNode(builder, flatDataOuter) : flatDataOuter;
        const outDims = (
            canHoist ? captureMixedList(builder, rawOutDims) : rawOutDims
        ) as ConcreteValueNode[];
        const startDims = (
            canHoist ? captureMixedList(builder, rawStartDims) : rawStartDims
        ) as ConcreteValueNode[];
        const stepDims = (
            canHoist ? captureMixedList(builder, rawStepDims) : rawStepDims
        ) as ConcreteValueNode[];
        const inStrides = canHoist ? captureMixedList(builder, rawInStrides) : rawInStrides;

        const outCoords: ConcreteValueNode[] =
            loopCtx && loopCtx.indexingMode === IndexingMode.Pointer
                ? getOrCreateIterationSpaceCounters(
                      builder,
                      loopCtx,
                      valueMap,
                      outDims,
                      `sli_cnt_${op.id}`,
                  ).counters
                : decodeMixedRadix(builder, iter, outDims, `sli_dec_${op.id}`);

        const inCoords: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            const [scaled] = builder.createOp("Mul", [outCoords[d], stepDims[d]]);
            const [coord] = builder.createOp("Add", [startDims[d], scaled]);
            inCoords.push(coord);
        }

        const flatInIdx = buildLinearIndex(builder, inCoords, inStrides, `sli_lin_${op.id}`);
        return gatherScalar(builder, flatData, flatInIdx);
    }
}
