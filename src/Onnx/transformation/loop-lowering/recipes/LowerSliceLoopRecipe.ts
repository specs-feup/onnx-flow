import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import { int64Vec, scalarInt64, toStaticShape } from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    ensureInnerValueNode,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

const isPresent = (n?: ValueNode): n is ValueNode => n !== undefined && n.id !== "";

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
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Min",
        "Max",
        "Less",
        "Greater",
        "Equal",
        "And",
        "Where",
        "Cast",
        "ReduceProd",
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

        // Positive step: clamp to [0, dim]
        const [sClampPos] = builder.createOp("Min", [
            builder.createOp("Max", [sWrap, zeroSc])[0],
            shapeData,
        ]);
        const [eClampPos] = builder.createOp("Min", [
            builder.createOp("Max", [eWrap, zeroSc])[0],
            shapeData,
        ]);

        // Negative step: clamp to [-1, dim - 1]
        const [sClampNeg] = builder.createOp("Min", [
            builder.createOp("Max", [sWrap, negOneSc])[0],
            maxIdx,
        ]);
        const [eClampNeg] = builder.createOp("Min", [
            builder.createOp("Max", [eWrap, negOneSc])[0],
            maxIdx,
        ]);

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
        const oneSc = builder.createConstant(`slb_o_${op.id}`, scalarInt64(1));
        const negOneSc = builder.createConstant(`slb_m1_${op.id}`, scalarInt64(-1));

        // Positive step: ceil(max(0, effEnds - effStarts) / fullSteps)
        const [spanPos] = builder.createOp("Max", [
            builder.createOp("Sub", [effEnds, effStarts])[0],
            zeroSc,
        ]);
        const [posDenom] = builder.createOp("Where", [posStep, fullSteps, oneSc]);
        const [dimPos] = builder.createOp("Div", [
            builder.createOp("Sub", [builder.createOp("Add", [spanPos, posDenom])[0], oneSc])[0],
            posDenom,
        ]);

        // Negative step: ceil(max(0, effStarts - effEnds) / (-fullSteps))
        const [spanNeg] = builder.createOp("Max", [
            builder.createOp("Sub", [effStarts, effEnds])[0],
            zeroSc,
        ]);
        const [negSteps] = builder.createOp("Mul", [fullSteps, negOneSc]);
        const [negDenom] = builder.createOp("Where", [posStep, oneSc, negSteps]);
        const [dimNeg] = builder.createOp("Div", [
            builder.createOp("Sub", [builder.createOp("Add", [spanNeg, negDenom])[0], oneSc])[0],
            negDenom,
        ]);

        const [targetShape] = builder.createOp("Where", [posStep, dimPos, dimNeg], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);

        const axes0 = builder.createConstant(`slb_ax0_${op.id}`, int64Vec([0]));
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, axes0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0]);

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

        const rawData = resolveRecipeInput(
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
        const rawStarts = resolveRecipeInput(
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
        const rawEnds = resolveRecipeInput(
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
            ? resolveRecipeInput(
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
            ? resolveRecipeInput(
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
            builder,
            rawData,
            rawStarts,
            rawEnds,
            rawAxes,
            rawSteps,
            rank,
            k,
            `sli_${op.id}`,
        );

        const innerTargetShape = ensureInnerValueNode(builder, targetShapeNode!);
        const flatData = ensureFlatInput(builder, rawData);

        const gatherElem = (vec: ValueNode, idx: number, tag: string): ConcreteValueNode => {
            const idxC = builder.createConstant(`${tag}_${idx}`, int64Vec([idx]));
            const [raw] = builder.createOp("Gather", [vec, idxC], { axis: 0 });
            return squeezeIfLen1(builder, raw, axes, `${tag}_sq_${idx}`) as ConcreteValueNode;
        };

        const inDims: ConcreteValueNode[] = [];
        const outDims: ConcreteValueNode[] = [];
        const startDims: ConcreteValueNode[] = [];
        const stepDims: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            inDims.push(gatherElem(shapeData, d, `sli_id_${op.id}`));
            outDims.push(gatherElem(innerTargetShape, d, `sli_od_${op.id}`));
            startDims.push(gatherElem(effStarts, d, `sli_st_${op.id}`));
            stepDims.push(gatherElem(fullSteps, d, `sli_sp_${op.id}`));
        }

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

        const inStrides: (number | ValueNode)[] = new Array(rank).fill(1);
        let curStride: number | ValueNode = 1;
        for (let d = rank - 1; d >= 0; d--) {
            inStrides[d] = curStride;
            if (d > 0) {
                curStride =
                    typeof curStride === "number"
                        ? inDims[d]
                        : builder.createOp("Mul", [curStride, inDims[d]])[0];
            }
        }

        const flatInIdx = buildLinearIndex(builder, inCoords, inStrides, `sli_lin_${op.id}`);
        const [idxUnsq] = builder.createOp("Unsqueeze", [flatInIdx, axes]);
        const [gathered] = builder.createOp("Gather", [flatData, idxUnsq], { axis: 0 });
        return squeezeIfLen1(builder, gathered, axes, `sli_sq_${op.id}`);
    }
}
