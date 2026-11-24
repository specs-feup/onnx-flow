import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, normalizeAxis, scalarInt64 } from "../../../Utils.js";
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

export class LowerOneHotLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerOneHotLoop";
    public readonly targetOp = "OneHot";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["indexing", "dynamic-depth", "fuseable"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Gather",
        "Concat",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Mul",
        "Div",
        "Mod",
        "Less",
        "Equal",
        "And",
        "Where",
        "Cast",
        "ReduceProd",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "OneHot") return false;
        const ins = op.getInputs();
        return !!ins && ins.length >= 3 && Array.isArray(ins[0].shape);
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
        const idxRank = ins[0].shape.length;
        const outRank = idxRank + 1;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), outRank);
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `oh_bounds_${op.id}`);

        const depth64: ConcreteValueNode =
            ins[1].literalType === DataType.INT64
                ? (ins[1] as ConcreteValueNode)
                : builder.createOp("Cast", [ins[1]], { to: DataType.INT64 })[0];
        const shape1 = builder.createConstant(`ohb_s1_${op.id}`, int64Vec([1]));
        const [depth1D] = builder.createOp("Reshape", [depth64, shape1], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);

        let targetShape: ConcreteValueNode;
        if (idxRank === 0) {
            targetShape = depth1D;
        } else {
            const [shapeIdx] = builder.createOp("Shape", [ins[0]], {}, [
                { type: DataType.INT64, shape: [idxRank] },
            ]);
            const parts: ConcreteValueNode[] = [];
            if (axis > 0) {
                const leftIdx = builder.createConstant(
                    `ohb_l_${op.id}`,
                    int64Vec(Array.from({ length: axis }, (_, i) => i)),
                );
                parts.push(
                    builder.createOp("Gather", [shapeIdx, leftIdx], { axis: 0 }, [
                        { type: DataType.INT64, shape: [axis] },
                    ])[0],
                );
            }
            parts.push(depth1D);
            if (axis < idxRank) {
                const rightIdx = builder.createConstant(
                    `ohb_r_${op.id}`,
                    int64Vec(Array.from({ length: idxRank - axis }, (_, i) => axis + i)),
                );
                parts.push(
                    builder.createOp("Gather", [shapeIdx, rightIdx], { axis: 0 }, [
                        { type: DataType.INT64, shape: [idxRank - axis] },
                    ])[0],
                );
            }
            [targetShape] = builder.createOp("Concat", parts, { axis: 0 }, [
                { type: DataType.INT64, shape: [outRank] },
            ]);
        }

        const axes0 = builder.createConstant(`ohb_ax0_${op.id}`, int64Vec([0]));
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
        const builder = new GraphBuilder(body, `ohloop_${op.id}`);
        const ins = op.getInputs()!;
        const idxRank = ins[0].shape.length;
        const outRank = idxRank + 1;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), outRank);

        const rawIndices = resolveRecipeInput(
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
        const rawValues = resolveRecipeInput(
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

        const innerTargetShape = ensureInnerValueNode(builder, targetShapeNode!);
        const outDims: ConcreteValueNode[] = [];
        for (let d = 0; d < outRank; d++) {
            const dIdx = builder.createConstant(`ohi_d_${op.id}_${d}`, int64Vec([d]));
            const [rawD] = builder.createOp("Gather", [innerTargetShape, dIdx], { axis: 0 });
            outDims.push(
                squeezeIfLen1(builder, rawD, axes, `ohi_sqd_${op.id}_${d}`) as ConcreteValueNode,
            );
        }

        const outCoords: ConcreteValueNode[] =
            loopCtx && loopCtx.indexingMode === IndexingMode.Pointer
                ? getOrCreateIterationSpaceCounters(
                      builder,
                      loopCtx,
                      valueMap,
                      outDims,
                      `ohi_cnt_${op.id}`,
                  ).counters
                : decodeMixedRadix(builder, iter, outDims, `ohi_dec_${op.id}`);

        const axisCoord = outCoords[axis];
        const depthScalar = outDims[axis];

        let idxVal: ConcreteValueNode;
        if (idxRank === 0) {
            const scShape = builder.createConstant(`ohi_sc_${op.id}`, int64Vec([]));
            [idxVal] = builder.createOp("Reshape", [rawIndices, scShape]);
        } else {
            const flatIndices = ensureFlatInput(builder, rawIndices);
            const idxDims = outDims.filter((_, d) => d !== axis);
            const idxCoords = outCoords.filter((_, d) => d !== axis);

            const idxStrides: (number | ValueNode)[] = new Array(idxRank).fill(1);
            let curStride: number | ValueNode = 1;
            for (let d = idxRank - 1; d >= 0; d--) {
                idxStrides[d] = curStride;
                if (d > 0) {
                    curStride =
                        typeof curStride === "number"
                            ? idxDims[d]
                            : builder.createOp("Mul", [curStride, idxDims[d]])[0];
                }
            }

            const flatIdxPos = buildLinearIndex(builder, idxCoords, idxStrides, `ohi_lin_${op.id}`);
            const [posUnsq] = builder.createOp("Unsqueeze", [flatIdxPos, axes]);
            const [rawIdx] = builder.createOp("Gather", [flatIndices, posUnsq], { axis: 0 });
            idxVal = squeezeIfLen1(builder, rawIdx, axes, `ohi_sqi_${op.id}`) as ConcreteValueNode;
        }

        if (ins[0].literalType !== DataType.INT64) {
            [idxVal] = builder.createOp("Cast", [idxVal], { to: DataType.INT64 });
        }

        const zeroI64 = builder.createConstant(`ohi_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`ohi_o64_${op.id}`, scalarInt64(1));

        // Wrap negative indices: idx < 0 ? idx + depth : idx
        const [isNeg] = builder.createOp("Less", [idxVal, zeroI64]);
        const [wrapped] = builder.createOp("Add", [idxVal, depthScalar]);
        const [normIdx] = builder.createOp("Where", [isNeg, wrapped, idxVal]);

        const [isMatch] = builder.createOp("Equal", [axisCoord, normIdx]);
        const [offVal] = builder.createOp("Gather", [rawValues, zeroI64], { axis: 0 }, [
            { type: ins[2].literalType, shape: [] },
        ]);
        const [onVal] = builder.createOp("Gather", [rawValues, oneI64], { axis: 0 }, [
            { type: ins[2].literalType, shape: [] },
        ]);

        return builder.createOp("Where", [isMatch, onVal, offVal], {}, [
            { type: ins[2].literalType, shape: [] },
        ])[0];
    }
}
