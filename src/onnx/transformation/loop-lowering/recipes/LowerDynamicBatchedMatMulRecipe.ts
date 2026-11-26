import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import { int64Vec, scalarInt64 } from "../../../utils/index.js";
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
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerDynamicBatchedMatMulRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerDynamicBatchedMatMul";
    public readonly targetOp = "MatMul";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["matmul", "dynamic-batch", "self-carry"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Concat",
        "Unsqueeze",
        "Squeeze",
        "ReduceProd",
        "Max",
        "Add",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "And",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "MatMul") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const shapeA = ins[0].shape;
        const shapeB = ins[1].shape;
        return (
            Array.isArray(shapeA) &&
            Array.isArray(shapeB) &&
            shapeA.length >= 2 &&
            shapeB.length >= 2
        );
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
        const rankA = ins[0].shape.length;
        const rankB = ins[1].shape.length;
        const batchRankA = rankA - 2;
        const batchRankB = rankB - 2;
        const outBatchRank = Math.max(batchRankA, batchRankB);
        const outRank = outBatchRank + 2;

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `dbmm_bounds_${op.id}`);
        const [shapeA] = builder.createOp("Shape", [ins[0]], {}, [
            { type: DataType.INT64, shape: [rankA] },
        ]);
        const [shapeB] = builder.createOp("Shape", [ins[1]], {}, [
            { type: DataType.INT64, shape: [rankB] },
        ]);

        const outDimParts: ConcreteValueNode[] = [];
        for (let i = 0; i < outBatchRank; i++) {
            const idxA = i - (outBatchRank - batchRankA);
            const idxB = i - (outBatchRank - batchRankB);

            const dimA =
                idxA >= 0
                    ? builder.createOp(
                          "Gather",
                          [
                              shapeA,
                              builder.createConstant(`dbmmb_a_${op.id}_${i}`, int64Vec([idxA])),
                          ],
                          { axis: 0 },
                          [{ type: DataType.INT64, shape: [1] }],
                      )[0]
                    : undefined;
            const dimB =
                idxB >= 0
                    ? builder.createOp(
                          "Gather",
                          [
                              shapeB,
                              builder.createConstant(`dbmmb_b_${op.id}_${i}`, int64Vec([idxB])),
                          ],
                          { axis: 0 },
                          [{ type: DataType.INT64, shape: [1] }],
                      )[0]
                    : undefined;

            if (dimA && dimB) {
                outDimParts.push(
                    builder.createOp("Max", [dimA, dimB], {}, [
                        { type: DataType.INT64, shape: [1] },
                    ])[0],
                );
            } else {
                outDimParts.push((dimA ?? dimB)!);
            }
        }

        const [dimM] = builder.createOp(
            "Gather",
            [shapeA, builder.createConstant(`dbmmb_m_${op.id}`, int64Vec([rankA - 2]))],
            { axis: 0 },
            [{ type: DataType.INT64, shape: [1] }],
        );
        const [dimK] = builder.createOp(
            "Gather",
            [shapeA, builder.createConstant(`dbmmb_k_${op.id}`, int64Vec([rankA - 1]))],
            { axis: 0 },
            [{ type: DataType.INT64, shape: [1] }],
        );
        const [dimN] = builder.createOp(
            "Gather",
            [shapeB, builder.createConstant(`dbmmb_n_${op.id}`, int64Vec([rankB - 1]))],
            { axis: 0 },
            [{ type: DataType.INT64, shape: [1] }],
        );

        outDimParts.push(dimM, dimN);
        const [targetShape] = builder.createOp("Concat", outDimParts, { axis: 0 }, [
            { type: DataType.INT64, shape: [outRank] },
        ]);

        const ax0 = builder.createConstant(`dbmmb_ax0_${op.id}`, int64Vec([0]));
        const [carryLen] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [carryLen, ax0]);
        const [kScalar] = builder.createOp("Squeeze", [dimK, ax0]);
        const [totalIters] = builder.createOp("Mul", [carryLen, kScalar]);

        return { totalIters, carryShape, targetShape };
    }

    private extractScalarDims(
        builder: GraphBuilder,
        shapeNode: ConcreteValueNode,
        rank: number,
        axes: ConcreteValueNode,
        prefix: string,
    ): ConcreteValueNode[] {
        const dims: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            const dIdx = builder.createConstant(`${prefix}_idx_${d}`, scalarInt64(d));
            dims.push(gatherScalar(builder, shapeNode, dIdx));
        }
        return dims;
    }

    private buildDynamicStrides(
        builder: GraphBuilder,
        dims: ConcreteValueNode[],
    ): (number | ValueNode)[] {
        const rank = dims.length;
        const strides: (number | ValueNode)[] = new Array(rank).fill(1);
        let cur: number | ValueNode = 1;
        for (let d = rank - 1; d >= 0; d--) {
            strides[d] = cur;
            if (d > 0) {
                cur =
                    typeof cur === "number" ? dims[d] : builder.createOp("Mul", [cur, dims[d]])[0];
            }
        }
        return strides;
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
        const builder = new GraphBuilder(body, `dbmm_${op.id}`);
        const ins = op.getInputs()! as ConcreteValueNode[];
        const rankA = ins[0].shape.length;
        const rankB = ins[1].shape.length;
        const batchRankA = rankA - 2;
        const batchRankB = rankB - 2;
        const outBatchRank = Math.max(batchRankA, batchRankB);
        const outRank = outBatchRank + 2;

        const rawA = resolveRecipeInput(
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
        const rawB = resolveRecipeInput(
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

        const flatA = ensureFlatInput(builder, rawA);
        const flatB = ensureFlatInput(builder, rawB);

        const [shapeA] = builder.createOp("Shape", [rawA], {}, [
            { type: DataType.INT64, shape: [rankA] },
        ]);
        const [shapeB] = builder.createOp("Shape", [rawB], {}, [
            { type: DataType.INT64, shape: [rankB] },
        ]);
        const innerTargetShape = ensureInnerValueNode(builder, targetShapeNode!);

        const dimsA = this.extractScalarDims(builder, shapeA, rankA, axes, `dbmm_da_${op.id}`);
        const dimsB = this.extractScalarDims(builder, shapeB, rankB, axes, `dbmm_db_${op.id}`);
        const outDims = this.extractScalarDims(
            builder,
            innerTargetShape as ConcreteValueNode,
            outRank,
            axes,
            `dbmm_do_${op.id}`,
        );
        const kDim = dimsA[rankA - 1];
        const iterDims = [...outDims, kDim];

        const coords: ConcreteValueNode[] =
            loopCtx && loopCtx.indexingMode === IndexingMode.Pointer
                ? getOrCreateIterationSpaceCounters(
                      builder,
                      loopCtx,
                      valueMap,
                      iterDims,
                      `dbmm_cnt_${op.id}`,
                  ).counters
                : decodeMixedRadix(builder, iter, iterDims, `dbmm_dec_${op.id}`);

        const batchCoords = coords.slice(0, outBatchRank);
        const mCoord = coords[outBatchRank];
        const nCoord = coords[outBatchRank + 1];
        const kCoord = coords[outBatchRank + 2];

        const zeroI64 = builder.createConstant(`dbmm_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`dbmm_o64_${op.id}`, scalarInt64(1));

        // 1. Flat output index across [...outBatch, M, N]
        const outStrides = this.buildDynamicStrides(builder, outDims);
        const flatOutIdx = buildLinearIndex(
            builder,
            coords.slice(0, outRank),
            outStrides,
            `dbmm_olin_${op.id}`,
        );

        // 2. Flat A index across [...batchA, M, K] with runtime broadcast guard
        const coordsA: ConcreteValueNode[] = [];
        for (let j = 0; j < batchRankA; j++) {
            const outIdx = j + (outBatchRank - batchRankA);
            const [isBcast] = builder.createOp("Equal", [dimsA[j], oneI64]);
            const [effCoord] = builder.createOp("Where", [isBcast, zeroI64, batchCoords[outIdx]]);
            coordsA.push(effCoord);
        }
        coordsA.push(mCoord, kCoord);
        const stridesA = this.buildDynamicStrides(builder, dimsA);
        const flatAIdx = buildLinearIndex(builder, coordsA, stridesA, `dbmm_alin_${op.id}`);

        // 3. Flat B index across [...batchB, K, N] with runtime broadcast guard
        const coordsB: ConcreteValueNode[] = [];
        for (let j = 0; j < batchRankB; j++) {
            const outIdx = j + (outBatchRank - batchRankB);
            const [isBcast] = builder.createOp("Equal", [dimsB[j], oneI64]);
            const [effCoord] = builder.createOp("Where", [isBcast, zeroI64, batchCoords[outIdx]]);
            coordsB.push(effCoord);
        }
        coordsB.push(kCoord, nCoord);
        const stridesB = this.buildDynamicStrides(builder, dimsB);
        const flatBIdx = buildLinearIndex(builder, coordsB, stridesB, `dbmm_blin_${op.id}`);

        // 4. Gather A, B and accumulate into self-carry output buffer
        const valA = gatherScalar(builder, flatA, flatAIdx);
        const valB = gatherScalar(builder, flatB, flatBIdx);

        const [prod] = builder.createOp("Mul", [valA, valB]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const prevAcc = gatherScalar(builder, carryNode, flatOutIdx);

        const [sumAcc] = builder.createOp("Add", [prevAcc, prod]);
        const [isFirstK] = builder.createOp("Equal", [kCoord, zeroI64]);
        const [newAcc] = builder.createOp("Where", [isFirstK, prod, sumAcc]);

        const [updUnsq] = builder.createOp("Unsqueeze", [newAcc, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, updUnsq], {
            axis: 0,
        });

        return { resultNode: newAcc, nextCarry };
    }
}
