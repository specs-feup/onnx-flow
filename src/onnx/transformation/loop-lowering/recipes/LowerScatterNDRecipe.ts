import type OnnxGraph from "../../../ir/OnnxGraph.js";
import type OperationNode from "../../../ir/OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../ir/OnnxTypes.js";
import { getStringAttr, int64Vec, scalarInt64, toStaticShape } from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    buildMixedStrides,
    decodeMixedRadix,
    ensureFlatInput,
    extractMixedDims,
    gatherScalar,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../ir/GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerScatterNDRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerScatterND";
    public readonly targetOp = "ScatterND";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["indexing", "self-carry"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "ReduceProd",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Max",
        "Min",
        "Equal",
        "Less",
        "And",
        "Not",
        "Where",
        "Cast",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "ScatterND") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        const dataDims = toStaticShape(ins[0].shape);
        const idxDims = toStaticShape(ins[1].shape);
        if (!Array.isArray(ins[2].shape) || dataDims.length === 0 || idxDims.length === 0) {
            return false;
        }
        const updDims = toStaticShape(ins[2].shape);
        const k = idxDims[idxDims.length - 1];
        return (
            k >= 1 &&
            k <= dataDims.length &&
            updDims.length === idxDims.length - 1 + (dataDims.length - k)
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
        const ins = op.getInputs()!;
        const dataDims = toStaticShape(ins[0].shape);
        const updDims = toStaticShape(ins[2].shape);
        if (!dataDims.includes(-1) && !updDims.includes(-1)) {
            const dataTotal = dataDims.reduce((a, b) => a * b, 1);
            const updTotal = updDims.length === 0 ? 1 : updDims.reduce((a, b) => a * b, 1);
            return { totalIters: dataTotal + updTotal, carryShape: dataDims };
        }

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `snd_bounds_${op.id}`);
        const ax0 = builder.createConstant(`sndb_ax0_${op.id}`, int64Vec([0]));
        const [dataShape] = builder.createOp("Shape", [ins[0]], {}, [
            { type: DataType.INT64, shape: [dataDims.length] },
        ]);
        const [dataTotal] = builder.createOp("ReduceProd", [dataShape, ax0], { keepdims: 0 });
        let updTotal: ConcreteValueNode;
        if (updDims.length === 0) {
            updTotal = builder.createConstant(`sndb_utot1_${op.id}`, scalarInt64(1));
        } else if (!updDims.includes(-1)) {
            updTotal = builder.createConstant(
                `sndb_utot_${op.id}`,
                scalarInt64(updDims.reduce((a, b) => a * b, 1)),
            );
        } else {
            const [updShape] = builder.createOp("Shape", [ins[2]], {}, [
                { type: DataType.INT64, shape: [updDims.length] },
            ]);
            [updTotal] = builder.createOp("ReduceProd", [updShape, ax0], { keepdims: 0 });
        }
        const [totalIters] = builder.createOp("Add", [dataTotal, updTotal]);
        const [carryShape] = builder.createOp("Unsqueeze", [dataTotal, ax0]);
        return { totalIters, carryShape, targetShape: dataShape };
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
        const builder = new GraphBuilder(body, `snd_${op.id}`);
        const ins = op.getInputs()!;
        const rawDataDims = toStaticShape(ins[0].shape);
        const rawIdxDims = toStaticShape(ins[1].shape);
        const rawUpdDims = toStaticShape(ins[2].shape);
        const q = rawIdxDims.length;
        const k = rawIdxDims[q - 1];
        const reduction = getStringAttr(op, "reduction", "none");

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
        const rawIndices = resolveRecipeInput(
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
        const rawUpdates = resolveRecipeInput(
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

        const dataDims = extractMixedDims(builder, rawData, rawDataDims, `snd_dd_${op.id}`);
        const idxDims = extractMixedDims(builder, rawIndices, rawIdxDims, `snd_id_${op.id}`);
        const updDims = extractMixedDims(builder, rawUpdates, rawUpdDims, `snd_ud_${op.id}`);
        const dataStrides = buildMixedStrides(builder, dataDims);
        const idxStrides = buildMixedStrides(builder, idxDims);

        const flatData = ensureFlatInput(builder, rawData);
        const flatIndices = ensureFlatInput(builder, rawIndices);
        const flatUpdates = ensureFlatInput(builder, rawUpdates);

        const zeroI64 = builder.createConstant(`snd_z64_${op.id}`, scalarInt64(0));
        let dataTotalNode: ConcreteValueNode;
        if (!rawDataDims.includes(-1)) {
            const dataTotal = rawDataDims.reduce((a, b) => a * b, 1);
            dataTotalNode = builder.createConstant(`snd_dtot_${op.id}`, scalarInt64(dataTotal));
        } else {
            const d0 =
                typeof dataDims[0] === "number"
                    ? builder.createConstant(`snd_d0_${op.id}`, scalarInt64(dataDims[0]))
                    : dataDims[0];
            const s0 =
                typeof dataStrides[0] === "number"
                    ? builder.createConstant(`snd_s0_${op.id}`, scalarInt64(dataStrides[0]))
                    : dataStrides[0];
            [dataTotalNode] = builder.createOp("Mul", [d0, s0]);
        }

        // Phase check: copy data when iter < dataTotal, else scatter updates
        let curIter: ConcreteValueNode = iter;
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const initPtr = loopCtx.outerBuilder.createConstant(
                `snd_init_it_${op.id}`,
                scalarInt64(0),
            );
            const ptrIn = loopCtx.addCarry(`snd_it_${op.id}`, initPtr, DataType.INT64, []);
            const oneI64 = builder.createConstant(`snd_one_${op.id}`, scalarInt64(1));
            const [nextPtr] = builder.createOp("Add", [ptrIn, oneI64]);
            loopCtx.setNextState(ptrIn, nextPtr);
            curIter = ptrIn;
        }

        const [isCopyPhase] = builder.createOp("Less", [curIter, dataTotalNode]);
        const [safeCopyIdx] = builder.createOp("Where", [isCopyPhase, curIter, zeroI64]);

        const [rawUpdIter] = builder.createOp("Sub", [curIter, dataTotalNode]);
        const [safeUpdIter] = builder.createOp("Where", [isCopyPhase, zeroI64, rawUpdIter]);

        // Decode safeUpdIter across updDims = [...idxDims[0..q-2], ...dataDims[k..r-1]]
        let idxTupleBase: ConcreteValueNode = zeroI64;
        let scatterTargetIdx: ConcreteValueNode = zeroI64;

        if (updDims.length > 0) {
            const idxPrefixStrides: (number | ConcreteValueNode)[] = [
                ...idxStrides.slice(0, q - 1),
                ...new Array<number>(updDims.length - (q - 1)).fill(0),
            ];
            const trailingStrides: (number | ConcreteValueNode)[] = [
                ...new Array<number>(q - 1).fill(0),
                ...dataStrides.slice(k),
            ];

            if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
                const [isUpdPhase] = builder.createOp("Not", [isCopyPhase]);
                idxTupleBase = buildAguPointer(
                    builder,
                    loopCtx,
                    valueMap,
                    updDims,
                    idxPrefixStrides,
                    `snd_idx_${op.id}`,
                    0,
                    isUpdPhase,
                );
                scatterTargetIdx = buildAguPointer(
                    builder,
                    loopCtx,
                    valueMap,
                    updDims,
                    trailingStrides,
                    `snd_tgt_${op.id}`,
                    0,
                    isUpdPhase,
                );
            } else {
                const updCoords = decodeMixedRadix(
                    builder,
                    safeUpdIter,
                    updDims,
                    `snd_dec_${op.id}`,
                );
                idxTupleBase = buildLinearIndex(
                    builder,
                    updCoords,
                    idxPrefixStrides,
                    `snd_ilin_${op.id}`,
                );
                scatterTargetIdx = buildLinearIndex(
                    builder,
                    updCoords,
                    trailingStrides,
                    `snd_tlin_${op.id}`,
                );
            }
        }

        for (let j = 0; j < k; j++) {
            let elemIdxPos: ConcreteValueNode = idxTupleBase;
            if (j > 0) {
                const jConst = builder.createConstant(`snd_j_${j}_${op.id}`, scalarInt64(j));
                [elemIdxPos] = builder.createOp("Add", [idxTupleBase, jConst]);
            }
            let coord = gatherScalar(builder, flatIndices, elemIdxPos);
            if (ins[1].literalType !== DataType.INT64) {
                [coord] = builder.createOp("Cast", [coord], { to: DataType.INT64 });
            }

            const indexedDim = dataDims[j];
            const dimNode =
                typeof indexedDim === "number"
                    ? builder.createConstant(`snd_dim_${j}_${op.id}`, scalarInt64(indexedDim))
                    : indexedDim;
            const [isNeg] = builder.createOp("Less", [coord, zeroI64]);
            const [wrapped] = builder.createOp("Add", [coord, dimNode]);
            const [normCoord] = builder.createOp("Where", [isNeg, wrapped, coord]);

            const stride = dataStrides[j];
            const term =
                typeof stride !== "number"
                    ? builder.createOp("Mul", [normCoord, stride])[0]
                    : stride === 1
                      ? normCoord
                      : builder.createOp("Mul", [
                            normCoord,
                            builder.createConstant(`snd_str_${j}_${op.id}`, scalarInt64(stride)),
                        ])[0];
            [scatterTargetIdx] = builder.createOp("Add", [scatterTargetIdx, term]);
        }

        // Gather copy value and update value
        const copyVal = gatherScalar(builder, flatData, safeCopyIdx);
        let updVal = gatherScalar(builder, flatUpdates, safeUpdIter);

        const [writeIdx] = builder.createOp("Where", [isCopyPhase, safeCopyIdx, scatterTargetIdx]);
        const [writeIdxUnsq] = builder.createOp("Unsqueeze", [writeIdx, axes]);

        if (reduction !== "none") {
            const curVal = gatherScalar(builder, carryNode, writeIdx);
            const redOp =
                reduction === "add"
                    ? "Add"
                    : reduction === "mul"
                      ? "Mul"
                      : reduction === "max"
                        ? "Max"
                        : "Min";
            [updVal] = builder.createOp(redOp, [curVal, updVal]);
        }

        const [finalVal] = builder.createOp("Where", [isCopyPhase, copyVal, updVal]);
        const [finalValUnsq] = builder.createOp("Unsqueeze", [finalVal, axes]);
        const [nextCarry] = builder.createOp(
            "ScatterElements",
            [carryNode, writeIdxUnsq, finalValUnsq],
            {
                axis: 0,
            },
        );

        return { resultNode: finalVal, nextCarry };
    }
}
