import type OnnxGraph from "../../../ir/OnnxGraph.js";
import type OperationNode from "../../../ir/OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../ir/OnnxTypes.js";
import { getIntAttr, int64Vec, scalarInt64, toStaticShape } from "../../../utils/index.js";
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

export class LowerGatherNDRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerGatherND";
    public readonly targetOp = "GatherND";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["indexing", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Concat",
        "ReduceProd",
        "Reshape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Mul",
        "Less",
        "Equal",
        "And",
        "Where",
        "Cast",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "GatherND") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const dataDims = toStaticShape(ins[0].shape);
        const idxDims = toStaticShape(ins[1].shape);
        const batchDims = getIntAttr(op, "batch_dims", 0);
        if (dataDims.length === 0 || idxDims.length === 0) return false;
        const k = idxDims[idxDims.length - 1];
        return k >= 1 && batchDims + k <= dataDims.length && batchDims < idxDims.length;
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
        const idxDims = toStaticShape(ins[1].shape);
        const batchDims = getIntAttr(op, "batch_dims", 0);
        const q = idxDims.length;
        const k = idxDims[q - 1];
        const outDims = [...idxDims.slice(0, q - 1), ...dataDims.slice(batchDims + k)];
        if (!outDims.includes(-1)) {
            const total = outDims.length === 0 ? 1 : outDims.reduce((a, b) => a * b, 1);
            return { totalIters: total, carryShape: outDims };
        }

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `gnd_bounds_${op.id}`);
        const ax0 = builder.createConstant(`gndb_ax0_${op.id}`, int64Vec([0]));
        const parts: ConcreteValueNode[] = [];
        if (q - 1 > 0) {
            const [idxShape] = builder.createOp("Shape", [ins[1]], {}, [
                { type: DataType.INT64, shape: [q] },
            ]);
            const prefixIdx = builder.createConstant(
                `gndb_pidx_${op.id}`,
                int64Vec(Array.from({ length: q - 1 }, (_, i) => i)),
            );
            const [pPart] = builder.createOp("Gather", [idxShape, prefixIdx], { axis: 0 });
            parts.push(pPart);
        }
        const sliceLen = dataDims.length - (batchDims + k);
        if (sliceLen > 0) {
            const [dataShape] = builder.createOp("Shape", [ins[0]], {}, [
                { type: DataType.INT64, shape: [dataDims.length] },
            ]);
            const suffixIdx = builder.createConstant(
                `gndb_sidx_${op.id}`,
                int64Vec(Array.from({ length: sliceLen }, (_, i) => batchDims + k + i)),
            );
            const [sPart] = builder.createOp("Gather", [dataShape, suffixIdx], { axis: 0 });
            parts.push(sPart);
        }
        const targetShape =
            parts.length === 1 ? parts[0] : builder.createOp("Concat", parts, { axis: 0 })[0];
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, ax0]);
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
        const builder = new GraphBuilder(body, `gnd_${op.id}`);
        const ins = op.getInputs()!;
        const rawDataDims = toStaticShape(ins[0].shape);
        const rawIdxDims = toStaticShape(ins[1].shape);
        const batchDims = getIntAttr(op, "batch_dims", 0);
        const q = rawIdxDims.length;
        const k = rawIdxDims[q - 1];

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
        const dataDims = extractMixedDims(builder, rawData, rawDataDims, `gnd_dd_${op.id}`);
        const idxDims = extractMixedDims(builder, rawIndices, rawIdxDims, `gnd_id_${op.id}`);
        const outDims = [...idxDims.slice(0, q - 1), ...dataDims.slice(batchDims + k)];
        const outRank = outDims.length;

        const dataStrides = buildMixedStrides(builder, dataDims);
        const idxStrides = buildMixedStrides(builder, idxDims);
        const flatData = ensureFlatInput(builder, rawData);
        const flatIndices = ensureFlatInput(builder, rawIndices);

        // Strides over outDims for:
        // 1) baseDataPtr: batch dims [0..b-1] + trailing slice dims [q-1..outRank-1]
        // 2) idxTupleBase: index prefix dims [0..q-2]
        const baseDataStrides: (number | ConcreteValueNode)[] = new Array(outRank).fill(0);
        for (let m = 0; m < batchDims; m++) {
            baseDataStrides[m] = dataStrides[m];
        }
        for (let s = 0; s < rawDataDims.length - (batchDims + k); s++) {
            baseDataStrides[q - 1 + s] = dataStrides[batchDims + k + s];
        }

        const idxBaseStrides: (number | ConcreteValueNode)[] = new Array(outRank).fill(0);
        for (let i = 0; i < q - 1; i++) {
            idxBaseStrides[i] = idxStrides[i];
        }

        let dataIdx: ConcreteValueNode;
        let idxTupleBase: ConcreteValueNode;

        if (outRank === 0) {
            dataIdx = builder.createConstant(`gnd_zdata_${op.id}`, scalarInt64(0));
            idxTupleBase = builder.createConstant(`gnd_zidx_${op.id}`, scalarInt64(0));
        } else if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            dataIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                outDims,
                baseDataStrides,
                `gnd_dbase_${op.id}`,
            );
            idxTupleBase = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                outDims,
                idxBaseStrides,
                `gnd_ibase_${op.id}`,
            );
        } else {
            const coords = decodeMixedRadix(builder, iter, outDims, `gnd_dec_${op.id}`);
            dataIdx = buildLinearIndex(builder, coords, baseDataStrides, `gnd_dlin_${op.id}`);
            idxTupleBase = buildLinearIndex(builder, coords, idxBaseStrides, `gnd_ilin_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`gnd_z64_${op.id}`, scalarInt64(0));

        // Accumulate each of the k indexed coordinates into dataIdx
        for (let j = 0; j < k; j++) {
            let elemIdxPos: ConcreteValueNode = idxTupleBase;
            if (j > 0) {
                const jConst = builder.createConstant(`gnd_j_${j}_${op.id}`, scalarInt64(j));
                [elemIdxPos] = builder.createOp("Add", [idxTupleBase, jConst]);
            }

            let coord = gatherScalar(builder, flatIndices, elemIdxPos);

            if (ins[1].literalType !== DataType.INT64) {
                [coord] = builder.createOp("Cast", [coord], { to: DataType.INT64 });
            }

            // Wrap negative index against dataDims[batchDims + j]
            const indexedDim = dataDims[batchDims + j];
            const dimNode =
                typeof indexedDim === "number"
                    ? builder.createConstant(`gnd_dim_${j}_${op.id}`, scalarInt64(indexedDim))
                    : indexedDim;
            const [isNeg] = builder.createOp("Less", [coord, zeroI64]);
            const [wrapped] = builder.createOp("Add", [coord, dimNode]);
            const [normCoord] = builder.createOp("Where", [isNeg, wrapped, coord]);

            const stride = dataStrides[batchDims + j];
            let term: ConcreteValueNode = normCoord;
            if (typeof stride !== "number") {
                [term] = builder.createOp("Mul", [normCoord, stride]);
            } else if (stride !== 1) {
                const strideConst = builder.createConstant(
                    `gnd_str_${j}_${op.id}`,
                    scalarInt64(stride),
                );
                [term] = builder.createOp("Mul", [normCoord, strideConst]);
            }
            [dataIdx] = builder.createOp("Add", [dataIdx, term]);
        }

        return gatherScalar(builder, flatData, dataIdx);
    }
}
