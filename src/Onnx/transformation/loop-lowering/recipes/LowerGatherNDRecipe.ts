import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import { computeStrides, getIntAttr, scalarInt64, toStaticShape } from "../../../Utils.js";
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
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerGatherNDRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerGatherND";
    public readonly targetOp = "GatherND";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
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
        if (dataDims.includes(-1) || idxDims.includes(-1)) return false;
        const k = idxDims[idxDims.length - 1];
        return k >= 1 && batchDims + k <= dataDims.length && batchDims < idxDims.length;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const ins = op.getInputs()!;
        const dataDims = toStaticShape(ins[0].shape);
        const idxDims = toStaticShape(ins[1].shape);
        const batchDims = getIntAttr(op, "batch_dims", 0);
        const k = idxDims[idxDims.length - 1];
        const outDims = [...idxDims.slice(0, -1), ...dataDims.slice(batchDims + k)];
        const total = outDims.reduce((a, b) => a * b, 1);
        return { totalIters: total, carryShape: outDims };
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
        const dataDims = toStaticShape(ins[0].shape);
        const idxDims = toStaticShape(ins[1].shape);
        const batchDims = getIntAttr(op, "batch_dims", 0);
        const q = idxDims.length;
        const k = idxDims[q - 1];

        const outDims = [...idxDims.slice(0, q - 1), ...dataDims.slice(batchDims + k)];
        const outRank = outDims.length;

        const dataStrides = computeStrides(dataDims);
        const idxStrides = computeStrides(idxDims);

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
        const flatData = ensureFlatInput(builder, rawData);
        const flatIndices = ensureFlatInput(builder, rawIndices);

        // Strides over outDims for:
        // 1) baseDataPtr: batch dims [0..b-1] + trailing slice dims [q-1..outRank-1]
        // 2) idxTupleBase: index prefix dims [0..q-2]
        const baseDataStrides = new Array<number>(outRank).fill(0);
        for (let m = 0; m < batchDims; m++) {
            baseDataStrides[m] = dataStrides[m];
        }
        for (let s = 0; s < dataDims.length - (batchDims + k); s++) {
            baseDataStrides[q - 1 + s] = dataStrides[batchDims + k + s];
        }

        const idxBaseStrides = new Array<number>(outRank).fill(0);
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

            const [posUnsq] = builder.createOp("Unsqueeze", [elemIdxPos, axes]);
            const [rawCoord] = builder.createOp("Gather", [flatIndices, posUnsq], { axis: 0 });
            let coord = squeezeIfLen1(
                builder,
                rawCoord,
                axes,
                `gnd_sq_c_${j}_${op.id}`,
            ) as ConcreteValueNode;

            if (ins[1].literalType !== DataType.INT64) {
                [coord] = builder.createOp("Cast", [coord], { to: DataType.INT64 });
            }

            // Wrap negative index against dataDims[batchDims + j]
            const dimConst = builder.createConstant(
                `gnd_dim_${j}_${op.id}`,
                scalarInt64(dataDims[batchDims + j]),
            );
            const [isNeg] = builder.createOp("Less", [coord, zeroI64]);
            const [wrapped] = builder.createOp("Add", [coord, dimConst]);
            const [normCoord] = builder.createOp("Where", [isNeg, wrapped, coord]);

            const stride = dataStrides[batchDims + j];
            let term: ConcreteValueNode = normCoord;
            if (stride !== 1) {
                const strideConst = builder.createConstant(
                    `gnd_str_${j}_${op.id}`,
                    scalarInt64(stride),
                );
                [term] = builder.createOp("Mul", [normCoord, strideConst]);
            }
            [dataIdx] = builder.createOp("Add", [dataIdx, term]);
        }

        const [dataIdxUnsq] = builder.createOp("Unsqueeze", [dataIdx, axes]);
        const [gathered] = builder.createOp("Gather", [flatData, dataIdxUnsq], { axis: 0 });
        return squeezeIfLen1(builder, gathered, axes, `gnd_sq_out_${op.id}`);
    }
}
