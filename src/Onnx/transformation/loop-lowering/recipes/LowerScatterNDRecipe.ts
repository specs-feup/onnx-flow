import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import { computeStrides, getStringAttr, scalarInt64, toStaticShape } from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerScatterNDRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerScatterND";
    public readonly targetOp = "ScatterND";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
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
        "Max",
        "Min",
        "Less",
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
        const updDims = toStaticShape(ins[2].shape);
        if (
            dataDims.length === 0 ||
            idxDims.length === 0 ||
            dataDims.includes(-1) ||
            idxDims.includes(-1) ||
            updDims.includes(-1)
        ) {
            return false;
        }
        const k = idxDims[idxDims.length - 1];
        return k >= 1 && k <= dataDims.length;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const ins = op.getInputs()!;
        const dataDims = toStaticShape(ins[0].shape);
        const dataTotal = dataDims.reduce((a, b) => a * b, 1);
        const updTotal = toStaticShape(ins[2].shape).reduce((a, b) => a * b, 1);
        return { totalIters: dataTotal + updTotal, carryShape: dataDims };
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
        const dataDims = toStaticShape(ins[0].shape);
        const idxDims = toStaticShape(ins[1].shape);
        const updDims = toStaticShape(ins[2].shape);
        const q = idxDims.length;
        const k = idxDims[q - 1];
        const reduction = getStringAttr(op, "reduction", "none");

        const dataTotal = dataDims.reduce((a, b) => a * b, 1);
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

        const flatData = ensureFlatInput(builder, rawData);
        const flatIndices = ensureFlatInput(builder, rawIndices);
        const flatUpdates = ensureFlatInput(builder, rawUpdates);

        const zeroI64 = builder.createConstant(`snd_z64_${op.id}`, scalarInt64(0));
        const dataTotalConst = builder.createConstant(`snd_dtot_${op.id}`, scalarInt64(dataTotal));

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

        const [isCopyPhase] = builder.createOp("Less", [curIter, dataTotalConst]);
        const [safeCopyIdx] = builder.createOp("Where", [isCopyPhase, curIter, zeroI64]);

        const [rawUpdIter] = builder.createOp("Sub", [curIter, dataTotalConst]);
        const [safeUpdIter] = builder.createOp("Where", [isCopyPhase, zeroI64, rawUpdIter]);

        // Decode safeUpdIter across updDims = [...idxDims[0..q-2], ...dataDims[k..r-1]]
        let idxTupleBase: ConcreteValueNode = zeroI64;
        let scatterTargetIdx: ConcreteValueNode = zeroI64;

        if (updDims.length > 0) {
            const updCoords = decodeMixedRadix(builder, safeUpdIter, updDims, `snd_dec_${op.id}`);
            const idxPrefixStrides = [
                ...idxStrides.slice(0, q - 1),
                ...new Array<number>(updDims.length - (q - 1)).fill(0),
            ];
            idxTupleBase = buildLinearIndex(
                builder,
                updCoords,
                idxPrefixStrides,
                `snd_ilin_${op.id}`,
            );

            const trailingStrides = [...new Array<number>(q - 1).fill(0), ...dataStrides.slice(k)];
            scatterTargetIdx = buildLinearIndex(
                builder,
                updCoords,
                trailingStrides,
                `snd_tlin_${op.id}`,
            );
        }

        for (let j = 0; j < k; j++) {
            let elemIdxPos: ConcreteValueNode = idxTupleBase;
            if (j > 0) {
                const jConst = builder.createConstant(`snd_j_${j}_${op.id}`, scalarInt64(j));
                [elemIdxPos] = builder.createOp("Add", [idxTupleBase, jConst]);
            }
            const [posUnsq] = builder.createOp("Unsqueeze", [elemIdxPos, axes]);
            const [rawCoord] = builder.createOp("Gather", [flatIndices, posUnsq], { axis: 0 });
            let coord = squeezeIfLen1(
                builder,
                rawCoord,
                axes,
                `snd_sqc_${j}_${op.id}`,
            ) as ConcreteValueNode;
            if (ins[1].literalType !== DataType.INT64) {
                [coord] = builder.createOp("Cast", [coord], { to: DataType.INT64 });
            }

            const dimConst = builder.createConstant(
                `snd_dim_${j}_${op.id}`,
                scalarInt64(dataDims[j]),
            );
            const [isNeg] = builder.createOp("Less", [coord, zeroI64]);
            const [wrapped] = builder.createOp("Add", [coord, dimConst]);
            const [normCoord] = builder.createOp("Where", [isNeg, wrapped, coord]);

            const stride = dataStrides[j];
            const term =
                stride === 1
                    ? normCoord
                    : builder.createOp("Mul", [
                          normCoord,
                          builder.createConstant(`snd_str_${j}_${op.id}`, scalarInt64(stride)),
                      ])[0];
            [scatterTargetIdx] = builder.createOp("Add", [scatterTargetIdx, term]);
        }

        // Gather copy value and update value
        const [copyUnsq] = builder.createOp("Unsqueeze", [safeCopyIdx, axes]);
        const [copyRaw] = builder.createOp("Gather", [flatData, copyUnsq], { axis: 0 });
        const copyVal = squeezeIfLen1(
            builder,
            copyRaw,
            axes,
            `snd_sqd_${op.id}`,
        ) as ConcreteValueNode;

        const [updUnsq] = builder.createOp("Unsqueeze", [safeUpdIter, axes]);
        const [updRaw] = builder.createOp("Gather", [flatUpdates, updUnsq], { axis: 0 });
        let updVal = squeezeIfLen1(builder, updRaw, axes, `snd_squ_${op.id}`) as ConcreteValueNode;

        const [writeIdx] = builder.createOp("Where", [isCopyPhase, safeCopyIdx, scatterTargetIdx]);
        const [writeIdxUnsq] = builder.createOp("Unsqueeze", [writeIdx, axes]);

        if (reduction !== "none") {
            const [curRaw] = builder.createOp("Gather", [carryNode, writeIdxUnsq], { axis: 0 });
            const curVal = squeezeIfLen1(
                builder,
                curRaw,
                axes,
                `snd_sqcur_${op.id}`,
            ) as ConcreteValueNode;
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
