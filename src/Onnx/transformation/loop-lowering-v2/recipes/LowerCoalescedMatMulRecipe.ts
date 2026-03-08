import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { asStaticDims, scalarInt64, broadcastShapes, computeStrides } from "../../../Utils.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import {
    resolveRecipeInput,
    squeezeIfLen1,
    ensureFlatInput,
    buildLinearIndex,
    decodeMixedRadix,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerCoalescedMatMulRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerCoalescedMatMul";

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "MatMul") return false;
        const inputs = op.getInputs();
        return (
            !!inputs &&
            inputs.length >= 2 &&
            inputs[0].shape.length > 0 &&
            inputs[1].shape.length > 0
        );
    }

    getLoopBounds(op: OperationNode.Class, outShape: KnownShape) {
        const [A, B] = op.getInputs()!;
        let sA = asStaticDims(A.shape),
            sB = asStaticDims(B.shape);
        if (sA.length === 1) sA = [1, sA[0]];
        if (sB.length === 1) sB = [sB[0], 1];

        const M = sA[sA.length - 2],
            K = sA[sA.length - 1],
            N = sB[sB.length - 1];
        const batchOut = broadcastShapes(...[sA.slice(0, -2), sB.slice(0, -2)]);
        const prodBatch = batchOut.reduce((a, b) => a * b, 1);

        return { totalIters: prodBatch * M * N * K, carryShape: [prodBatch * M * N] };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `coal_mm_${op.id}`);
        const inputs = op.getInputs()!;
        const dtype = (op.getOutputs()[0].literalType as DataType) ?? DataType.FLOAT;

        let sA = asStaticDims(inputs[0].shape),
            sB = asStaticDims(inputs[1].shape);
        if (sA.length === 1) sA = [1, sA[0]];
        if (sB.length === 1) sB = [sB[0], 1];

        const M = sA[sA.length - 2],
            K = sA[sA.length - 1],
            N = sB[sB.length - 1];
        const bA = sA.slice(0, -2),
            bB = sB.slice(0, -2);
        const batchOut = broadcastShapes(...[bA, bB]);

        // 1. Decode multi-index: Batch, I, J, K
        const [batchIter] = builder.createOp("Div", [
            iter,
            builder.createConstant(`MNK`, scalarInt64(M * N * K)),
        ]);
        const [remMNK] = builder.createOp("Mod", [
            iter,
            builder.createConstant(`MNK_rem`, scalarInt64(M * N * K)),
        ]);
        const [iIdx] = builder.createOp("Div", [
            remMNK,
            builder.createConstant(`NK`, scalarInt64(N * K)),
        ]);
        const [remNK] = builder.createOp("Mod", [
            remMNK,
            builder.createConstant(`NK_rem`, scalarInt64(N * K)),
        ]);
        const [jIdx] = builder.createOp("Div", [
            remNK,
            builder.createConstant(`K`, scalarInt64(K)),
        ]);
        const [kIdx] = builder.createOp("Mod", [
            remNK,
            builder.createConstant(`K_rem`, scalarInt64(K)),
        ]);

        // 2. Map coordinates to flat input indices for A and B
        const getFlatIdx = (
            targetBatch: number[],
            rowIdx: ValueNode,
            colIdx: ValueNode,
            rows: number,
            cols: number,
            tag: string,
        ) => {
            let batchOffset: ValueNode = builder.createConstant(
                `${tag}_batch_zero`,
                scalarInt64(0),
            );

            if (targetBatch.length > 0) {
                // Proper broadcast mapping: decode global batch index and re-encode for target shape
                const batchIndices = decodeMixedRadix(
                    builder,
                    batchIter,
                    batchOut,
                    `${tag}_decode`,
                );
                const actualIndices = targetBatch.map((dim, i) => {
                    const outPos = batchOut.length - targetBatch.length + i;
                    // If target dimension is 1, index is always 0; otherwise use decoded index
                    return dim === 1
                        ? builder.createConstant(`${tag}_dim_${i}_zero`, scalarInt64(0))
                        : batchIndices[outPos];
                });
                batchOffset = buildLinearIndex(
                    builder,
                    actualIndices,
                    computeStrides(targetBatch),
                    `${tag}_batch_offset`,
                );
            }

            const [offset] = builder.createOp("Mul", [
                batchOffset,
                builder.createConstant(`${tag}_mul`, scalarInt64(rows * cols)),
            ]);
            const [rowOff] = builder.createOp("Mul", [
                rowIdx,
                builder.createConstant(`${tag}_col_const`, scalarInt64(cols)),
            ]);
            const [base] = builder.createOp("Add", [offset, rowOff]);
            return builder.createOp("Add", [base, colIdx])[0];
        };

        const idxA = getFlatIdx(bA, iIdx, kIdx, M, K, "A");
        const idxB = getFlatIdx(bB, kIdx, jIdx, K, N, "B");

        // 3. Gather elements and compute partial product
        const flatA = ensureFlatInput(
            builder,
            resolveRecipeInput(builder, inputs[0], valueMap, iter, axes, outShape, false, false),
        );
        const flatB = ensureFlatInput(
            builder,
            resolveRecipeInput(builder, inputs[1], valueMap, iter, axes, outShape, false, false),
        );

        const [gAOut] = builder.createOp(
            "Gather",
            [flatA, builder.createOp("Unsqueeze", [idxA, axes])[0]],
            { axis: 0 },
        );
        const [gBOut] = builder.createOp(
            "Gather",
            [flatB, builder.createOp("Unsqueeze", [idxB, axes])[0]],
            { axis: 0 },
        );

        const valA = squeezeIfLen1(builder, gAOut, axes, `sqA`);
        const valB = squeezeIfLen1(builder, gBOut, axes, `sqB`);
        const [prod] = builder.createOp("Mul", [valA, valB]);

        // 4. Update the accumulator (Carry state)
        const flatOutIdx = buildLinearIndex(
            builder,
            [batchIter, iIdx, jIdx],
            [M * N, N, 1],
            `out_idx`,
        );
        const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        const [gAccOut] = builder.createOp("Gather", [carryNode, flatOutIdxUnsq], { axis: 0 });
        const currentAcc = squeezeIfLen1(builder, gAccOut, axes, `sqAcc`);

        const [newSum] = builder.createOp("Add", [currentAcc, prod]);
        const [scatterOut] = builder.createOp(
            "ScatterElements",
            [carryNode, flatOutIdxUnsq, builder.createOp("Unsqueeze", [newSum, axes])[0]],
            { axis: 0 },
        );

        return { resultNode: newSum, nextCarry: scatterOut };
    }
}
