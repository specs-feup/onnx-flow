import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    uniq,
    asStaticDims,
    broadcastShapes,
    scalarInt64,
    computeStrides,
    int64Vec,
} from "../../../Utils.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput, buildLinearIndex, decodeMixedRadix } from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerMatMulRecipe implements LoopLoweringRecipe {
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

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `matmul_${op.id}`);
        const inputs = op.getInputs()!;
        const dtype = (op.getOutputs()[0].literalType as DataType) ?? DataType.FLOAT;

        let shapeA = asStaticDims(inputs[0].shape);
        let shapeB = asStaticDims(inputs[1].shape);

        // Normalize 1D vectors to 2D
        if (shapeA.length === 1) shapeA = [1, shapeA[0]];
        if (shapeB.length === 1) shapeB = [shapeB[0], 1];

        const M = shapeA[shapeA.length - 2];
        const K = shapeA[shapeA.length - 1];
        const N = shapeB[shapeB.length - 1];

        const batchA = shapeA.slice(0, -2);
        const batchB = shapeB.slice(0, -2);
        const batchOut = broadcastShapes(...[batchA, batchB]);

        // 1. Decode global loop iteration into Batch, I, and J indices
        const MNConst = builder.createConstant(`MN`, scalarInt64(M * N));
        const NConst = builder.createConstant(`N`, scalarInt64(N));

        const [batchIter] = builder.createOp("Div", [iter, MNConst]);
        const [remMN] = builder.createOp("Mod", [iter, MNConst]);
        const [iIdx] = builder.createOp("Div", [remMN, NConst]);
        const [jIdx] = builder.createOp("Mod", [remMN, NConst]);

        // 2. Resolve batch offsets for inputs
        const getBatchOffset = (targetBatch: number[], tag: string) => {
            if (targetBatch.length === 0)
                return builder.createConstant(`${tag}_zero`, scalarInt64(0));
            const batchIndices = decodeMixedRadix(builder, batchIter, batchOut, `${tag}_decode`);

            // Handle broadcasting for the batch dimensions
            const actualIndices = targetBatch.map((dim, i) => {
                const outPos = batchOut.length - targetBatch.length + i;
                return dim === 1
                    ? builder.createConstant(`${tag}_dim_${i}_zero`, scalarInt64(0))
                    : batchIndices[outPos];
            });

            return buildLinearIndex(
                builder,
                actualIndices,
                computeStrides(targetBatch),
                `${tag}_offset`,
            );
        };

        const bOffsetA = getBatchOffset(batchA, "bA");
        const bOffsetB = getBatchOffset(batchB, "bB");

        // 3. Capture and Reshape inputs to 3D [Batch, Dim1, Dim2] for easier gathering
        const tInnerA = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
        );
        const tInnerB = resolveRecipeInput(
            builder,
            inputs[1],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
        );

        const [A3D] = builder.createOp("Reshape", [
            tInnerA,
            builder.createConstant("shapeA3D", int64Vec([-1, M, K])),
        ]);
        const [B3D] = builder.createOp("Reshape", [
            tInnerB,
            builder.createConstant("shapeB3D", int64Vec([-1, K, N])),
        ]);

        // 4. Gather the specific Matrix, Row, and Column for this iteration
        const [bA_unsq] = builder.createOp("Unsqueeze", [bOffsetA, axes]);
        const [AMatrix] = builder.createOp("Gather", [A3D, bA_unsq], { axis: 0 });

        const [bB_unsq] = builder.createOp("Unsqueeze", [bOffsetB, axes]);
        const [BMatrix] = builder.createOp("Gather", [B3D, bB_unsq], { axis: 0 });

        const [iIdx_unsq] = builder.createOp("Unsqueeze", [iIdx, axes]);
        const [rowA] = builder.createOp("Gather", [AMatrix, iIdx_unsq], { axis: 1 });

        const [jIdx_unsq] = builder.createOp("Unsqueeze", [jIdx, axes]);
        const [colB] = builder.createOp("Gather", [BMatrix, jIdx_unsq], { axis: 2 });

        // Reshape both vectors to 1D [-1] so they align for element-wise multiplication
        const [rowA_flat] = builder.createOp("Reshape", [
            rowA,
            builder.createConstant("K_flat_A", int64Vec([-1])),
        ]);
        const [colB_flat] = builder.createOp("Reshape", [
            colB,
            builder.createConstant("K_flat_B", int64Vec([-1])),
        ]);

        // 5. Mul computes [A_ik * B_kj] for k=0..K-1
        const [mulOp] = builder.createOp("Mul", [rowA_flat, colB_flat]);
        const [sumOut] = builder.createOp("ReduceSum", [mulOp], { keepdims: 0 });

        return sumOut;
    }
}
