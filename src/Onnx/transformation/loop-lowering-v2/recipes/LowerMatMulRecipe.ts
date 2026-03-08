import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
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
import { resolveRecipeInput, buildLinearIndex, decodeMixedRadix, squeezeIfLen1 } from "../RecipeUtils.js";
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

        let staticShapeA = asStaticDims(inputs[0].shape);
        let staticShapeB = asStaticDims(inputs[1].shape);

        // Normalize 1D vectors to 2D for static checks
        if (staticShapeA.length === 1) staticShapeA = [1, staticShapeA[0]];
        if (staticShapeB.length === 1) staticShapeB = [staticShapeB[0], 1];

        // --- NEW DYNAMIC SUPPORT: Extract M, K, N ---
        let M: number | ValueNode, K: number | ValueNode, N: number | ValueNode;
        let batchA: number[], batchB: number[], batchOut: number[];

        const isDynamicA = staticShapeA.includes(-1) || staticShapeA.length === 0;
        const isDynamicB = staticShapeB.includes(-1) || staticShapeB.length === 0;

        if (isDynamicA || isDynamicB) {
            // Helper to gather a dimension from the back (e.g., -1 is last, -2 is second to last)
            const getDimNode = (tensor: ValueNode, rankNode: ValueNode, offsetFromEnd: number, tag: string) => {
                const [shapeNode] = builder.createOp("Shape", [tensor]);
                const offsetNode = builder.createConstant(`${tag}_offset`, scalarInt64(offsetFromEnd));
                const [targetAxis] = builder.createOp("Add", [rankNode, offsetNode]); // rank + (-offset)
                const [dimRaw] = builder.createOp("Gather", [shapeNode, builder.createOp("Unsqueeze", [targetAxis, axes])[0]], { axis: 0 });
                return squeezeIfLen1(builder, dimRaw, axes, `${tag}_sq`);
            };

            const [rankA] = builder.createOp("Size", [builder.createOp("Shape", [inputs[0]])[0]]);
            const [rankB] = builder.createOp("Size", [builder.createOp("Shape", [inputs[1]])[0]]);

            M = isDynamicA ? getDimNode(inputs[0], rankA, -2, "M") : staticShapeA[staticShapeA.length - 2];
            K = isDynamicA ? getDimNode(inputs[0], rankA, -1, "K") : staticShapeA[staticShapeA.length - 1];
            N = isDynamicB ? getDimNode(inputs[1], rankB, -1, "N") : staticShapeB[staticShapeB.length - 1];
        } else {
            M = staticShapeA[staticShapeA.length - 2];
            K = staticShapeA[staticShapeA.length - 1];
            N = staticShapeB[staticShapeB.length - 1];
        }

        // For fully dynamic batches, we rely on the pass's Shape inference to populate outShape.
        // We slice the broadcasted shapes out of the statically known bounds for now.
        batchA = staticShapeA.slice(0, -2);
        batchB = staticShapeB.slice(0, -2);
        batchOut = broadcastShapes(...[batchA as number[], batchB as number[]]); // Needs dynamic broadcast support if batches vary

        // 1. Decode global loop iteration into Batch, I, and J indices
        let MNConst: ValueNode, NConst: ValueNode;
        
        if (typeof M === "number" && typeof N === "number") {
            MNConst = builder.createConstant(`MN`, scalarInt64(M * N));
            NConst = builder.createConstant(`N`, scalarInt64(N));
        } else {
            const mNode = typeof M === "number" ? builder.createConstant(`M_const`, scalarInt64(M)) : M as ValueNode;
            NConst = typeof N === "number" ? builder.createConstant(`N_const`, scalarInt64(N)) : N as ValueNode;
            [MNConst] = builder.createOp("Mul", [mNode, NConst]);
        }

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
        const tInnerA = resolveRecipeInput(builder, inputs[0], valueMap, iter, axes, outShape, false, false);
        const tInnerB = resolveRecipeInput(builder, inputs[1], valueMap, iter, axes, outShape, false, false);

        // Build dynamic 3D shapes [-1, M, K] and [-1, K, N] using Concat
        const build3DShape = (dim1: number | ValueNode, dim2: number | ValueNode, tag: string) => {
            const minusOne = builder.createConstant(`${tag}_m1`, int64Vec([-1]));
            const d1 = typeof dim1 === "number" ? builder.createConstant(`${tag}_d1`, int64Vec([dim1])) : builder.createOp("Unsqueeze", [dim1, axes])[0];
            const d2 = typeof dim2 === "number" ? builder.createConstant(`${tag}_d2`, int64Vec([dim2])) : builder.createOp("Unsqueeze", [dim2, axes])[0];
            return builder.createOp("Concat", [minusOne, d1, d2], { axis: 0 })[0];
        };

        const [A3D] = builder.createOp("Reshape", [tInnerA, build3DShape(M, K, "A")]);
        const [B3D] = builder.createOp("Reshape", [tInnerB, build3DShape(K, N, "B")]);

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
