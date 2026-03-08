import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, toStaticShape, broadcastShapes } from "../../../Utils.js";

export class LowerCoalescedMatMulRecipe implements DecompositionRecipe {
    public readonly name = "LowerCoalescedMatMul";
    public readonly targetOp = "MatMul";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Gather",
        "Mul",
        "Add",
        "ScatterElements",
        "Reshape",
        "Div",
        "Mod",
        "Unsqueeze",
    ];

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

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [A, B] = op.getInputs() as ConcreteValueNode[];
        const output = op.getOutputs()[0];

        let shapeA = toStaticShape(A.shape);
        let shapeB = toStaticShape(B.shape);

        // Robust DType Inference
        let dtype = (output.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED) {
            dtype = (A.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        }
        if (dtype === DataType.UNDEFINED) {
            dtype = DataType.FLOAT;
        }

        // Standardize 1D inputs to 2D
        let is1DA = false;
        let is1DB = false;

        if (shapeA.length === 1) {
            is1DA = true;
            shapeA = [1, shapeA[0]];
        }
        if (shapeB.length === 1) {
            is1DB = true;
            shapeB = [shapeB[0], 1];
        }

        const M = shapeA[shapeA.length - 2];
        const K = shapeA[shapeA.length - 1];
        const N = shapeB[shapeB.length - 1];

        const batchA = shapeA.slice(0, -2);
        const batchB = shapeB.slice(0, -2);
        const batchOut = broadcastShapes(batchA, batchB);

        let outShape = toStaticShape(output.shape);
        if (outShape.length === 0) {
            outShape = [...batchOut];
            if (!is1DA) outShape.push(M);
            if (!is1DB) outShape.push(N);
            if (outShape.length === 0) outShape = [1];
        }

        const prodBatch = batchOut.reduce((a, b) => a * b, 1);
        const totalOutElements = prodBatch * M * N;
        const totalIters = totalOutElements * K;

        const shapeConst = builder.createConstant(
            `shape_${op.id}`,
            makeTensorProto(DataType.INT64, [outShape.length], outShape),
        );

        // Reshape to 3D outside the loops to prevent duplicating work
        const prodBatchA = batchA.reduce((a, b) => a * b, 1);
        const prodBatchB = batchB.reduce((a, b) => a * b, 1);

        const shapeA3DConst = builder.createConstant(
            `shapeA3D_${op.id}`,
            makeTensorProto(DataType.INT64, [3], [prodBatchA, M, K]),
        );
        const shapeB3DConst = builder.createConstant(
            `shapeB3D_${op.id}`,
            makeTensorProto(DataType.INT64, [3], [prodBatchB, K, N]),
        );

        const A3D = builder.createOp("Reshape", [A, shapeA3DConst])[0];
        const B3D = builder.createOp("Reshape", [B, shapeB3DConst])[0];

        // ====================================================================
        // 1. Single Coalesced Loop Region (Iterates BatchProd * M * N * K times)
        // ====================================================================
        const {
            innerBuilder: loopBuilder,
            trip,
            vInitial: carryInit,
            loopOutput,
            finalize,
        } = builder.createForLoopRegion(
            builder,
            totalIters,
            dtype,
            [totalOutElements], // Carried state is a flat 1D tensor of all output elements
            `CoalescedMatMul_Loop_${op.id}`,
        );

        // --------------------------------------------------------------------
        // Decode `trip` into [batchIter, i, j, k]
        // K is the innermost (fastest changing) index.
        // --------------------------------------------------------------------
        const MNKConst = loopBuilder.createConstant(
            `MNK_${op.id}`,
            makeTensorProto(DataType.INT64, [], [M * N * K]),
        );
        const NKConst = loopBuilder.createConstant(
            `NK_${op.id}`,
            makeTensorProto(DataType.INT64, [], [N * K]),
        );
        const KConst = loopBuilder.createConstant(
            `K_${op.id}`,
            makeTensorProto(DataType.INT64, [], [K]),
        );
        const NConst = loopBuilder.createConstant(
            `N_${op.id}`,
            makeTensorProto(DataType.INT64, [], [N]),
        );

        const batchIter = loopBuilder.createOp("Div", [trip, MNKConst])[0];
        const remMNK = loopBuilder.createOp("Mod", [trip, MNKConst])[0];

        const iIdx = loopBuilder.createOp("Div", [remMNK, NKConst])[0];
        const remNK = loopBuilder.createOp("Mod", [remMNK, NKConst])[0];

        const jIdx = loopBuilder.createOp("Div", [remNK, KConst])[0];
        const kIdx = loopBuilder.createOp("Mod", [remNK, KConst])[0];

        // Local helper to decode `batchIter` into flat batch offsets for A and B
        const buildBatchIndex = (
            batchOutDims: number[],
            inBatchDims: number[],
            tag: string,
        ): ConcreteValueNode => {
            if (inBatchDims.length === 0) {
                return loopBuilder.createConstant(
                    `${tag}_zero`,
                    makeTensorProto(DataType.INT64, [], [0]),
                );
            }

            const rankOut = batchOutDims.length;
            const rankIn = inBatchDims.length;

            const outStrides = new Array(rankOut);
            let acc = 1;
            for (let i = rankOut - 1; i >= 0; i--) {
                outStrides[i] = acc;
                acc *= batchOutDims[i];
            }

            const inStrides = new Array(rankIn);
            acc = 1;
            for (let i = rankIn - 1; i >= 0; i--) {
                inStrides[i] = acc;
                acc *= inBatchDims[i];
            }

            let flatInIdx: ConcreteValueNode = loopBuilder.createConstant(
                `${tag}_zero`,
                makeTensorProto(DataType.INT64, [], [0]),
            );

            for (let i = 0; i < rankIn; i++) {
                const inDim = inBatchDims[i];
                if (inDim === 1) continue;

                const outPos = rankOut - rankIn + i;
                const outStride = outStrides[outPos];
                const outDim = batchOutDims[outPos];

                let dimIdx: ValueNode = batchIter;
                if (outStride > 1) {
                    const outStrideConst = loopBuilder.createConstant(
                        `${tag}_ostride_${i}`,
                        makeTensorProto(DataType.INT64, [], [outStride]),
                    );
                    dimIdx = loopBuilder.createOp("Div", [dimIdx, outStrideConst])[0];
                }

                const outDimConst = loopBuilder.createConstant(
                    `${tag}_odim_${i}`,
                    makeTensorProto(DataType.INT64, [], [outDim]),
                );
                dimIdx = loopBuilder.createOp("Mod", [dimIdx, outDimConst])[0];

                const inStrideConst = loopBuilder.createConstant(
                    `${tag}_istride_${i}`,
                    makeTensorProto(DataType.INT64, [], [inStrides[i]]),
                );
                const offset = loopBuilder.createOp("Mul", [dimIdx, inStrideConst])[0];
                flatInIdx = loopBuilder.createOp("Add", [flatInIdx, offset])[0];
            }
            return flatInIdx;
        };

        const batchIdxA = buildBatchIndex(batchOut, batchA, `bA_${op.id}`);
        const batchIdxB = buildBatchIndex(batchOut, batchB, `bB_${op.id}`);

        // --------------------------------------------------------------------
        // Fetch A_val and B_val, then Multiply
        // --------------------------------------------------------------------
        const A_matrix = loopBuilder.createOp("Gather", [A3D, batchIdxA], { axis: 0 })[0];
        const B_matrix = loopBuilder.createOp("Gather", [B3D, batchIdxB], { axis: 0 })[0];

        const rowA = loopBuilder.createOp("Gather", [A_matrix, iIdx], { axis: 0 })[0]; // -> [K]
        const valA = loopBuilder.createOp("Gather", [rowA, kIdx], { axis: 0 })[0]; // -> []

        const rowB = loopBuilder.createOp("Gather", [B_matrix, kIdx], { axis: 0 })[0]; // -> [N]
        const valB = loopBuilder.createOp("Gather", [rowB, jIdx], { axis: 0 })[0]; // -> []

        const prod = loopBuilder.createOp("Mul", [valA, valB])[0]; // -> []

        // --------------------------------------------------------------------
        // Compute Flat Output Index for C: (batchIter * M * N) + (iIdx * N) + jIdx
        // --------------------------------------------------------------------
        const MNConst = loopBuilder.createConstant(
            `MN_${op.id}`,
            makeTensorProto(DataType.INT64, [], [M * N]),
        );
        const batchOffset = loopBuilder.createOp("Mul", [batchIter, MNConst])[0];
        const iOffset = loopBuilder.createOp("Mul", [iIdx, NConst])[0];

        const iPlusJ = loopBuilder.createOp("Add", [iOffset, jIdx])[0];
        const flatOutIdx = loopBuilder.createOp("Add", [batchOffset, iPlusJ])[0]; // -> []

        // --------------------------------------------------------------------
        // Update Carried State (Gather -> Add -> ScatterElements)
        // --------------------------------------------------------------------
        const flatAxes = loopBuilder.createConstant(
            `axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const flatOutIdxUnsq = loopBuilder.createOp("Unsqueeze", [flatOutIdx, flatAxes])[0]; // -> [1]

        // 1. Fetch current accumulated value at C[flatOutIdx]
        const currentSum = loopBuilder.createOp("Gather", [carryInit, flatOutIdxUnsq], {
            axis: 0,
        })[0]; // -> [1]

        // 2. Add the new product
        const prodUnsq = loopBuilder.createOp("Unsqueeze", [prod, flatAxes])[0]; // -> [1]
        const newSum = loopBuilder.createOp("Add", [currentSum, prodUnsq])[0]; // -> [1]

        // 3. Scatter back into the carried 1D state array.
        // All ranks match: carryInit[TotalElements], flatOutIdxUnsq[1], newSum[1]
        const nextCarry = loopBuilder.createOp(
            "ScatterElements",
            [carryInit, flatOutIdxUnsq, newSum],
            { axis: 0 },
        )[0];

        finalize([nextCarry]);

        // ====================================================================
        // Final reshape in the outermost graph
        // ====================================================================
        const finalReshape = builder.createOp("Reshape", [loopOutput, shapeConst], {}, [
            { type: dtype, shape: outShape },
        ])[0];

        builder.replaceAllUsesWith(output, finalReshape);
        op.remove();
    }
}
