import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, ValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, toStaticShape, broadcastShapes } from "../../../Utils.js";

export class LowerMatMulRecipe implements DecompositionRecipe {
    public readonly name = "LowerMatMul";
    public readonly targetOp = "MatMul";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Gather",
        "Mul",
        "ReduceSum",
        "ScatterElements",
        "Reshape",
        "Div",
        "Mod",
        "Add",
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

        // 1. Robust DType Inference (Fixes matmuladd_test INT32 crash)
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

        // Extract batch dimensions and compute broadcasted batch shape
        const batchA = shapeA.slice(0, -2);
        const batchB = shapeB.slice(0, -2);
        const batchOut = broadcastShapes(batchA, batchB);

        // Establish output shape
        let outShape = toStaticShape(output.shape);
        if (outShape.length === 0) {
            outShape = [...batchOut];
            if (!is1DA) outShape.push(M);
            if (!is1DB) outShape.push(N);
            if (outShape.length === 0) outShape = [1];
        }

        // Calculate loop properties
        const prodBatch = batchOut.reduce((a, b) => a * b, 1);
        const totalElements = prodBatch * M * N;

        const shapeConst = builder.createConstant(
            `shape_${op.id}`,
            makeTensorProto(DataType.INT64, [outShape.length], outShape),
        );

        const { innerBuilder, trip, vInitial, loopOutput, finalize } = builder.createLoopRegion(
            builder,
            totalElements,
            dtype,
            [totalElements],
            `MatMulLoop_${op.id}`,
        );

        // 2. Decode iterator into Batch, Row (i), and Column (j) indices
        const MNConst = innerBuilder.createConstant(
            `MN_${op.id}`,
            makeTensorProto(DataType.INT64, [], [M * N]),
        );
        const NConst = innerBuilder.createConstant(
            `N_${op.id}`,
            makeTensorProto(DataType.INT64, [], [N]),
        );

        const batchIter = innerBuilder.createOp("Div", [trip, MNConst])[0];
        const remMN = innerBuilder.createOp("Mod", [trip, MNConst])[0];
        const iIdx = innerBuilder.createOp("Div", [remMN, NConst])[0];
        const jIdx = innerBuilder.createOp("Mod", [remMN, NConst])[0];

        // 3. Local helper to decode `batchIter` into flattened input batch offsets
        const buildBatchIndex = (
            batchOutDims: number[],
            inBatchDims: number[],
            tag: string,
        ): ConcreteValueNode => {
            if (inBatchDims.length === 0) {
                return innerBuilder.createConstant(
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

            let flatInIdx: ConcreteValueNode = innerBuilder.createConstant(
                `${tag}_zero`,
                makeTensorProto(DataType.INT64, [], [0]),
            );

            for (let i = 0; i < rankIn; i++) {
                const inDim = inBatchDims[i];
                if (inDim === 1) continue; // broadcast dimension

                const outPos = rankOut - rankIn + i;
                const outStride = outStrides[outPos];
                const outDim = batchOutDims[outPos];

                let dimIdx: ValueNode = batchIter;
                if (outStride > 1) {
                    const outStrideConst = innerBuilder.createConstant(
                        `${tag}_ostride_${i}`,
                        makeTensorProto(DataType.INT64, [], [outStride]),
                    );
                    dimIdx = innerBuilder.createOp("Div", [dimIdx, outStrideConst])[0];
                }

                const outDimConst = innerBuilder.createConstant(
                    `${tag}_odim_${i}`,
                    makeTensorProto(DataType.INT64, [], [outDim]),
                );
                dimIdx = innerBuilder.createOp("Mod", [dimIdx, outDimConst])[0];

                const inStrideConst = innerBuilder.createConstant(
                    `${tag}_istride_${i}`,
                    makeTensorProto(DataType.INT64, [], [inStrides[i]]),
                );
                const offset = innerBuilder.createOp("Mul", [dimIdx, inStrideConst])[0];
                flatInIdx = innerBuilder.createOp("Add", [flatInIdx, offset])[0];
            }

            return flatInIdx;
        };

        const batchIdxA = buildBatchIndex(batchOut, batchA, `bA_${op.id}`);
        const batchIdxB = buildBatchIndex(batchOut, batchB, `bB_${op.id}`);

        // 4. Flatten the inputs down to 3D: [BatchProd, M, K] and [BatchProd, K, N]
        const prodBatchA = batchA.reduce((a, b) => a * b, 1);
        const prodBatchB = batchB.reduce((a, b) => a * b, 1);

        const shapeA3DConst = innerBuilder.createConstant(
            `shapeA3D_${op.id}`,
            makeTensorProto(DataType.INT64, [3], [prodBatchA, M, K]),
        );
        const shapeB3DConst = innerBuilder.createConstant(
            `shapeB3D_${op.id}`,
            makeTensorProto(DataType.INT64, [3], [prodBatchB, K, N]),
        );

        const A3D = innerBuilder.createOp("Reshape", [A, shapeA3DConst])[0];
        const B3D = innerBuilder.createOp("Reshape", [B, shapeB3DConst])[0];

        // 5. Gather the matrices using scalar indices to drop the axis dimension
        const A_matrix = innerBuilder.createOp("Gather", [A3D, batchIdxA], { axis: 0 })[0]; // -> [M, K]
        const B_matrix = innerBuilder.createOp("Gather", [B3D, batchIdxB], { axis: 0 })[0]; // -> [K, N]

        // 6. Gather row and col
        const rowA = innerBuilder.createOp("Gather", [A_matrix, iIdx], { axis: 0 })[0]; // -> [K]
        const colB = innerBuilder.createOp("Gather", [B_matrix, jIdx], { axis: 1 })[0]; // -> [K]

        // 7. Dot Product
        const mul = innerBuilder.createOp("Mul", [rowA, colB])[0];
        const sumScalar = innerBuilder.createOp("ReduceSum", [mul], { keepdims: 0 })[0]; // -> []

        // 8. Scatter
        const flatAxes = innerBuilder.createConstant(
            `axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const iterUnsq = innerBuilder.createOp("Unsqueeze", [trip, flatAxes])[0]; // -> [1]
        const updateVal = innerBuilder.createOp("Unsqueeze", [sumScalar, flatAxes])[0]; // -> [1]

        const scatterOut = innerBuilder.createOp(
            "ScatterElements",
            [vInitial, iterUnsq, updateVal],
            { axis: 0 },
        )[0];

        finalize([scatterOut]);

        // 9. Final reshape in the outer graph
        const finalReshape = builder.createOp("Reshape", [loopOutput, shapeConst], {}, [
            { type: dtype, shape: outShape },
        ])[0];

        builder.replaceAllUsesWith(output, finalReshape);
        op.remove();
    }
}
