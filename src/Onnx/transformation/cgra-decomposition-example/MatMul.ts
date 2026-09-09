import type OperationNode from "../../OperationNode.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { DecompositionRecipe } from "../Recipe.js";
import type { ConcreteValueNode } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import { chunkTensor, makeTensorProto, toStaticShape } from "../../Utils.js";

export class MatMulGridDecompositionRecipe implements DecompositionRecipe {
    public readonly name = "MatMulGridDecomposition";
    public readonly targetOp = "MatMul";

    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Gather", "Mul", "ReduceSum", "Concat", "Unsqueeze"];

    match(op: OperationNode.Class): boolean {
        const inputs = op.getInputs();
        if (!inputs || inputs.length !== 2) return false;
        // Only apply if inputs are 2D (basic matrix multiplication)
        if (inputs[0].shape.length === 2 && inputs[1].shape.length === 2) {
            return true;
        }
        return false;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [A, B] = op.getInputs() as ConcreteValueNode[];

        const shapeA = toStaticShape(A.shape);
        const shapeB = toStaticShape(B.shape);
        const numRows = shapeA[0];
        const numCols = shapeB[1];

        // 1. Chunk inputs into 1D vectors
        const rowsA = chunkTensor(builder, A, 0); // M vectors of size K
        const colsB = chunkTensor(builder, B, 1); // N vectors of size K

        const zeroConst = builder.createConstant(
            `zero_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const finalRows: ConcreteValueNode[] = [];

        // 2. Build the dot products (M x N grid)
        for (let r = 0; r < numRows; r++) {
            const colResults: ConcreteValueNode[] = [];

            for (let c = 0; c < numCols; c++) {
                // Element-wise multiplication of the two vectors
                const mul = builder.createOp("Mul", [rowsA[r], colsB[c]])[0];

                // Sum to a scalar
                const reduce = builder.createOp("ReduceSum", [mul], { keepdims: 0 })[0];

                // Unsqueeze scalar [] to [1] so it can be concatenated
                const unsq = builder.createOp("Unsqueeze", [reduce, zeroConst])[0];
                colResults.push(unsq);
            }

            // 3. Concat the columns into a single 1D row: [N]
            let rowOut: ConcreteValueNode;
            if (colResults.length === 1) {
                rowOut = colResults[0];
            } else {
                rowOut = builder.createOp("Concat", colResults, { axis: 0 })[0];
            }

            // Unsqueeze the 1D row [N] to a 2D row [1, N] for final stacking
            const row2D = builder.createOp("Unsqueeze", [rowOut, zeroConst])[0];
            finalRows.push(row2D);
        }

        // 4. Concat all 2D rows into the final matrix: [M, N]
        let finalMatrix: ConcreteValueNode;
        if (finalRows.length === 1) {
            finalMatrix = finalRows[0];
        } else {
            finalMatrix = builder.createOp("Concat", finalRows, { axis: 0 })[0];
        }

        // 5. Safely replace the original MatMul output and let the builder clean up
        const originalOutput = op.getOutputs()[0];
        builder.replaceAllUsesWith(originalOutput, finalMatrix);
        op.remove();
    }
}
