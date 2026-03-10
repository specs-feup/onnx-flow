import type OperationNode from "../../OperationNode.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { DecompositionRecipe } from "../Recipe.js";
import type { ConcreteValueNode } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import { chunkTensor, makeTensorProto, shapesEqual, toStaticShape } from "../../Utils.js";

export class AddGridDecompositionRecipe implements DecompositionRecipe {
    public readonly name = "AddGridDecomposition";
    public readonly targetOp = "Add";

    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Gather", "Add", "Unsqueeze", "Concat"];

    canApply(op: OperationNode.Class): boolean {
        const inputs = op.getInputs();
        if (!inputs || inputs.length !== 2) return false;

        const [in1, in2] = inputs as ConcreteValueNode[];
        // Only apply if inputs are 2D and exactly match
        return shapesEqual(in1, in2) && in1.shape.length === 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [A, B] = op.getInputs() as ConcreteValueNode[];
        const numRows = toStaticShape(A.shape)[0];

        // 1. Chunk inputs into 1D vectors along axis 0
        const rowsA = chunkTensor(builder, A, 0);
        const rowsB = chunkTensor(builder, B, 0);

        const zeroConst = builder.createConstant(
            `zero_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const finalRows: ConcreteValueNode[] = [];

        // 2. Build the independent additions
        for (let r = 0; r < numRows; r++) {
            // Element-wise Add of the two 1D vectors
            const addOut = builder.createOp("Add", [rowsA[r], rowsB[r]])[0];

            // Unsqueeze the 1D row back to 2D [1, K] for stacking
            const row2D = builder.createOp("Unsqueeze", [addOut, zeroConst])[0];
            finalRows.push(row2D);
        }

        // 3. Concat all 2D rows back into the final matrix: [M, K]
        let finalMatrix: ConcreteValueNode;
        if (finalRows.length === 1) {
            finalMatrix = finalRows[0];
        } else {
            finalMatrix = builder.createOp("Concat", finalRows, { axis: 0 })[0];
        }

        // 4. Safely replace the original output
        const originalOutput = op.getOutputs()[0];
        builder.replaceAllUsesWith(originalOutput, finalMatrix);
        op.remove();
    }
}
