import type OperationNode from "../../OperationNode.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { DecompositionRecipe } from "../Recipe.js";
import type { ConcreteValueNode } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import { chunkTensor, makeTensorProto, toStaticShape } from "../../Utils.js";

export class ReluGridDecompositionRecipe implements DecompositionRecipe {
    public readonly name = "ReluDecomposition";
    public readonly targetOp = "Relu";

    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Gather", "Greater", "Where", "Unsqueeze", "Concat"];

    canApply(op: OperationNode.Class): boolean {
        const inputs = op.getInputs();
        if (!inputs || inputs.length !== 1) return false;
        // Restrict to 2D for this specific grid decomposition
        return inputs[0].shape.length === 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const numRows = toStaticShape(X.shape)[0];

        // 1. Chunk input into 1D vectors
        const rowsX = chunkTensor(builder, X, 0);

        // Constants
        const unsqueezeAxis = builder.createConstant(
            `unsq_axis_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        // Zero value of the same type as X (e.g. FLOAT) for comparisons
        const zeroVal = builder.createConstant(
            `relu_zero_${op.id}`,
            makeTensorProto(X.literalType, [], [0]),
        );

        const finalRows: ConcreteValueNode[] = [];

        // 2. Build the comparison and mux (Where) logic per row
        for (let r = 0; r < numRows; r++) {
            const row = rowsX[r];

            // Mask = row > 0
            const greaterOut = builder.createOp("Greater", [row, zeroVal])[0];

            // Out = Mask ? row : 0
            const whereOut = builder.createOp("Where", [greaterOut, row, zeroVal])[0];

            // Unsqueeze back to 2D
            const row2D = builder.createOp("Unsqueeze", [whereOut, unsqueezeAxis])[0];
            finalRows.push(row2D);
        }

        // 3. Concat all rows back into the final matrix
        let finalMatrix: ConcreteValueNode;
        if (finalRows.length === 1) {
            finalMatrix = finalRows[0];
        } else {
            finalMatrix = builder.createOp("Concat", finalRows, { axis: 0 })[0];
        }

        // 4. Safely replace
        const originalOutput = op.getOutputs()[0];
        builder.replaceAllUsesWith(originalOutput, finalMatrix);
        op.remove();
    }
}
