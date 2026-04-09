import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerDivRecipe implements DecompositionRecipe {
    public readonly name = "LowerDiv";
    public readonly targetOp = "Div";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Mul", "Reciprocal"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Div";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = op.getOutputs()[0];

        // Output is a float tensor of the same shape as input
        const Output = [{ type: (A.literalType as DataType | undefined) ?? DataType.FLOAT, shape: A.shape as KnownShape }];

        // Mul (A , Reciprocal (B) )
        const reciprocalB = builder.createOp("Reciprocal", [B], {}, Output)[0];
        const mulOut = builder.createOp("Mul",[A, reciprocalB], {}, Output)[0];

        builder.replaceAllUsesWith(Y, mulOut);
        op.remove();
    }
}