import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerModRecipe implements DecompositionRecipe {
    public readonly name = "LowerMod";
    public readonly targetOp = "Mod";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Add", "Div", "Floor", "Neg", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Mod";
    }

    apply(node: OperationNode.Class, builder: GraphBuilder): void {
        const ins = node.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = node.getOutputs()[0];

        // Output is a float tensor of the same shape as input
        const Output = [{ type: (A.literalType as DataType | undefined) ?? DataType.FLOAT, shape: A.shape as KnownShape }];

        // Floor ( A/ B )
        const div = builder.createOp("Div", [A, B], {}, Output)[0];
        const floor = builder.createOp("Floor", [div], {}, Output)[0];

        // Neg ( Mul ( Floor * B ) )
        const mul = builder.createOp("Mul", [floor, B], {}, Output)[0];
        const neg = builder.createOp("Neg", [mul], {}, Output)[0];

        // Add ( A , Neg )
        const add = builder.createOp("Add", [A, neg], {}, Output)[0];

        builder.replaceAllUsesWith(Y, add);
        node.remove();
    }
}