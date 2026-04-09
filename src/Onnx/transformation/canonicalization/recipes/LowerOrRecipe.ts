import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerOrRecipe implements DecompositionRecipe {
    public readonly name = "LowerOr";
    public readonly targetOp = "Or";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Not", "And"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Or";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = op.getOutputs()[0];

        // Output is a boolean tensor of the same shape as input
        const Output = [{ type: DataType.BOOL, shape: A.shape as KnownShape }];
        
        // Not (And (Not A, Not B))
        const notA = builder.createOp("Not", [A], {}, Output)[0];
        const notB = builder.createOp("Not", [B], {}, Output)[0];
        const andOut = builder.createOp("And", [notA, notB], {}, Output)[0];
        const notAndOut = builder.createOp("Not", [andOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, notAndOut);
        op.remove();
    }
}
