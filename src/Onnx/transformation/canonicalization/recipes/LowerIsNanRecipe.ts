import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerIsNanRecipe implements DecompositionRecipe {
    public readonly name = "LowerIsNan";
    public readonly targetOp = "IsNaN";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Equal","Not"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "IsNaN";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {   
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            DataType.BOOL;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        // Not(Equal(A, A))
        const EqualOut = builder.createOp("Equal", [A, A], {}, output)[0];
        const NotOut = builder.createOp("Not", [EqualOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, NotOut);
        op.remove();
    }
}        
    