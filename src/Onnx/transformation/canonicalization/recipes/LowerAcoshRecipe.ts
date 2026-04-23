import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerAcoshRecipe implements DecompositionRecipe {
    public readonly name = "LowerAcosh";
    public readonly targetOp = "Acosh";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Mul", "Add", "Sqrt", "Log"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Acosh";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {   
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const minusOneConst = builder.createConstant(
            `acosh_minus_one_${op.id}`,
            makeTensorProto(dtype, [], [-1]),
        );

        // Add(Mul(A, A), -1)
        const MulOut = builder.createOp("Mul", [A, A], {}, output)[0];
        const AddOut = builder.createOp("Add", [MulOut, minusOneConst], {}, output)[0];

        //Add (A , Sqrt(AddOut))
        const SqrtOut = builder.createOp("Sqrt", [AddOut], {}, output)[0];
        const AddFinal = builder.createOp("Add", [A, SqrtOut], {}, output)[0];

        // Log(AddFinal)
        const LogOut = builder.createOp("Log", [AddFinal], {}, output)[0];

        builder.replaceAllUsesWith(Y, LogOut);
        op.remove();
    }
}        
    