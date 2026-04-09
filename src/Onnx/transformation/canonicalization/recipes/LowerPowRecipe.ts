import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerPowRecipe implements DecompositionRecipe {
    public readonly name = "LowerPow";
    public readonly targetOp = "Pow";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Mul", "Exp", "Log"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Pow";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        // Mul (B , Log (A) )
        const LogA = builder.createOp("Log", [A], {}, output)[0];
        const MulOut = builder.createOp("Mul",[B, LogA], {}, output)[0];

        // Exp ( MulOut )
        const ExpOut = builder.createOp("Exp", [MulOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, ExpOut);
        op.remove();
    }
}