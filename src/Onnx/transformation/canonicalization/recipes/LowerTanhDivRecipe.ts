import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerTanhDivRecipe implements DecompositionRecipe {
    public readonly name = "LowerTanhDiv";
    public readonly targetOp = "TanhDiv";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Sinh", "Cosh", "Div"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "TanhDiv";
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

        // Div(Sinh(A), Cosh(A))
        const SinhOut = builder.createOp("Sinh", [A], {}, output)[0];
        const CoshOut = builder.createOp("Cosh", [A], {}, output)[0];
        const DivOut = builder.createOp("Div", [SinhOut, CoshOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, DivOut);
        op.remove();
    }
}        
    