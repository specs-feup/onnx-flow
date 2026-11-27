import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";

export class LowerTanhDivRecipe implements DecompositionRecipe {
    public readonly name = "LowerTanhDiv";
    public readonly targetOp = "Tanh";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 40;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["hyperbolic-div"];
    public readonly producedOps = ["Sinh", "Cosh", "Div"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Tanh";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

        const output = [{ type: dtype, shape: outShape }];

        // Div(Sinh(A), Cosh(A))
        const SinhOut = builder.createOp("Sinh", [A], {}, output)[0];
        const CoshOut = builder.createOp("Cosh", [A], {}, output)[0];
        const DivOut = builder.createOp("Div", [SinhOut, CoshOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, DivOut);
        op.remove();
    }
}
