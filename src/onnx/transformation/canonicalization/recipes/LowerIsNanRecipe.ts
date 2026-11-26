import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";

export class LowerIsNanRecipe implements DecompositionRecipe {
    public readonly name = "LowerIsNan";
    public readonly targetOp = "IsNaN";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Equal", "Not"];

    match(op: OperationNode.Class): boolean {
        return op.type === "IsNaN";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype = (Y.literalType as DataType | undefined) ?? DataType.BOOL;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

        const output = [{ type: dtype, shape: outShape }];

        // Not(Equal(A, A))
        const EqualOut = builder.createOp("Equal", [A, A], {}, output)[0];
        const NotOut = builder.createOp("Not", [EqualOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, NotOut);
        op.remove();
    }
}
