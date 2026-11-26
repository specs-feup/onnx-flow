import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerSoftsignRecipe implements DecompositionRecipe {
    public readonly name = "LowerSoftsign";
    public readonly targetOp = "Softsign";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Div", "Add", "Abs"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Softsign";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            DataType.BOOL;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

        const output = [{ type: dtype, shape: outShape }];

        const oneConst = builder.createConstant(
            `softsign_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );

        // Add (1 + Abs(A))
        const AbsOut = builder.createOp("Abs", [A], {}, output)[0];
        const AddOut = builder.createOp("Add", [oneConst, AbsOut], {}, output)[0];

        // Div ( A, AddOut)
        const DivOut = builder.createOp("Div", [A, AddOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, DivOut);
        op.remove();
    }
}
