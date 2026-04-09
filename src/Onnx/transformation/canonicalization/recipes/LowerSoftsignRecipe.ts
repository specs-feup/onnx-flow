import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerSoftsignRecipe implements DecompositionRecipe {
    public readonly name = "LowerSoftsign";
    public readonly targetOp = "Softsign";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Div", "Add", "Abs"];

    canApply(op: OperationNode.Class): boolean {
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

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

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