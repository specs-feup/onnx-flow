import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerSigmoidRecipe implements DecompositionRecipe {
    public readonly name = "LowerSigmoid";
    public readonly targetOp = "Sigmoid";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Div", "Add", "Exp", "Neg"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Sigmoid";
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
            `sigmoid_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );

        // Add (1 + Exp(Neg(A)))
        const NegA = builder.createOp("Neg", [A], {}, output)[0];
        const ExpNegA = builder.createOp("Exp", [NegA], {}, output)[0];
        const Addout = builder.createOp("Add", [oneConst, ExpNegA], {}, output)[0];

        // Div (1, AddOut)
        const DivOut = builder.createOp("Div", [oneConst, Addout], {}, output)[0];

        builder.replaceAllUsesWith(Y, DivOut);
        op.remove();
    }
}
