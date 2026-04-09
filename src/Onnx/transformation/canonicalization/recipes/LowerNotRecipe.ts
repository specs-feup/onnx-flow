import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerNotRecipe implements DecompositionRecipe {
    public readonly name = "LowerNot";
    public readonly targetOp = "Not";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Xor"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Not ";
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
            `not_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );

        // Xor (A, 1)
        const XorOut = builder.createOp("Xor", [A, oneConst], {}, output)[0];

        builder.replaceAllUsesWith(Y, XorOut);
        op.remove();
    }
}