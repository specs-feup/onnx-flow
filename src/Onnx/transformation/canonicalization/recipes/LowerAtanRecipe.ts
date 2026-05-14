import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerAtanRecipe implements DecompositionRecipe {
    public readonly name = "LowerAtan";
    public readonly targetOp = "Atan";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Add", "Sub", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Atan";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (A.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const oneThirdConst = builder.createConstant(
            `atan_one_third_${op.id}`,
            makeTensorProto(dtype, [], [1.0 / 3.0]),
        );

        const oneFifthConst = builder.createConstant(
            `atan_one_fifth_${op.id}`,
            makeTensorProto(dtype, [], [1.0 / 5.0]),
        );

        // x^3 and x^5 are needed
        const x2 = builder.createOp("Mul", [A, A], {}, output)[0];
        const x3 = builder.createOp("Mul", [x2, A], {}, output)[0];
        const x5 = builder.createOp("Mul", [x3, x2], {}, output)[0];

        const x3Term = builder.createOp("Mul", [oneThirdConst, x3], {}, output)[0];
        const x5Term = builder.createOp("Mul", [oneFifthConst, x5], {}, output)[0];

        const sub1 = builder.createOp("Sub", [A, x3Term], {}, output)[0];
        const atanApprox = builder.createOp("Add", [sub1, x5Term], {}, output)[0];

        builder.replaceAllUsesWith(Y, atanApprox);
        op.remove();
    }
}