import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerAcosRecipe implements DecompositionRecipe {
    public readonly name = "LowerAcos";
    public readonly targetOp = "Acos";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Asin", "Sub"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Acos";
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

        const piOverTwoConst = builder.createConstant(
            `acos_pi_over_two_${op.id}`,
            makeTensorProto(dtype, [], [Math.PI / 2]),
        );

        // asin(A)
        const asinOut: ConcreteValueNode = builder.createOp("Asin", [A], {}, output)[0];

        // pi/2 - asin(A)
        const acosOut: ConcreteValueNode = builder.createOp("Sub", [piOverTwoConst, asinOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, acosOut);
        op.remove();
    }
}