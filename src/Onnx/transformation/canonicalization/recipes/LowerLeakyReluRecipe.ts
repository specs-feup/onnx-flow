import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerLeakyReluRecipe implements DecompositionRecipe {
    public readonly name = "LowerLeakyRelu";
    public readonly targetOp = "LeakyRelu";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Greater", "Where", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "LeakyRelu";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Fallback to FLOAT if undefined
        const dtype = (X.literalType as DataType | undefined) ?? DataType.FLOAT;

        // Create a scalar '0' constant of the same type
        const zeroConst = builder.createConstant(
            `leakyrelu_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );

        const alpha = getFloatAttr(op, "alpha", 0.01);

        const alphaConst = builder.createConstant(
            `leakyrelu_alpha_${op.id}`,
            makeTensorProto(dtype, [], [alpha]),
        )

        // Mul (X*Alpha)
        const expectedValue = [{ type: dtype, shape: X.shape as KnownShape }];
        const mulOut = builder.createOp("Mul", [X, alphaConst], {}, expectedValue)[0];

        // Condition: Mask = Greater(X, 0)
        const expectedBool = [{ type: DataType.BOOL, shape: X.shape as KnownShape }];
        const greaterOut = builder.createOp("Greater", [X, zeroConst], {}, expectedBool)[0];

        // Selection: Out = Where(Mask, X, mulOut)
        const whereOut = builder.createOp("Where", [greaterOut, X, mulOut],{}, expectedValue)[0];

        // Safely replace the original Y with the new Where output
        builder.replaceAllUsesWith(Y, whereOut);
        op.remove();
    }
}
