import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerMishRecipe implements DecompositionRecipe {
    public readonly name = "LowerMish";
    public readonly targetOp = "Mish";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Mul", "Tanh", "Softplus"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Mish";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Output is a float tensor of the same shape as input
        const OutputType = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const Output = [{ type: OutputType, shape: X.shape as KnownShape }];

        //Softplus(X)
        const softplusOut = builder.createOp("Softplus", [X], {}, Output)[0];

        //Tanh(softplusOut)
        const tanhOut = builder.createOp("Tanh", [softplusOut], {}, Output)[0];

        //Mul(X, tanhOut)
        const mulOut = builder.createOp("Mul", [X, tanhOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, mulOut);
        op.remove();
    }
}