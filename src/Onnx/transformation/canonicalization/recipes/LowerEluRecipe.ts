import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerEluRecipe implements DecompositionRecipe {
    public readonly name = "LowerElu";
    public readonly targetOp = "Elu";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["GreaterOrEqual", "Exp", "Sub", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Elu";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];
        
        // Output is a float tensor of the same shape as input
        const OutputType = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const Output = [{ type: OutputType, shape: X.shape as KnownShape }];
        const BooleanOutput = [{ type: DataType.BOOL, shape: X.shape as KnownShape }];

        // Get alpha
        const alpha = getFloatAttr(op, "alpha", 1.0);

        const alphaConst = builder.createConstant(
            `elu_alpha_${op.id}`,
            makeTensorProto(OutputType, [], [alpha]),
        );

        const zeroConst = builder.createConstant(
            `elu_zero_${op.id}`,
            makeTensorProto(OutputType, [], [0]),
        );

        const oneConst = builder.createConstant(
            `elu_one_${op.id}`,
            makeTensorProto(OutputType, [], [1]),
        );
        
        // GreaterOrEqual(X,0)
        const goeOut = builder.createOp("GreaterOrEqual", [X, zeroConst], {}, BooleanOutput)[0];

        // alpha * (Exp(X) - 1)
        const expOut = builder.createOp("Exp", [X], {}, Output)[0];
        const subOut = builder.createOp("Sub", [expOut, oneConst], {}, Output)[0];
        const mulOut = builder.createOp("Mul", [alphaConst, subOut], {}, Output)[0];

        // Where(goeOut, X, mulOut)
        const whereOut = builder.createOp("Where", [goeOut, X, mulOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, whereOut);
        op.remove();
    }  
}