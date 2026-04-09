import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerSeluRecipe implements DecompositionRecipe {
    public readonly name = "LowerSelu";
    public readonly targetOp = "Selu";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Mul", "Exp", "Sub", "GreaterOrEqual", "Where"];

    canApply(op: OperationNode.Class): boolean {    
        return op.type === "Selu";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];
        
        // Output is a float tensor of the same shape as input
        const OutputType = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const BooleanOut = [{ type: DataType.BOOL, shape: X.shape as KnownShape }];
        const Output = [{ type: OutputType, shape: X.shape as KnownShape }];

        // Create constants
        const alpha = getFloatAttr(op, "alpha", 1.67326319217681884765625);
        const gamma = getFloatAttr(op, "gamma", 1.05070102214813232421875);

        const alphaConst = builder.createConstant(
            `selu_alpha_${op.id}`,
            makeTensorProto(OutputType, [], [alpha]),
        );

        const gammaConst = builder.createConstant(
            `selu_gamma_${op.id}`,
            makeTensorProto(OutputType, [], [gamma]),
        );

        const zeroConst = builder.createConstant(
            `selu_zero_${op.id}`,
            makeTensorProto(OutputType, [], [0]),
        );

        // Gamma * (alpha * Exp(X) - alpha) if X <= 0
        const expOut = builder.createOp("Exp", [X], {}, Output)[0];
        const mulOut = builder.createOp("Mul", [alphaConst, expOut], {}, Output)[0];
        const subOut = builder.createOp("Sub", [mulOut, alphaConst], {}, Output)[0];
        const finalOut = builder.createOp("Mul", [gammaConst, subOut], {}, Output)[0];

        // Gamma * X if X > 0
        const mulOut2 = builder.createOp("Mul", [gammaConst, X], {}, Output)[0];

        // Where(GreaterOrEqual(0, X), finalOut, mulOut2)
        const goeOut = builder.createOp("GreaterOrEqual", [zeroConst, X], {}, BooleanOut)[0];
        const whereOut = builder.createOp("Where", [goeOut, finalOut, mulOut2], {}, Output)[0];

        builder.replaceAllUsesWith(Y, whereOut);
        op.remove();
    }
}

