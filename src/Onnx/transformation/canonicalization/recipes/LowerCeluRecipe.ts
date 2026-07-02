import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerCeluRecipe implements DecompositionRecipe {
    public readonly name = "LowerCelu";
    public readonly targetOp = "Celu";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Max", "Min", "Exp", "Sub", "Mul", "Div", "Add"];

    canApply(op: OperationNode.Class): boolean {    
        return op.type === "Celu";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Output is a float tensor of the same shape as input
        const OutputType = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const Output = [{ type: OutputType, shape: X.shape as KnownShape }];
        
        // Create constants
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
        
        // max(X,0)
        const maxOut = builder.createOp("Max", [X, zeroConst], {}, Output)[0];

        // Min(0, Alpha * (Exp(x/alpha) -1 ))
        const divOut = builder.createOp("Div", [X, alphaConst], {}, Output)[0];
        const expOut = builder.createOp("Exp", [divOut], {}, Output)[0];
        const subOut = builder.createOp("Sub", [expOut, oneConst], {}, Output)[0];
        const mulOut = builder.createOp("Mul", [alphaConst, subOut], {}, Output)[0];
        const minOut = builder.createOp("Min", [zeroConst, mulOut], {}, Output)[0];

        // Add(maxOut, minOut)
        const addOut = builder.createOp("Add", [maxOut, minOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, addOut);
        op.remove();
    }
}

        