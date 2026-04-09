import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerHardSigmoidRecipe implements DecompositionRecipe {
    public readonly name = "LowerHardSigmoid";
    public readonly targetOp = "HardSigmoid";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Max", "Min", "Mul", "Add"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "HardSigmoid";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Output is a float tensor of the same shape as input
        const OutputType = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const Output = [{ type: OutputType, shape: X.shape as KnownShape }];

        // Create alpha and beta constants
        const alpha = getFloatAttr(op, "alpha", 0.2);
        const beta = getFloatAttr(op, "beta", 0.5);

        const alphaConst = builder.createConstant(
            `hardsigmoid_alpha_${op.id}`,
            makeTensorProto(OutputType, [], [alpha]),
        );

        const betaConst = builder.createConstant(
            `hardsigmoid_beta_${op.id}`,
            makeTensorProto(OutputType, [], [beta]),
        );

        const zeroConst = builder.createConstant(
            `hardsigmoid_zero_${op.id}`,
            makeTensorProto(OutputType, [], [0]),
        );

        const oneConst = builder.createConstant(
            `hardsigmoid_one_${op.id}`,
            makeTensorProto(OutputType, [], [1]),
        );

        // Mul (X*Alpha) + Beta
        const mulOut = builder.createOp("Mul", [X,alphaConst], {}, Output)[0];
        const addOut = builder.createOp("Add", [mulOut, betaConst], {}, Output)[0];

        // Min (1, addOut)
        const minOut = builder.createOp("Min", [oneConst, addOut], {}, Output)[0];

        // Max (0, minOut)
        const maxOut = builder.createOp("Max", [zeroConst, minOut], {}, Output)[0];

        // Replace original output with the new Max output
        builder.replaceAllUsesWith(Y, maxOut);
        op.remove();
    }
}