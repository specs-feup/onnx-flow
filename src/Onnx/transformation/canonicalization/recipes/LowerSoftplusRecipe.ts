import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerSoftplusRecipe implements DecompositionRecipe {
    public readonly name = "LowerSoftplus";
    public readonly targetOp = "Softplus";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Log", "Exp", "Add"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Softplus";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Output is a float tensor of the same shape as input
        const OutputType = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const Output = [{ type: OutputType, shape: X.shape as KnownShape }];

        const oneConst = builder.createConstant(
            `softplus_one_${op.id}`,
            makeTensorProto(OutputType, [], [1]),
        );

        // Add(Exp (X),1)
        const expOut = builder.createOp("Exp", [X], {}, Output)[0];
        const addOut = builder.createOp("Add", [expOut, oneConst], {}, Output)[0];

        // Log (addOut)
        const logOut = builder.createOp("Log", [addOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, logOut);
        op.remove();
    }
}


