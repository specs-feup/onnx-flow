import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerTanhSigmoidRecipe implements DecompositionRecipe {
    public readonly name = "LowerTanhSigmoid";
    public readonly targetOp = "TanhSigmoid";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Mul", "Add", "Sigmoid"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "TanhSigmoid";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {   
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const twoConst = builder.createConstant(
            `tanh_two_${op.id}`,
            makeTensorProto(dtype, [], [2]),
        );

        const minusOneConst = builder.createConstant(
            `tanh_minus_one_${op.id}`,
            makeTensorProto(dtype, [], [-1]),
        );

        // Sigmoid (Mul (A, 2))
        const MulOut = builder.createOp("Mul", [A, twoConst], {}, output)[0];
        const SigmoidOut = builder.createOp("Sigmoid", [MulOut], {}, output)[0];

        // Add (Mul(SigmoidOut, 2), -1)
        const MulSigmoid = builder.createOp("Mul", [SigmoidOut, twoConst], {}, output)[0];
        const AddOut = builder.createOp("Add", [MulSigmoid, minusOneConst], {}, output)[0];

        builder.replaceAllUsesWith(Y, AddOut);
        op.remove();
    }
}        
    