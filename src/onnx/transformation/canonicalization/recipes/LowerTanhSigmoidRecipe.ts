import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerTanhSigmoidRecipe implements DecompositionRecipe {
    public readonly name = "LowerTanhSigmoid";
    public readonly targetOp = "Tanh";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 60;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["sigmoid-based"];
    public readonly producedOps = ["Mul", "Add", "Sigmoid"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Tanh";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

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
