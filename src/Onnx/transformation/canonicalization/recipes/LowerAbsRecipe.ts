import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerAbsRecipe implements DecompositionRecipe {
    public readonly name = "LowerAbs";
    public readonly targetOp = "Abs";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Where","Greater","Neg"];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "Abs") return false;

    const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        //Expected output type is same as input type and shape is same as input shape
        const OutType = A.literalType as DataType | undefined ?? DataType.FLOAT;
        const OutShape = (Y.shape as KnownShape) ?? (A.shape as KnownShape);
        const Output = [{ type: OutType, shape: OutShape }];

         // 1. Create a scalar '0' constant of the same type
        const zeroConst = builder.createConstant(
            `abs_zero_${op.id}`,
            makeTensorProto(OutType, [], [0]),
        );

        //Greater (A,0)
        const GreaterOut = builder.createOp("Greater", [A, zeroConst], {},Output)[0];

        //Neg(A)
        const NegOut = builder.createOp("Neg", [A], {},Output)[0];

        //Where(GreaterOut, A, Neg(A))
        const WhereOut = builder.createOp("Where", [GreaterOut, A, NegOut], {},Output)[0];

        builder.replaceAllUsesWith(Y, WhereOut);
        op.remove();
    }
}