import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerSinhRecipe implements DecompositionRecipe {
    public readonly name = "LowerSinh";
    public readonly targetOp = "Sinh";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Mul", "Add", "Neg", "Exp"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Sinh";
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


        const halfConst = builder.createConstant(
            `sinh_half_${op.id}`,
            makeTensorProto(dtype, [], [0.5]),
        );

        // Neg(Exp(Neg(A))))
        const NegA = builder.createOp("Neg", [A], {}, [{ type: dtype, shape: outShape }])[0];
        const ExpNegA = builder.createOp("Exp", [NegA], {}, [{ type: dtype, shape: outShape }])[0];
        const NegExpNegA = builder.createOp("Neg", [ExpNegA], {}, [{ type: dtype, shape: outShape }])[0];

        // Add (Exp(A), NegExpNegA)
        const ExpA = builder.createOp("Exp", [A], {}, [{ type: dtype, shape: outShape }])[0];
        const AddOut = builder.createOp("Add", [ExpA, NegExpNegA], {}, [{ type: dtype, shape: outShape }])[0];

        // Mul (0.5, AddOut)
        const MulOut = builder.createOp("Mul", [halfConst, AddOut], {}, [{ type: dtype, shape: outShape }])[0];

        builder.replaceAllUsesWith(Y, MulOut);
        op.remove();
    }
}