import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerAtanhRecipe implements DecompositionRecipe {
    public readonly name = "LowerAtanh";
    public readonly targetOp = "Atanh";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Mul", "Log", "Div", "Add", "Sub"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Atanh";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {   
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const oneConst = builder.createConstant(
            `atanh_one_${op.id}_one`,
            makeTensorProto(dtype, [], [1]),
        );

        const halfConst = builder.createConstant(
            `atanh_half_${op.id}_half`,
            makeTensorProto(dtype, [], [0.5]),
        );

        //Div ( Add (1, A) , Sub(1, A))
        const AddOut = builder.createOp("Add", [oneConst, A], {}, output)[0];
        const SubOut = builder.createOp("Sub", [oneConst, A], {}, output)[0];
        const DivOut = builder.createOp("Div", [AddOut, SubOut], {}, output)[0];

        //Mul (0.5, Log(DivOut))
        const LogOut = builder.createOp("Log", [DivOut], {}, output)[0];
        const MulOut = builder.createOp("Mul", [halfConst, LogOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, MulOut);
        op.remove();
    }
}        
    