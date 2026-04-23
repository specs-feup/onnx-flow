import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerIsInfRecipe implements DecompositionRecipe {
    public readonly name = "LowerIsInf";
    public readonly targetOp = "IsInf";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Equal","Abs"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "IsInf";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {   
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (A.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];
        const boolOutput = [{ type: DataType.BOOL, shape: outShape }];

        const INF = builder.createConstant(
            `isinf_inf_${op.id}_inf`,
            makeTensorProto(DataType.FLOAT, [], [Infinity]),
        );

        //Equal(Abs(A), Inf)
        const AbsOut = builder.createOp("Abs", [A], {}, output)[0];
        const EqualOut = builder.createOp("Equal", [AbsOut, INF], {}, boolOutput)[0];


        builder.replaceAllUsesWith(Y, EqualOut);
        op.remove();
    }
}        
    