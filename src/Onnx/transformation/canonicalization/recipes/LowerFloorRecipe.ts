import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerFloorRecipe implements DecompositionRecipe {
    public readonly name = "LowerFloor";
    public readonly targetOp = "Floor";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Sub", "Mod"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Floor";
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

        const oneConst = builder.createConstant(
            `floor_one_${op.id}_one`,
            makeTensorProto(dtype, [], [1]),
        );

        //Sub (A, Mod(A, 1))
        const ModOut = builder.createOp("Mod", [A, oneConst], {}, output)[0];
        const SubOut = builder.createOp("Sub", [A, ModOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, SubOut);
        op.remove();
    }
}        
    