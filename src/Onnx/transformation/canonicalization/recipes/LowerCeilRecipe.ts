import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerCeilRecipe implements DecompositionRecipe {
    public readonly name = "LowerCeil";
    public readonly targetOp = "Ceil";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Neg", "Floor"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Ceil";
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

        // Neg(Floor(Neg(A)))
        const NegA = builder.createOp("Neg", [A], {}, output)[0];
        const FloorNegA = builder.createOp("Floor", [NegA], {}, output)[0];
        const NegFloor = builder.createOp("Neg", [FloorNegA], {}, output)[0];

        builder.replaceAllUsesWith(Y, NegFloor);
        op.remove();
    }
}