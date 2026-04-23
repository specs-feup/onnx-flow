import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerTanRecipe implements DecompositionRecipe {
    public readonly name = "LowerTan";
    public readonly targetOp = "Tan";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false
    public readonly producedOps = ["Div", "Sin", "Cos"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Tan";
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

        // Div ( Sin(A9), Cos(A) )
        const SinOut = builder.createOp("Sin", [A], {}, output)[0];
        const CosOut = builder.createOp("Cos", [A], {}, output)[0];
        const DivOut = builder.createOp("Div", [SinOut, CosOut], {}, output)[0];

        builder.replaceAllUsesWith(Y, DivOut);
        op.remove();
    }
}