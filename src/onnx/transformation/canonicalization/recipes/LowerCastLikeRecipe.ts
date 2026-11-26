import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";

export class LowerCastLikeRecipe implements DecompositionRecipe {
    public readonly name = "LowerCastLike";
    public readonly targetOp = "CastLike";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Cast"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "CastLike") return false;
        const ins = op.getInputs();
        return !!ins && ins.length >= 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [X, target] = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];

        const targetDtype = target.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: targetDtype, shape: outShape }];

        const [out] = builder.createOp("Cast", [X], { to: targetDtype }, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
