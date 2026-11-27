import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

/**
 * Canonicalizes Swish (Opset 24): Swish(X, alpha) = X * Sigmoid(alpha * X).
 */
export class LowerSwishRecipe implements DecompositionRecipe {
    public readonly name = "LowerSwish";
    public readonly targetOp = "Swish";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Mul", "Sigmoid"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Swish" && (op.getInputs()?.length ?? 0) >= 1;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const outShape = Y.shape as KnownShape;
        const expected = [{ type: dtype, shape: outShape }];
        const alpha = getFloatAttr(op, "alpha", 1.0);

        let sigIn: ConcreteValueNode = X;
        if (alpha !== 1.0) {
            const alphaConst = builder.createConstant(
                `swish_alpha_${op.id}`,
                makeTensorProto(dtype, [], [alpha]),
            );
            [sigIn] = builder.createOp("Mul", [X, alphaConst], {}, expected);
        }

        const [sigOut] = builder.createOp("Sigmoid", [sigIn], {}, expected);
        const [out] = builder.createOp("Mul", [X, sigOut], {}, expected);

        builder.replaceAllUsesWith(Y, out);
        op.remove();
    }
}
