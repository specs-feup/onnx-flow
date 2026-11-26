import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerNegRecipe implements DecompositionRecipe {
    public readonly name = "LowerNeg";
    public readonly targetOp = "Neg";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Mul"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Neg";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const outShape = Y.shape as KnownShape;
        const output = [{ type: dtype, shape: outShape }];

        const minusOne = builder.createConstant(
            `neg_minus_one_${op.id}`,
            makeTensorProto(dtype, [], [-1]),
        );
        const [negOut] = builder.createOp("Mul", [X, minusOne], {}, output);

        builder.replaceAllUsesWith(Y, negOut);
        builder.removeNode(op);
    }
}
