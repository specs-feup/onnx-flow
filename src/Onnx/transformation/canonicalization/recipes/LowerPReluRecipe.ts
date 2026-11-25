import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerPReluRecipe implements DecompositionRecipe {
    public readonly name = "LowerPRelu";
    public readonly targetOp = "PRelu";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Greater", "Mul", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "PRelu" && (op.getInputs()?.length ?? 0) >= 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [X, slope] = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        const zero = builder.createConstant(`prelu_zero_${op.id}`, makeTensorProto(dtype, [], [0]));
        const [isPos] = builder.createOp("Greater", [X, zero], {}, boolMeta);
        const [negBranch] = builder.createOp("Mul", [X, slope], {}, outMeta);
        const [out] = builder.createOp("Where", [isPos, X, negBranch], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
