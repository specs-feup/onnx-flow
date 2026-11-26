import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../utils/index.js";

export class LowerShrinkRecipe implements DecompositionRecipe {
    public readonly name = "LowerShrink";
    public readonly targetOp = "Shrink";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Less", "Greater", "Add", "Sub", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Shrink";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        const biasVal = getFloatAttr(op, "bias", 0.0);
        const lambdVal = getFloatAttr(op, "lambd", 0.5);

        const bias = builder.createConstant(
            `shrink_bias_${op.id}`,
            makeTensorProto(dtype, [], [biasVal]),
        );
        const posL = builder.createConstant(
            `shrink_pl_${op.id}`,
            makeTensorProto(dtype, [], [lambdVal]),
        );
        const negL = builder.createConstant(
            `shrink_nl_${op.id}`,
            makeTensorProto(dtype, [], [-lambdVal]),
        );
        const zero = builder.createConstant(
            `shrink_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );

        const [isNeg] = builder.createOp("Less", [X, negL], {}, boolMeta);
        const [isPos] = builder.createOp("Greater", [X, posL], {}, boolMeta);
        const [xPlusBias] = builder.createOp("Add", [X, bias], {}, outMeta);
        const [xMinusBias] = builder.createOp("Sub", [X, bias], {}, outMeta);

        const [posOrZero] = builder.createOp("Where", [isPos, xMinusBias, zero], {}, outMeta);
        const [out] = builder.createOp("Where", [isNeg, xPlusBias, posOrZero], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
