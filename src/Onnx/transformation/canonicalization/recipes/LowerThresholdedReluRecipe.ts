import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerThresholdedReluRecipe implements DecompositionRecipe {
    public readonly name = "LowerThresholdedRelu";
    public readonly targetOp = "ThresholdedRelu";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Greater", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "ThresholdedRelu";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        const alphaVal = getFloatAttr(op, "alpha", 1.0);
        const alpha = builder.createConstant(
            `trelu_alpha_${op.id}`,
            makeTensorProto(dtype, [], [alphaVal]),
        );
        const zero = builder.createConstant(`trelu_zero_${op.id}`, makeTensorProto(dtype, [], [0]));

        const [cond] = builder.createOp("Greater", [X, alpha], {}, boolMeta);
        const [out] = builder.createOp("Where", [cond, X, zero], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
