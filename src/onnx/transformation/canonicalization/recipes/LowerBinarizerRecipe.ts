import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../utils/index.js";

export class LowerBinarizerRecipe implements DecompositionRecipe {
    public readonly name = "LowerBinarizer";
    public readonly targetOp = "Binarizer";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Greater", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Binarizer";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        const threshVal = getFloatAttr(op, "threshold", 0.0);
        const thresh = builder.createConstant(
            `bin_thresh_${op.id}`,
            makeTensorProto(dtype, [], [threshVal]),
        );
        const one = builder.createConstant(`bin_one_${op.id}`, makeTensorProto(dtype, [], [1]));
        const zero = builder.createConstant(`bin_zero_${op.id}`, makeTensorProto(dtype, [], [0]));

        const [cond] = builder.createOp("Greater", [X, thresh], {}, boolMeta);
        const [out] = builder.createOp("Where", [cond, one, zero], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
