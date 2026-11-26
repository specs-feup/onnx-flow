import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, normalizeAxis } from "../../../utils/index.js";

export class LowerLogSoftmaxRecipe implements DecompositionRecipe {
    public readonly name = "LowerLogSoftmax";
    public readonly targetOp = "LogSoftmax";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["ReduceMax", "Sub", "Exp", "ReduceSum", "Log"];

    match(op: OperationNode.Class): boolean {
        return op.type === "LogSoftmax";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const rank = Array.isArray(X.shape) ? X.shape.length : 0;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const axesConst = builder.createConstant(`lsm_axes_${op.id}`, int64Vec([axis]));

        const [maxVal] = builder.createOp("ReduceMax", [X, axesConst], { keepdims: 1 });
        const [shifted] = builder.createOp("Sub", [X, maxVal]);
        const [expVal] = builder.createOp("Exp", [shifted]);
        const [sumExp] = builder.createOp("ReduceSum", [expVal, axesConst], { keepdims: 1 });
        const [logSumExp] = builder.createOp("Log", [sumExp]);
        const [out] = builder.createOp("Sub", [shifted, logSumExp]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
