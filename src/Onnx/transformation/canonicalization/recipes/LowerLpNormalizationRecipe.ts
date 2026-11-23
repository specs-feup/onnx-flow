import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, normalizeAxis } from "../../../Utils.js";

export class LowerLpNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerLpNormalization";
    public readonly targetOp = "LpNormalization";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["ReduceL1", "ReduceL2", "Div"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LpNormalization") return false;
        const p = getIntAttr(op, "p", 2);
        return p === 1 || p === 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const rank = Array.isArray(X.shape) ? X.shape.length : 0;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const p = getIntAttr(op, "p", 2);
        const axesConst = builder.createConstant(`lpnorm_axes_${op.id}`, int64Vec([axis]));

        const reduceOp = p === 1 ? "ReduceL1" : "ReduceL2";
        const [norm] = builder.createOp(reduceOp, [X, axesConst], { keepdims: 1 });
        const [out] = builder.createOp("Div", [X, norm]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
