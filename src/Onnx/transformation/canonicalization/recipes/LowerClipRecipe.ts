import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";

export class LowerClipRecipe implements DecompositionRecipe {
    public readonly name = "LowerClip";
    public readonly targetOp = "Clip";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Max", "Min", "Identity"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Clip";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as (ConcreteValueNode | undefined)[];
        const X = ins[0]!;
        const minT = ins.length > 1 ? ins[1] : undefined;
        const maxT = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        let cur = X;

        // if (min) cur = Max(cur, min)
        if (minT) {
            cur = builder.createOp("Max", [cur, minT])[0];
        }

        // if (max) cur = Min(cur, max)
        if (maxT) {
            cur = builder.createOp("Min", [cur, maxT])[0];
        }

        // Degenerate case: no min, no max
        if (cur === X) {
            cur = builder.createOp("Identity", [X])[0];
        }

        // Safely replace Y and clean up
        builder.replaceAllUsesWith(Y, cur);
        op.remove();
    }
}
