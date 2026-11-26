import type OperationNode from "../../../OperationNode.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { IndexingMode } from "../../../../DecompositionOptions.js";
import { lowerRowNormFamily } from "./LowerLayerNormLoopRecipe.js";

/**
 * Nested-loop lowering for RMSNormalization (Opset 23) reusing the row-normalization
 * loop builder without mean subtraction and honoring `stash_type`.
 */
export class LowerRMSNormLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerRMSNormLoop";
    public readonly targetOp = "RMSNormalization";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["normalization", "rmsnorm", "nested-loop"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Concat",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "ReduceProd",
        "Identity",
        "Cast",
        "Sub",
        "Add",
        "Mul",
        "Sqrt",
        "Div",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "RMSNormalization") return false;
        const ins = op.getInputs();
        return !!ins && ins.length >= 2 && Array.isArray(ins[0].shape) && ins[0].shape.length >= 1;
    }

    apply(): never {
        throw new Error("LowerRMSNormLoop uses lowerStandalone");
    }

    lowerStandalone(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void {
        lowerRowNormFamily(op, outerBuilder, indexingMode, true);
    }
}
