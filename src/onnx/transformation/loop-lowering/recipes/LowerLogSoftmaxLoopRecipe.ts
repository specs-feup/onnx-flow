import type OperationNode from "../../../ir/OperationNode.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { IndexingMode } from "../../../../DecompositionOptions.js";
import { lowerOnlineSoftmaxFamily } from "./LowerSoftmaxLoopRecipe.js";

/**
 * Online nested-loop LogSoftmax lowering reusing the Softmax row-loop builder
 * without emitting Exp in the second normalization pass.
 */
export class LowerLogSoftmaxLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerLogSoftmaxLoop";
    public readonly targetOp = "LogSoftmax";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["logsoftmax", "online-fused", "nested-loop"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Transpose",
        "Reshape",
        "Concat",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "ReduceProd",
        "Identity",
        "Max",
        "Sub",
        "Exp",
        "Mul",
        "Add",
        "Log",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LogSoftmax") return false;
        const X = op.getInputs()?.[0];
        return !!X && Array.isArray(X.shape) && X.shape.length >= 1;
    }

    apply(): never {
        throw new Error("LowerLogSoftmaxLoop uses lowerStandalone");
    }

    lowerStandalone(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void {
        lowerOnlineSoftmaxFamily(op, outerBuilder, indexingMode, true);
    }
}
