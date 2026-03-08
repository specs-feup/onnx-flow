import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput } from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerRangeRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        return op.type === "Range";
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `range_${op.id}`);
        const inputs = op.getInputs()!;

        // 1. Resolve start and delta as scalars (auto-broadcast/gather)
        const start = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            true,
            true,
        );
        const delta = resolveRecipeInput(
            builder,
            inputs[2],
            valueMap,
            iter,
            axes,
            outShape,
            true,
            true,
        );

        const dtype = start.literalType || DataType.FLOAT;

        // 2. Cast current iteration index to the target data type
        const [iterCast] = builder.createOp("Cast", [iter], { to: dtype });

        // 3. Compute current value: start + (iter * delta)
        const [iterStep] = builder.createOp("Mul", [iterCast, delta]);
        const [currentVal] = builder.createOp("Add", [start, iterStep]);

        return currentVal;
    }
}
