import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput, squeezeIfLen1 } from "../RecipeUtils.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerElementWiseRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        return schema?.category === OpCategory.ElementWise;
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): ValueNode {
        const builder = new GraphBuilder(body, `lowering_${op.id}`);

        // 1. Resolve scalar inputs
        const inputs = op
            .getInputs()!
            .map((inp) => resolveRecipeInput(builder, inp, valueMap, iter, axes, outShape));

        // Turn [1] -> [] (pure scalar) to match element-wise expectations in the loop body
        const effInputs = inputs.map((inp, i) =>
            squeezeIfLen1(builder, inp, axes, `${op.id}_in${i}_scalar`),
        );

        // 2. Perform the operation on the scalars
        // createOp handles node creation, wiring, and shape inference automatically.
        const [out] = builder.createOp(op.type, effInputs);

        return out;
    }
}
