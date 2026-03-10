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
        if (schema?.category !== OpCategory.ElementWise) return false;

        const inputs = op.getInputs() ?? [];
        if (
            inputs.length > 0 &&
            inputs.every((inp) => inp && inp.shape && inp.shape.length === 0)
        ) {
            return false;
        }

        return true;
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
        const builder = new GraphBuilder(body, `ew_${op.id}`);

        // 1. Resolve scalar inputs (safely handling optional/undefined inputs)
        const inputs = op
            .getInputs()!
            .map((inp) =>
                inp ? resolveRecipeInput(builder, inp, valueMap, iter, axes, outShape) : undefined,
            );

        // Turn [1] -> [] (pure scalar) to match element-wise expectations in the loop body
        const effInputs = inputs.map((inp, i) =>
            inp ? squeezeIfLen1(builder, inp, axes, `${op.id}_in${i}_scalar`) : undefined,
        );

        // 2. Perform the operation on the scalars
        // Pass op.attributes to preserve required attributes (like 'to' for Cast, 'alpha' for LeakyRelu)
        const [out] = builder.createOp(
            op.type,
            effInputs.filter((inp) => inp !== undefined),
            op.attributes,
        );

        return out!;
    }
}
