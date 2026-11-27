import type OperationNode from "../../ir/OperationNode.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import type { OpCategory } from "../../schema/OpSchema.js";
import type { ValueNode } from "../../ir/OnnxTypes.js";
import type {
    DecompositionRecipe,
    RecipePhase,
    RecipePrecision,
    RecipeShapeSupport,
} from "../Recipe.js";

/**
 * Optional abstract base class for canonicalization recipes, providing standard
 * metadata defaults and a `replaceWith` helper for single-output rewrites.
 */
export abstract class BaseCanonicalizationRecipe implements DecompositionRecipe {
    public abstract readonly name: string;
    public abstract readonly targetOp: string | OpCategory;
    public abstract readonly producedOps: string[];

    public readonly phase: RecipePhase = "canonicalization";
    public readonly precision: RecipePrecision = "exact";
    public readonly priority: number = 0;
    public readonly shapeSupport: RecipeShapeSupport = "dynamic";
    public readonly exposesControlFlow: boolean = false;
    public readonly exposesDataAccess: boolean = false;
    public readonly tags?: readonly string[];

    public abstract match(node: OperationNode.Class): boolean;
    public abstract apply(node: OperationNode.Class, builder: GraphBuilder): void;

    /** Replaces `node`'s primary output with `replacement` and removes `node` from the graph. */
    protected replaceWith(
        node: OperationNode.Class,
        replacement: ValueNode,
        builder: GraphBuilder,
    ): void {
        const outputs = node.getOutputs();
        if (outputs.length > 0) {
            builder.replaceAllUsesWith(outputs[0], replacement);
        }
        builder.removeNode(node);
    }
}
