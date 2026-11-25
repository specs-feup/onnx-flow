import type OperationNode from "../OperationNode.js";
import type { GraphBuilder } from "../GraphBuilder.js";
import type { OpCategory } from "../Schema/OpSchema.js";

export type RecipePhase = "canonicalization" | "loop-lowering" | "optimization" | "custom";
export type RecipePrecision = "exact" | "approximation";

export interface RecipeMetadata {
    readonly name: string;
    readonly targetOp: string | OpCategory;
    readonly phase: RecipePhase;
    readonly precision: RecipePrecision;
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];
}

// 1. The Common Ancestor
export interface BaseRecipe {
    readonly name: string;
    readonly targetOp: string | OpCategory;
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];
    /** Optional explicit phase (inferred from exposesControlFlow if omitted) */
    readonly phase?: RecipePhase;
    /** Optional explicit precision (defaults to 'exact' unless specified or in known approximations) */
    readonly precision?: RecipePrecision;

    // Pure boolean match
    match(node: OperationNode.Class): boolean;
}

// 2. The Decomposition Contract
export interface DecompositionRecipe extends BaseRecipe {
    /** Perform the global graph rewrite */
    apply(node: OperationNode.Class, builder: GraphBuilder): void;
}

/** Known polynomial/Taylor/piecewise approximation recipes in the standard library */
const KNOWN_APPROXIMATION_RECIPES = new Set<string>([
    "LowerErf",
    "LowerSin",
    "LowerCos",
    "LowerAsin",
    "LowerAtan",
    "LowerLog",
    "LowerSqrt",
    "LowerPow",
    "LowerReciprocal",
    "LowerExp",
]);

/**
 * Normalizes metadata for any BaseRecipe instance, filling in inferred phase and precision
 * if the recipe class does not explicitly declare them yet.
 */
export function getRecipeMetadata(recipe: BaseRecipe): RecipeMetadata {
    const phase: RecipePhase =
        recipe.phase ?? (recipe.exposesControlFlow ? "loop-lowering" : "canonicalization");
    const precision: RecipePrecision =
        recipe.precision ??
        (KNOWN_APPROXIMATION_RECIPES.has(recipe.name) ? "approximation" : "exact");

    return {
        name: recipe.name,
        targetOp: recipe.targetOp,
        phase,
        precision,
        exposesControlFlow: recipe.exposesControlFlow,
        exposesDataAccess: recipe.exposesDataAccess,
        producedOps: [...recipe.producedOps],
    };
}
