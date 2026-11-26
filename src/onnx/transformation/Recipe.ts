import type OperationNode from "../OperationNode.js";
import type { GraphBuilder } from "../GraphBuilder.js";
import type { OpCategory } from "../schema/OpSchema.js";

export type RecipePhase = "canonicalization" | "loop-lowering" | "optimization" | "custom";
export type RecipePrecision = "exact" | "approximation";
export type RecipeShapeSupport = "static" | "dynamic-batch" | "dynamic";

export interface RecipeMetadata {
    readonly name: string;
    readonly targetOp: string | OpCategory;
    readonly phase: RecipePhase;
    readonly precision: RecipePrecision;
    readonly priority: number;
    readonly shapeSupport: RecipeShapeSupport;
    readonly tags: readonly string[];
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];
}

// 1. The Common Ancestor
export interface BaseRecipe {
    readonly name: string;
    readonly targetOp: string | OpCategory;
    readonly phase: RecipePhase;
    readonly precision: RecipePrecision;
    readonly priority: number;
    readonly shapeSupport: RecipeShapeSupport;
    readonly tags?: readonly string[];
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];

    match(node: OperationNode.Class): boolean;
}

// 2. The Decomposition Contract
export interface DecompositionRecipe extends BaseRecipe {
    /** Perform the global graph rewrite */
    apply(node: OperationNode.Class, builder: GraphBuilder): void;
}

/**
 * Normalizes metadata for any BaseRecipe instance, filling in inferred phase and precision
 * if the recipe class does not explicitly declare them yet.
 */
export function getRecipeMetadata(recipe: BaseRecipe): RecipeMetadata {
    return {
        name: recipe.name,
        targetOp: recipe.targetOp,
        phase: recipe.phase,
        precision: recipe.precision,
        priority: recipe.priority,
        shapeSupport: recipe.shapeSupport,
        tags: [...(recipe.tags ?? [])],
        exposesControlFlow: recipe.exposesControlFlow,
        exposesDataAccess: recipe.exposesDataAccess,
        producedOps: [...recipe.producedOps],
    };
}
