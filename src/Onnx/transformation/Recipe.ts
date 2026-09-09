import type OperationNode from "../OperationNode.js";
import type { GraphBuilder } from "../GraphBuilder.js";
import type { OpCategory } from "../Schema/OpSchema.js";

// 1. The Common Ancestor
export interface BaseRecipe {
    readonly name: string;
    readonly targetOp: string | OpCategory;
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];

    // Pure boolean match
    match(node: OperationNode.Class): boolean;
}

// 2. The Decomposition Contract
export interface DecompositionRecipe extends BaseRecipe {
    /** Perform the global graph rewrite */
    apply(node: OperationNode.Class, builder: GraphBuilder): void;
}
