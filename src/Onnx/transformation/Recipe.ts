import type OperationNode from "../OperationNode.js";
import type { GraphBuilder } from "../GraphBuilder.js";
import type { OpCategory } from "../Schema/OpSchema.js";
import type { TransformationOpportunity } from "./TransformationOpportunity.js";

export interface DecompositionRecipe {
    readonly name: string;

    /** The specific OpType (e.g., "MatMul") or a broader OpCategory (e.g., OpCategory.ElementWise) */
    readonly targetOp: string | OpCategory;

    /** Metadata for the Orchestrator to make optimization decisions */
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];

    /** Check if this specific node can/should be decomposed by this recipe right now */
    match(node: OperationNode.Class): TransformationOpportunity | null;

    /** Perform the actual decomposition using the GraphBuilder */
    apply(node: OperationNode.Class, builder: GraphBuilder): void;
}
