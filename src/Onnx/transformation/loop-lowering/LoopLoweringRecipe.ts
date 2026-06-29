import type OperationNode from "../../OperationNode.js";
import type OnnxGraph from "../../OnnxGraph.js";
import type { ConcreteValueNode, ValueNode, KnownShape } from "../../OnnxTypes.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { OpCategory } from "../../Schema/OpSchema.js";

export type RecipeApplyResult = ValueNode | { resultNode: ValueNode; nextCarry: ValueNode };

export interface LoopLoweringRecipe {
    readonly name: string;

    /** The specific OpType (e.g., "MatMul") or a broader OpCategory (e.g., OpCategory.ElementWise) */
    readonly targetOp: string | OpCategory;

    /** Metadata for the Orchestrator to make optimization decisions */
    readonly exposesControlFlow: boolean;
    readonly exposesDataAccess: boolean;
    readonly producedOps: string[];

    /**
     * Checks if this recipe can safely lower the given operation into scalar math.
     */
    canApply(op: OperationNode.Class): boolean;

    /**
     * Optional: Defines the total iterations and carry state shape for the loop.
     * If omitted, the pass defaults to `totalIters = product(outShape)`
     * and `carryShape = [totalIters]`.
     */
    getLoopBounds?(
        op: OperationNode.Class,
        outShape: KnownShape,
    ): {
        totalIters: number | ValueNode;
        carryShape: KnownShape | ValueNode;
        targetShape?: KnownShape | ValueNode;
    };

    /**
     * Optional: Performs operations on the loop output before final reshaping.
     * Useful for ops like ReduceL2 (Sqrt) or ReduceLogSum (Log).
     */
    postProcess?(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode;

    /**
     * Generates scalar mathematics for the operation inside a loop body.
     * Can return just the result scalar, or an object containing the result scalar
     * and a custom carry update tensor.
     */
    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
    ): RecipeApplyResult;
}
