import type OperationNode from "../../OperationNode.js";
import type OnnxGraph from "../../OnnxGraph.js";
import type { ConcreteValueNode, ValueNode, KnownShape } from "../../OnnxTypes.js";
import { GraphBuilder } from "../../GraphBuilder.js";

export type RecipeApplyResult = ValueNode | { resultNode: ValueNode; nextCarry: ValueNode };

export interface LoopLoweringRecipe {
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
    ): { totalIters: number; carryShape: KnownShape };

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
    ): RecipeApplyResult;
}
