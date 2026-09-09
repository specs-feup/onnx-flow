import type OperationNode from "../../OperationNode.js";
import type OnnxGraph from "../../OnnxGraph.js";
import type { ConcreteValueNode, ValueNode, KnownShape } from "../../OnnxTypes.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { BaseRecipe } from "../Recipe.js";

export type RecipeApplyResult = ValueNode | { resultNode: ValueNode; nextCarry: ValueNode };

export interface LoopLoweringRecipe extends BaseRecipe {
    /**
     * Optional: Defines the total iterations and carry state shape for the loop.
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
     */
    postProcess?(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode;

    /**
     * Generates scalar mathematics for the operation inside a loop body.
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
