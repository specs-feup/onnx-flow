import type OperationNode from "../../ir/OperationNode.js";
import type OnnxGraph from "../../ir/OnnxGraph.js";
import type { ConcreteValueNode, ValueNode, KnownShape, DataType } from "../../ir/OnnxTypes.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import type { BaseRecipe } from "../Recipe.js";
import type { IndexingMode } from "../../../DecompositionOptions.js";
import type TensorNode from "../../ir/TensorNode.js";

export type RecipeApplyResult = ValueNode | { resultNode: ValueNode; nextCarry: ValueNode };

/**
 * Context passed to loop recipes to register AGU pointers and counters as loop-carried states.
 */
export interface LoopCarryContext {
    readonly indexingMode: IndexingMode;
    readonly outerBuilder: GraphBuilder;
    readonly innerBuilder: GraphBuilder;
    /**
     * Registers a new loop-carried variable on the enclosing ONNX Loop.
     * Returns the inner-graph formal input TensorNode for the current iteration.
     */
    addCarry(
        tag: string,
        initialVal: ValueNode,
        elemTy: DataType,
        shape: KnownShape,
    ): TensorNode.Class;
    /**
     * Binds the next-iteration state node for a previously registered carry input.
     */
    setNextState(carryInputNode: TensorNode.Class, nextState: ConcreteValueNode): void;
}

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
     * Optional standalone lowering for hierarchical multi-loop ops (e.g., SoftmaxLoop, LayerNormLoop)
     * that operate row-wise rather than element-wise.
     */
    lowerStandalone?(
        op: OperationNode.Class,
        outerBuilder: GraphBuilder,
        indexingMode: IndexingMode,
    ): void;

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
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult;
}
