import type OperationNode from "../../OperationNode.js";
import type OnnxGraph from "../../OnnxGraph.js";
import type { ConcreteValueNode, ValueNode, KnownShape } from "../../OnnxTypes.js";

export interface LoopLoweringRecipe {
    /**
     * Checks if this recipe can safely lower the given operation into scalar math.
     * Use this to reject ops with unsupported attributes or data types.
     */
    canApply(op: OperationNode.Class): boolean;

    /**
     * Generates scalar mathematics for the operation inside a loop body.
     * MUST return the scalar ValueNode representing the result.
     */
    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape
    ): ValueNode;
}