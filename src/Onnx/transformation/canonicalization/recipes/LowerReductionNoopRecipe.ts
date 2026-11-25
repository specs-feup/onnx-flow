import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import ConstantNode from "../../../ConstantNode.js";
import { getIntAttr, getIntsAttr, readConstIntegerVectorFromTensorNode } from "../../../Utils.js";

export const REDUCTION_NOOP_OP_TYPES = [
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
] as const;

/**
 * Canonicalizes Reduce* ops with `noop_with_empty_axes=1` and empty axes (Opset 13/18+)
 * into an Identity node.
 */
export class LowerReductionNoopRecipe implements DecompositionRecipe {
    public readonly name: string;
    public readonly targetOp: string;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 60;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["reduction", "noop"];
    public readonly producedOps = ["Identity"];

    constructor(targetOp: string = "ReduceSum") {
        this.targetOp = targetOp;
        this.name = `Lower${targetOp}Noop`;
    }

    match(op: OperationNode.Class): boolean {
        if (op.type !== this.targetOp) return false;
        if (getIntAttr(op, "noop_with_empty_axes", 0) !== 1) return false;
        const ins = op.getInputs();
        if (!ins || ins.length === 0) return false;

        const axesInput = ins.length > 1 && ins[1].id !== "" ? ins[1] : undefined;
        if (axesInput && !axesInput.is(ConstantNode)) {
            return false;
        }

        const axes = axesInput
            ? (readConstIntegerVectorFromTensorNode(axesInput.as(ConstantNode)) ?? [])
            : getIntsAttr(op, "axes", []);

        return axes.length === 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = (op.getInputs() as ConcreteValueNode[])[0];
        const Y = op.getOutputs()[0];
        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const [out] = builder.createOp("Identity", [X], {}, [
            { type: dtype, shape: Y.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        op.remove();
    }
}
