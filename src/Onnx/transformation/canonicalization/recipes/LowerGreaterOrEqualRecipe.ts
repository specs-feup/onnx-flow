import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerGreaterOrEqualRecipe implements DecompositionRecipe {
    public readonly name = "LowerGreaterOrEqual";
    public readonly targetOp = "GreaterOrEqual";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Greater", "Equal", "Or"];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "GreaterOrEqual") return false;

    const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = op.getOutputs()[0];

        //Expected output type is BOOL
        const OutType = DataType.BOOL;

        const OutShape = (Y.shape as KnownShape) ?? (A.shape as KnownShape);

        const Output = [{ type: OutType, shape: OutShape }];

        // 1. Compare A > B
        const GreaterOut = builder.createOp("Greater", [A, B], {}, Output)[0];

        // 2. Compare A == B
        const EqualOut = builder.createOp("Equal", [A, B], {}, Output)[0];

        // 3. OR
        // Create the OR operation
        const OrOut = builder.createOp("Or", [GreaterOut, EqualOut], {}, Output)[0];

        builder.replaceAllUsesWith(Y, OrOut);
        op.remove();
    }
}