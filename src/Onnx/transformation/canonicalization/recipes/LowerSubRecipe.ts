import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerSubRecipe implements DecompositionRecipe {
    public readonly name = "LowerSub";
    public readonly targetOp = "Sub";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Neg", "Add"];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "Sub") return false;

        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        const A = ins[0];
        const dtype = A.literalType as DataType | undefined;

        // ONNX 'Neg' does not support unsigned integers.
        // We must abort the decomposition for these types to prevent downstream crashes.
        if (
            dtype === DataType.UINT8 ||
            dtype === DataType.UINT16 ||
            dtype === DataType.UINT32 ||
            dtype === DataType.UINT64
        ) {
            return false;
        }

        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = op.getOutputs()[0];

        // Fallback to FLOAT if undefined
        const dtype = (A.literalType as DataType | undefined) ?? DataType.FLOAT;

        // 1. Negate B
        // The shape of Neg(B) is exactly B's shape
        const expectedNeg = [{ type: dtype, shape: B.shape as KnownShape }];
        const BnegOut = builder.createOp("Neg", [B], {}, expectedNeg)[0];

        // 2. Add (A, -B)
        // Use Y.shape to inherit the correctly broadcasted shape inferred by the original Sub node.
        // We fallback to A.shape only as a safety net if Y.shape is missing.
        const outShape = (Y.shape as KnownShape | undefined) ?? (A.shape as KnownShape);
        const expectedAdd = [{ type: dtype, shape: outShape }];

        const AddOut = builder.createOp("Add", [A, BnegOut], {}, expectedAdd)[0];

        // 3. Safely replace original output with the new Add output
        builder.replaceAllUsesWith(Y, AddOut);
        op.remove();
    }
}
