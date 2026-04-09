import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";

export class LowerMaxRecipe implements DecompositionRecipe {
    public readonly name = "LowerMax";
    public readonly targetOp = "Max";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Where","Greater"];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "Max") return false;

    const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        return true;
    }

    apply(node: OperationNode.Class, builder: GraphBuilder): void {
        const ins = node.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = node.getOutputs()[0];

        //Expected output type is float and shape is broadcasted shape of A and B
        const OutType = A.literalType as DataType | undefined ?? DataType.FLOAT;
        const OutShape = (Y.shape as KnownShape) ?? (A.shape as KnownShape);
        const Output = [{ type: OutType, shape: OutShape }];

        //Greater(A,B)
        const GreaterOut = builder.createOp("Greater", [A, B], {}, Output)[0];

        //Where(GreaterOut)
        const WhereOut = builder.createOp("Where", [GreaterOut, A, B],{},Output)[0];

        builder.replaceAllUsesWith(Y, WhereOut);
        node.remove();
    }
}