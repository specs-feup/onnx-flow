import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { TransformationOpportunity } from "../../TransformationOpportunity.js";

export class LowerExpandRecipe implements DecompositionRecipe {
    public readonly name = "LowerExpand";
    public readonly targetOp = "Expand";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["ConstantOfShape", "Cast", "Add"];

    match(op: OperationNode.Class): TransformationOpportunity | null {
        if (op.type !== "Expand") return null;
        return new TransformationOpportunity(
            this.name,
            op.id,
            "Lower Expand",
            (builder: GraphBuilder) => this.apply(op, builder),
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as (ConcreteValueNode | undefined)[];
        const X = ins[0]!;
        const shape = ins[1]!;
        const Y = op.getOutputs()[0];

        const dt =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const outShape =
            (Y.shape as KnownShape | undefined) !== undefined && Y.shape.length > 0
                ? Y.shape
                : X.shape;

        const expectedZeros = [{ type: DataType.FLOAT, shape: outShape as KnownShape }];
        let zeros = builder.createOp("ConstantOfShape", [shape], {}, expectedZeros)[0];

        if (dt !== DataType.FLOAT) {
            const expectedCast = [{ type: dt, shape: outShape as KnownShape }];
            zeros = builder.createOp("Cast", [zeros], { to: dt }, expectedCast)[0];
        }

        const expectedAdd = [{ type: dt, shape: outShape as KnownShape }];
        const addOp = builder.createOp("Add", [X, zeros], {}, expectedAdd)[0];

        builder.replaceAllUsesWith(Y, addOp);
        builder.removeNode(op);
    }
}
