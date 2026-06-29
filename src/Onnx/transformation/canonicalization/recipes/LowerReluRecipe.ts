import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";
import { TransformationOpportunity } from "../../TransformationOpportunity.js";

export class LowerReluRecipe implements DecompositionRecipe {
    public readonly name = "LowerRelu";
    public readonly targetOp = "Relu";
    public readonly exposesControlFlow = true; // Exposes Where/MUX logic
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Greater", "Where"];

    match(op: OperationNode.Class): TransformationOpportunity | null {
        if (op.type !== "Relu") return null;
        return new TransformationOpportunity(
            this.name,
            op.id,
            "Lower Relu",
            (builder: GraphBuilder) => this.apply(op, builder),
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Fallback to FLOAT if undefined
        const dtype = (X.literalType as DataType | undefined) ?? DataType.FLOAT;

        // 1. Create a scalar '0' constant of the same type
        const zeroConst = builder.createConstant(
            `relu_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );

        // 2. Condition: Mask = Greater(X, 0)
        // Expected shape is the same as X, but dtype is BOOL
        const expectedGreater = [{ type: DataType.BOOL, shape: X.shape as KnownShape }];
        const greaterOut = builder.createOp("Greater", [X, zeroConst], {}, expectedGreater)[0];

        // 3. Selection: Out = Where(Mask, X, 0)
        // Where behaves like a MUX: if Mask[i] is true, pick X[i], else pick zeroConst
        const expectedWhere = [{ type: dtype, shape: X.shape as KnownShape }];
        const whereOut = builder.createOp(
            "Where",
            [greaterOut, X, zeroConst],
            {},
            expectedWhere,
        )[0];

        // 4. Safely replace the original Y with the new Where output
        builder.replaceAllUsesWith(Y, whereOut);
        op.remove();
    }
}
