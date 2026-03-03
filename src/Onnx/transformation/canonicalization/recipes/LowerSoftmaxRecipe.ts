import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, makeTensorProto } from "../../../Utils.js";

export class LowerSoftmaxRecipe implements DecompositionRecipe {
    public readonly name = "LowerSoftmax";
    public readonly targetOp = "Softmax";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["ReduceMax", "Sub", "Exp", "ReduceSum", "Div"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Softmax";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const inShape = Array.isArray(X.shape) ? [...X.shape] : [];
        const rank = inShape.length;

        // Parse axis, correctly handling negative wrapping
        let axis = getIntAttr(op, "axis", -1);
        if (rank > 0 && axis < 0) {
            axis = (axis + rank) % rank;
        }

        // 1. M = ReduceMax(X, axes=[axis], keepdims=1)
        const M = builder.createOp("ReduceMax", [X], { keepdims: 1, axes: [axis] })[0];

        // 2. SH = Sub(X, M)
        const SH = builder.createOp("Sub", [X, M])[0];

        // 3. EX = Exp(SH)
        const EX = builder.createOp("Exp", [SH])[0];

        // 4. DEN = ReduceSum(EX, axes=[axis], keepdims=1)
        // Note: For ReduceSum, newer ONNX opsets use `axes` as an input rather than an attribute.
        const axesConst = builder.createConstant(
            `sm_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [axis]),
        );
        const DEN = builder.createOp("ReduceSum", [EX, axesConst], { keepdims: 1 })[0];

        // 5. Y = Div(EX, DEN)
        const finalY = builder.createOp("Div", [EX, DEN])[0];

        // Safely replace Y and clean up
        builder.replaceAllUsesWith(Y, finalY);
        op.remove();
    }
}
