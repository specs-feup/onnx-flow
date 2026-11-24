import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, toStaticShape } from "../../../Utils.js";

export class LowerFlattenRecipe implements DecompositionRecipe {
    public readonly name = "LowerFlatten";
    public readonly targetOp = "Flatten";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic-batch" as const;
    public readonly tags = ["reshape-based"];
    public readonly producedOps = ["Reshape"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Flatten") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length === 0) return false;

        const inShape = toStaticShape(X.shape);
        const rank = inShape.length;
        let axis = getIntAttr(op, "axis", 1);
        if (axis < 0) axis += rank;
        axis = Math.max(0, Math.min(rank, axis));

        const leftDynamic = inShape.slice(0, axis).includes(-1);
        const rightDynamic = inShape.slice(axis).includes(-1);
        return !(leftDynamic && rightDynamic);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const inShape = toStaticShape(X.shape);
        const rank = inShape.length;

        let axis = getIntAttr(op, "axis", 1);
        if (axis < 0) axis += rank;
        axis = Math.max(0, Math.min(rank, axis));

        const leftDims = inShape.slice(0, axis);
        const rightDims = inShape.slice(axis);
        const leftKnown = !leftDims.includes(-1);
        const rightKnown = !rightDims.includes(-1);

        const dim0 = leftKnown ? leftDims.reduce((a, b) => a * b, 1) : -1;
        const dim1 = rightKnown ? rightDims.reduce((a, b) => a * b, 1) : -1;

        const targetShape = builder.createConstant(
            `flatten_shape_${op.id}`,
            int64Vec([dim0, dim1]),
        );
        const outShape = (
            Array.isArray(Y.shape) && Y.shape.length === 2 ? Y.shape : [dim0, dim1]
        ) as KnownShape;
        const [out] = builder.createOp("Reshape", [X, targetShape], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
