import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { getIntAttr, int64Vec, toStaticShape } from "../../../utils/index.js";

export class LowerFlattenRecipe implements DecompositionRecipe {
    public readonly name = "LowerFlatten";
    public readonly targetOp = "Flatten";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["reshape-based"];
    public readonly producedOps = ["Reshape", "Shape", "Slice", "ReduceProd", "Concat"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Flatten") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length === 0) return false;
        return true;
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

        const dim0Static = leftKnown ? leftDims.reduce((a, b) => a * b, 1) : -1;
        const dim1Static = rightKnown ? rightDims.reduce((a, b) => a * b, 1) : -1;

        let targetShape: ConcreteValueNode;
        if (leftKnown || rightKnown) {
            targetShape = builder.createConstant(
                `flatten_shape_${op.id}`,
                int64Vec([dim0Static, dim1Static]),
            );
        } else {
            const [shapeX] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            const ax0 = builder.createConstant(`flat_ax0_${op.id}`, int64Vec([0]));
            const zeroV = builder.createConstant(`flat_z_${op.id}`, int64Vec([0]));
            const axisV = builder.createConstant(`flat_ax_${op.id}`, int64Vec([axis]));
            const rankV = builder.createConstant(`flat_rk_${op.id}`, int64Vec([rank]));

            const [leftSlice] = builder.createOp("Slice", [shapeX, zeroV, axisV, ax0], {}, [
                { type: DataType.INT64, shape: [axis] },
            ]);
            const [rightSlice] = builder.createOp("Slice", [shapeX, axisV, rankV, ax0], {}, [
                { type: DataType.INT64, shape: [rank - axis] },
            ]);
            const [leftProd] = builder.createOp("ReduceProd", [leftSlice, ax0], { keepdims: 1 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const [rightProd] = builder.createOp("ReduceProd", [rightSlice, ax0], { keepdims: 1 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            [targetShape] = builder.createOp("Concat", [leftProd, rightProd], { axis: 0 }, [
                { type: DataType.INT64, shape: [2] },
            ]);
        }
        const outShape = (
            Array.isArray(Y.shape) && Y.shape.length === 2 ? Y.shape : [dim0Static, dim1Static]
        ) as KnownShape;
        const [out] = builder.createOp("Reshape", [X, targetShape], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
