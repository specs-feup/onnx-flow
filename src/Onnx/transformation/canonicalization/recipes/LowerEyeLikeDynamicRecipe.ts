import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, makeTensorProto, scalarInt64 } from "../../../Utils.js";

export class LowerEyeLikeDynamicRecipe implements DecompositionRecipe {
    public readonly name = "LowerEyeLikeDynamic";
    public readonly targetOp = "EyeLike";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["generator", "dynamic-matrix"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Range",
        "Unsqueeze",
        "Add",
        "Equal",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "EyeLike") return false;
        const X = op.getInputs()?.[0];
        return X !== undefined && Array.isArray(X.shape) && X.shape.length === 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const attrDtype = getIntAttr(op, "dtype", 0);
        const outDtype = (
            attrDtype !== 0 ? attrDtype : X.literalType || DataType.FLOAT
        ) as DataType;
        const outShape = Y.shape as KnownShape;
        const k = getIntAttr(op, "k", 0);

        const [shapeX] = builder.createOp("Shape", [X], {}, [{ type: DataType.INT64, shape: [2] }]);
        const idx0 = builder.createConstant(`eyed_i0_${op.id}`, scalarInt64(0));
        const idx1 = builder.createConstant(`eyed_i1_${op.id}`, scalarInt64(1));
        const [mDim] = builder.createOp("Gather", [shapeX, idx0], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);
        const [nDim] = builder.createOp("Gather", [shapeX, idx1], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);

        const startC = builder.createConstant(`eyed_s_${op.id}`, scalarInt64(0));
        const deltaC = builder.createConstant(`eyed_d_${op.id}`, scalarInt64(1));
        const [rows1D] = builder.createOp("Range", [startC, mDim, deltaC]);
        const [cols1D] = builder.createOp("Range", [startC, nDim, deltaC]);

        const ax1 = builder.createConstant(`eyed_ax1_${op.id}`, int64Vec([1]));
        const ax0 = builder.createConstant(`eyed_ax0_${op.id}`, int64Vec([0]));
        const [rows2D] = builder.createOp("Unsqueeze", [rows1D, ax1]);
        const [cols2D] = builder.createOp("Unsqueeze", [cols1D, ax0]);

        let rowsShifted: ConcreteValueNode = rows2D;
        if (k !== 0) {
            const kC = builder.createConstant(`eyed_k_${op.id}`, scalarInt64(k));
            [rowsShifted] = builder.createOp("Add", [rows2D, kC]);
        }

        const [mask] = builder.createOp("Equal", [cols2D, rowsShifted], {}, [
            { type: DataType.BOOL, shape: outShape },
        ]);
        const oneC = builder.createConstant(`eyed_1_${op.id}`, makeTensorProto(outDtype, [], [1]));
        const zeroC = builder.createConstant(`eyed_0_${op.id}`, makeTensorProto(outDtype, [], [0]));
        const [out] = builder.createOp("Where", [mask, oneC, zeroC], {}, [
            { type: outDtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
