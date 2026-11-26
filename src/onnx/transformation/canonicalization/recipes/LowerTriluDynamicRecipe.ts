import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, makeTensorProto, scalarInt64 } from "../../../utils/index.js";

export class LowerTriluDynamicRecipe implements DecompositionRecipe {
    public readonly name = "LowerTriluDynamic";
    public readonly targetOp = "Trilu";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["mask-based", "dynamic-matrix"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Range",
        "Unsqueeze",
        "Cast",
        "Add",
        "GreaterOrEqual",
        "LessOrEqual",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Trilu") return false;
        const X = op.getInputs()?.[0];
        return X !== undefined && Array.isArray(X.shape) && X.shape.length >= 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs()! as ConcreteValueNode[];
        const X = ins[0];
        const kNode = ins.length >= 2 && ins[1].id !== "" ? ins[1] : undefined;
        const Y = op.getOutputs()[0];

        const dtype = (X.literalType || DataType.FLOAT) as DataType;
        const outShape = Y.shape as KnownShape;
        const rank = X.shape.length;
        const upper = getIntAttr(op, "upper", 1) !== 0;

        const [shapeX] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const mIdx = builder.createConstant(`trilud_mi_${op.id}`, scalarInt64(rank - 2));
        const nIdx = builder.createConstant(`trilud_ni_${op.id}`, scalarInt64(rank - 1));
        const [mDim] = builder.createOp("Gather", [shapeX, mIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);
        const [nDim] = builder.createOp("Gather", [shapeX, nIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);

        const startC = builder.createConstant(`trilud_s_${op.id}`, scalarInt64(0));
        const deltaC = builder.createConstant(`trilud_d_${op.id}`, scalarInt64(1));
        const [rows1D] = builder.createOp("Range", [startC, mDim, deltaC]);
        const [cols1D] = builder.createOp("Range", [startC, nDim, deltaC]);

        const ax1 = builder.createConstant(`trilud_ax1_${op.id}`, int64Vec([1]));
        const ax0 = builder.createConstant(`trilud_ax0_${op.id}`, int64Vec([0]));
        const [rows2D] = builder.createOp("Unsqueeze", [rows1D, ax1]);
        const [cols2D] = builder.createOp("Unsqueeze", [cols1D, ax0]);

        let rowsShifted: ConcreteValueNode = rows2D;
        if (kNode) {
            const k64: ConcreteValueNode =
                kNode.literalType === DataType.INT64
                    ? kNode
                    : builder.createOp("Cast", [kNode], { to: DataType.INT64 })[0];
            [rowsShifted] = builder.createOp("Add", [rows2D, k64]);
        }

        const cmpOp = upper ? "GreaterOrEqual" : "LessOrEqual";
        const [mask2D] = builder.createOp(cmpOp, [cols2D, rowsShifted]);

        const zeroC = builder.createConstant(`trilud_0_${op.id}`, makeTensorProto(dtype, [], [0]));
        const [out] = builder.createOp("Where", [mask2D, X, zeroC], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
