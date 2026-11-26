import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asConcreteValueNode,
    getIntAttr,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";

export class LowerEyeLikeRecipe implements DecompositionRecipe {
    public readonly name = "LowerEyeLike";
    public readonly targetOp = "EyeLike";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["generator"];
    public readonly producedOps = ["Range", "Reshape", "Add", "Equal", "Where"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "EyeLike") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length !== 2) return false;
        const [M, N] = toStaticShape(X.shape);
        return M > 0 && N > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = asConcreteValueNode(op.getInputs()![0]);
        const Y = op.getOutputs()[0];
        const attrDtype = getIntAttr(op, "dtype", 0);
        const xDtype = X.literalType !== DataType.UNDEFINED ? X.literalType : DataType.FLOAT;
        const outDtype: DataType = attrDtype > 0 ? (attrDtype as DataType) : xDtype;

        const [M, N] = toStaticShape(X.shape);
        const k = getIntAttr(op, "k", 0);

        const zero = builder.createConstant(`eye_zero_${op.id}`, scalarInt64(0));
        const one = builder.createConstant(`eye_one_${op.id}`, scalarInt64(1));
        const mConst = builder.createConstant(`eye_m_${op.id}`, scalarInt64(M));
        const nConst = builder.createConstant(`eye_n_${op.id}`, scalarInt64(N));

        const [rowRange] = builder.createOp("Range", [zero, mConst, one], {}, [
            { type: DataType.INT64, shape: [M] },
        ]);
        const [colRange] = builder.createOp("Range", [zero, nConst, one], {}, [
            { type: DataType.INT64, shape: [N] },
        ]);

        const rowShape = builder.createConstant(`eye_rshape_${op.id}`, int64Vec([M, 1]));
        const colShape = builder.createConstant(`eye_cshape_${op.id}`, int64Vec([1, N]));

        let [rows2D] = builder.createOp("Reshape", [rowRange, rowShape], {}, [
            { type: DataType.INT64, shape: [M, 1] },
        ]);
        const [cols2D] = builder.createOp("Reshape", [colRange, colShape], {}, [
            { type: DataType.INT64, shape: [1, N] },
        ]);

        if (k !== 0) {
            const kConst = builder.createConstant(`eye_k_${op.id}`, scalarInt64(k));
            [rows2D] = builder.createOp("Add", [rows2D, kConst], {}, [
                { type: DataType.INT64, shape: [M, 1] },
            ]);
        }

        const outShape = [M, N] as KnownShape;
        const [mask] = builder.createOp("Equal", [rows2D, cols2D], {}, [
            { type: DataType.BOOL, shape: outShape },
        ]);
        const oneVal = builder.createConstant(
            `eye_one_val_${op.id}`,
            makeTensorProto(outDtype, [], [1]),
        );
        const zeroVal = builder.createConstant(
            `eye_zero_val_${op.id}`,
            makeTensorProto(outDtype, [], [0]),
        );
        const [out] = builder.createOp("Where", [mask, oneVal, zeroVal], {}, [
            { type: outDtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
