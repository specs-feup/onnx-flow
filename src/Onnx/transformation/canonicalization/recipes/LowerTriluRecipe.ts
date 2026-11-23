import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";

export class LowerTriluRecipe implements DecompositionRecipe {
    public readonly name = "LowerTrilu";
    public readonly targetOp = "Trilu";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Range",
        "Reshape",
        "Add",
        "GreaterOrEqual",
        "LessOrEqual",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Trilu") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 2) return false;
        const s = toStaticShape(X.shape);
        return s[s.length - 2] > 0 && s[s.length - 1] > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const kNode = ins.length >= 2 ? ins[1] : undefined;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const inShape = toStaticShape(X.shape);
        const M = inShape[inShape.length - 2];
        const N = inShape[inShape.length - 1];
        const upper = getIntAttr(op, "upper", 1);

        const zeroI64 = builder.createConstant(`trilu_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`trilu_o64_${op.id}`, scalarInt64(1));
        const mConst = builder.createConstant(`trilu_m_${op.id}`, scalarInt64(M));
        const nConst = builder.createConstant(`trilu_n_${op.id}`, scalarInt64(N));

        const [rowRange] = builder.createOp("Range", [zeroI64, mConst, oneI64], {}, [
            { type: DataType.INT64, shape: [M] },
        ]);
        const [colRange] = builder.createOp("Range", [zeroI64, nConst, oneI64], {}, [
            { type: DataType.INT64, shape: [N] },
        ]);

        const rowShape = builder.createConstant(`trilu_rshape_${op.id}`, int64Vec([M, 1]));
        const colShape = builder.createConstant(`trilu_cshape_${op.id}`, int64Vec([1, N]));

        let [rows2D] = builder.createOp("Reshape", [rowRange, rowShape], {}, [
            { type: DataType.INT64, shape: [M, 1] },
        ]);
        const [cols2D] = builder.createOp("Reshape", [colRange, colShape], {}, [
            { type: DataType.INT64, shape: [1, N] },
        ]);

        if (kNode) {
            [rows2D] = builder.createOp("Add", [rows2D, kNode], {}, [
                { type: DataType.INT64, shape: [M, 1] },
            ]);
        }

        // upper=1: keep cols >= rows + k; upper=0: keep cols <= rows + k
        const cmpOp = upper === 1 ? "GreaterOrEqual" : "LessOrEqual";
        const [mask] = builder.createOp(cmpOp, [cols2D, rows2D], {}, [
            { type: DataType.BOOL, shape: [M, N] },
        ]);

        const zeroVal = builder.createConstant(
            `trilu_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const outShape = Y.shape as KnownShape;
        const [out] = builder.createOp("Where", [mask, X, zeroVal], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
