import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { getIntAttr, int64Vec, toStaticShape } from "../../../utils/index.js";

/**
 * Canonicalizes SpaceToDepth (DCR mode):
 *   [N, C, H, W] -> Reshape([N, C, H/b, b, W/b, b])
 *                -> Transpose(perm=[0, 3, 5, 1, 2, 4])
 *                -> Reshape([N, C*b*b, H/b, W/b])
 */
export class LowerSpaceToDepthRecipe implements DecompositionRecipe {
    public readonly name = "LowerSpaceToDepth";
    public readonly targetOp = "SpaceToDepth";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Div",
        "Mul",
        "Concat",
        "Reshape",
        "Transpose",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "SpaceToDepth") return false;
        const ins = op.getInputs();
        const b = getIntAttr(op, "blocksize", 0);
        return !!ins && ins.length >= 1 && ins[0].shape.length === 4 && b > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = (op.getInputs() as ConcreteValueNode[])[0];
        const Y = op.getOutputs()[0];
        const b = getIntAttr(op, "blocksize", 1);
        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const inStatic = toStaticShape(X.shape);
        const [N, C, H, W] = inStatic;
        const outH = H > 0 ? Math.floor(H / b) : -1;
        const outW = W > 0 ? Math.floor(W / b) : -1;
        const outC = C > 0 ? C * b * b : -1;

        const shape6D: KnownShape = [N, C, outH, b, outW, b];
        const transShape: KnownShape = [N, b, b, C, outH, outW];
        const finalShape: KnownShape = [N, outC, outH, outW];

        let shape6DNode: ConcreteValueNode;
        let finalShapeNode: ConcreteValueNode;

        if (!inStatic.includes(-1)) {
            shape6DNode = builder.createConstant(
                `s2d_s6_${op.id}`,
                int64Vec([N, C, outH, b, outW, b]),
            );
            finalShapeNode = builder.createConstant(
                `s2d_s4_${op.id}`,
                int64Vec([N, outC, outH, outW]),
            );
        } else {
            const [shapeX] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [4] },
            ]);
            const idx0 = builder.createConstant(`s2d_i0_${op.id}`, int64Vec([0]));
            const idx1 = builder.createConstant(`s2d_i1_${op.id}`, int64Vec([1]));
            const idx2 = builder.createConstant(`s2d_i2_${op.id}`, int64Vec([2]));
            const idx3 = builder.createConstant(`s2d_i3_${op.id}`, int64Vec([3]));
            const bVec = builder.createConstant(`s2d_bv_${op.id}`, int64Vec([b]));
            const b2Vec = builder.createConstant(`s2d_b2_${op.id}`, int64Vec([b * b]));

            const [nDim] = builder.createOp("Gather", [shapeX, idx0], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const [cDim] = builder.createOp("Gather", [shapeX, idx1], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const [hDim] = builder.createOp("Gather", [shapeX, idx2], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const [wDim] = builder.createOp("Gather", [shapeX, idx3], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);

            const [hDiv] = builder.createOp("Div", [hDim, bVec], {}, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const [wDiv] = builder.createOp("Div", [wDim, bVec], {}, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const [cMul] = builder.createOp("Mul", [cDim, b2Vec], {}, [
                { type: DataType.INT64, shape: [1] },
            ]);

            [shape6DNode] = builder.createOp(
                "Concat",
                [nDim, cDim, hDiv, bVec, wDiv, bVec],
                { axis: 0 },
                [{ type: DataType.INT64, shape: [6] }],
            );
            [finalShapeNode] = builder.createOp("Concat", [nDim, cMul, hDiv, wDiv], { axis: 0 }, [
                { type: DataType.INT64, shape: [4] },
            ]);
        }

        const [x6D] = builder.createOp("Reshape", [X, shape6DNode], {}, [
            { type: dtype, shape: shape6D },
        ]);
        const [xTrans] = builder.createOp("Transpose", [x6D], { perm: [0, 3, 5, 1, 2, 4] }, [
            { type: dtype, shape: transShape },
        ]);
        const [out] = builder.createOp("Reshape", [xTrans, finalShapeNode], {}, [
            { type: dtype, shape: finalShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        op.remove();
    }
}
