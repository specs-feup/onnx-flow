import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    getIntsAttr,
    getStringAttr,
    int64Vec,
    toStaticShape,
} from "../../../utils/index.js";

export class LowerConv1x1ToMatMulRecipe implements DecompositionRecipe {
    public readonly name = "LowerConv1x1ToMatMul";
    public readonly targetOp = "Conv";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["pointwise-conv", "gemm"];
    public readonly producedOps = [
        "Reshape",
        "Transpose",
        "MatMul",
        "Add",
        "Shape",
        "Slice",
        "ReduceProd",
        "Unsqueeze",
        "Concat",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Conv") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        if (xDims.length < 3 || wDims.length !== xDims.length) return false;
        if (wDims[0] <= 0 || wDims[1] <= 0) return false;

        const spatialRank = xDims.length - 2;
        if (wDims.slice(2).some((k) => k !== 1)) return false;

        const kernelAttr = getIntsAttr(op, "kernel_shape", []);
        if (kernelAttr.length > 0 && kernelAttr.some((k) => k !== 1)) return false;

        const group = getIntAttr(op, "group", 1);
        if (group !== 1) return false;

        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        if (strides.some((s) => s !== 1)) return false;

        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        if (dilations.some((d) => d !== 1)) return false;

        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        if (autoPad === "NOTSET") {
            const pads = getIntsAttr(op, "pads", []);
            if (pads.some((p) => p !== 0)) return false;
        } else if (autoPad !== "VALID" && autoPad !== "SAME_UPPER" && autoPad !== "SAME_LOWER") {
            return false;
        }

        return true;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const W = ins[1];
        const B = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const xDims = toStaticShape(X.shape);
        const wDims = toStaticShape(W.shape);
        const yDims = toStaticShape(Y.shape);
        const rank = xDims.length;
        const M = wDims[0];
        const C = wDims[1];

        const spatialStatic = !xDims.slice(2).includes(-1);
        const sProdStatic = spatialStatic ? xDims.slice(2).reduce((a, b) => a * b, 1) : -1;

        const w2dShapeConst = builder.createConstant(`c1x1_w2d_${op.id}`, int64Vec([M, C]));
        const [W2d] = builder.createOp("Reshape", [W, w2dShapeConst], {}, [
            { type: dtype, shape: [M, C] },
        ]);

        let result: ConcreteValueNode;

        // Transpose-free fast path for batch size N = 1:
        // Reshape X [1, C, *S] -> [C, S_prod], compute MatMul([M, C], [C, S_prod]) = [M, S_prod].
        if (xDims[0] === 1) {
            let x2dShapeNode: ConcreteValueNode;
            let spatialVecNode: ConcreteValueNode | undefined;
            if (spatialStatic) {
                x2dShapeNode = builder.createConstant(
                    `c1x1_x2d_${op.id}`,
                    int64Vec([C, sProdStatic]),
                );
            } else {
                const [shapeX] = builder.createOp("Shape", [X], {}, [
                    { type: DataType.INT64, shape: [rank] },
                ]);
                const st = builder.createConstant(`c1x1_st2_${op.id}`, int64Vec([2]));
                const en = builder.createConstant(`c1x1_enr_${op.id}`, int64Vec([rank]));
                const ax0 = builder.createConstant(`c1x1_ax0_${op.id}`, int64Vec([0]));
                [spatialVecNode] = builder.createOp("Slice", [shapeX, st, en, ax0], {}, [
                    { type: DataType.INT64, shape: [rank - 2] },
                ]);
                const [sProd] = builder.createOp("ReduceProd", [spatialVecNode, ax0], {
                    keepdims: 1,
                });
                const cVec = builder.createConstant(`c1x1_cvec_${op.id}`, int64Vec([C]));
                [x2dShapeNode] = builder.createOp("Concat", [cVec, sProd], { axis: 0 }, [
                    { type: DataType.INT64, shape: [2] },
                ]);
            }

            const [X2d] = builder.createOp("Reshape", [X, x2dShapeNode], {}, [
                { type: dtype, shape: [C, sProdStatic] },
            ]);
            let [mm] = builder.createOp("MatMul", [W2d, X2d], {}, [
                { type: dtype, shape: [M, sProdStatic] },
            ]);

            if (B) {
                const b2dShape = builder.createConstant(`c1x1_b2d_${op.id}`, int64Vec([M, 1]));
                const [B2d] = builder.createOp("Reshape", [B, b2dShape], {}, [
                    { type: dtype, shape: [M, 1] },
                ]);
                [mm] = builder.createOp("Add", [mm, B2d], {}, [
                    { type: dtype, shape: [M, sProdStatic] },
                ]);
            }

            let outShapeNode: ConcreteValueNode;
            if (spatialStatic) {
                const finalDims = [1, M, ...xDims.slice(2)];
                outShapeNode = builder.createConstant(`c1x1_osh_${op.id}`, int64Vec(finalDims));
            } else {
                const nmVec = builder.createConstant(`c1x1_nm_${op.id}`, int64Vec([1, M]));
                [outShapeNode] = builder.createOp("Concat", [nmVec, spatialVecNode!], { axis: 0 }, [
                    { type: DataType.INT64, shape: [rank] },
                ]);
            }

            const outShapeKnown = spatialStatic ? [1, M, ...xDims.slice(2)] : yDims;
            [result] = builder.createOp("Reshape", [mm, outShapeNode], {}, [
                { type: Y.literalType || dtype, shape: outShapeKnown },
            ]);
        } else {
            // General N >= 1 / dynamic N: [N, C, S_prod] -> [N, S_prod, C] @ [C, M] -> [N, M, S_prod]
            const [WT] = builder.createOp("Transpose", [W2d], { perm: [1, 0] }, [
                { type: dtype, shape: [C, M] },
            ]);

            let x3dShapeNode: ConcreteValueNode;
            let nVecNode: ConcreteValueNode | undefined;
            let spatialVecNode: ConcreteValueNode | undefined;

            if (!xDims.includes(-1)) {
                x3dShapeNode = builder.createConstant(
                    `c1x1_x3d_${op.id}`,
                    int64Vec([xDims[0], C, sProdStatic]),
                );
            } else {
                const [shapeX] = builder.createOp("Shape", [X], {}, [
                    { type: DataType.INT64, shape: [rank] },
                ]);
                const ax0 = builder.createConstant(`c1x1_ax0_${op.id}`, int64Vec([0]));
                const z = builder.createConstant(`c1x1_z_${op.id}`, int64Vec([0]));
                const one = builder.createConstant(`c1x1_one_${op.id}`, int64Vec([1]));
                const two = builder.createConstant(`c1x1_two_${op.id}`, int64Vec([2]));
                const rConst = builder.createConstant(`c1x1_rk_${op.id}`, int64Vec([rank]));

                [nVecNode] = builder.createOp("Slice", [shapeX, z, one, ax0], {}, [
                    { type: DataType.INT64, shape: [1] },
                ]);
                [spatialVecNode] = builder.createOp("Slice", [shapeX, two, rConst, ax0], {}, [
                    { type: DataType.INT64, shape: [rank - 2] },
                ]);
                const [sProd] = builder.createOp("ReduceProd", [spatialVecNode, ax0], {
                    keepdims: 1,
                });
                const cVec = builder.createConstant(`c1x1_cvec_${op.id}`, int64Vec([C]));
                [x3dShapeNode] = builder.createOp("Concat", [nVecNode, cVec, sProd], { axis: 0 }, [
                    { type: DataType.INT64, shape: [3] },
                ]);
            }

            const [X3d] = builder.createOp("Reshape", [X, x3dShapeNode], {}, [
                { type: dtype, shape: [xDims[0], C, sProdStatic] },
            ]);
            const [Xperm] = builder.createOp("Transpose", [X3d], { perm: [0, 2, 1] }, [
                { type: dtype, shape: [xDims[0], sProdStatic, C] },
            ]);
            const [mm] = builder.createOp("MatMul", [Xperm, WT], {}, [
                { type: dtype, shape: [xDims[0], sProdStatic, M] },
            ]);
            const [mmPerm] = builder.createOp("Transpose", [mm], { perm: [0, 2, 1] }, [
                { type: dtype, shape: [xDims[0], M, sProdStatic] },
            ]);

            let outShapeNode: ConcreteValueNode;
            if (!xDims.includes(-1)) {
                const finalDims = [xDims[0], M, ...xDims.slice(2)];
                outShapeNode = builder.createConstant(`c1x1_osh_${op.id}`, int64Vec(finalDims));
            } else {
                const mVec = builder.createConstant(`c1x1_mvec_${op.id}`, int64Vec([M]));
                [outShapeNode] = builder.createOp(
                    "Concat",
                    [nVecNode!, mVec, spatialVecNode!],
                    { axis: 0 },
                    [{ type: DataType.INT64, shape: [rank] }],
                );
            }

            const outShapeKnown = [xDims[0], M, ...xDims.slice(2)];
            [result] = builder.createOp("Reshape", [mmPerm, outShapeNode], {}, [
                { type: Y.literalType || dtype, shape: outShapeKnown },
            ]);

            if (B) {
                const bBroadcastDims = [1, M, ...new Array<number>(rank - 2).fill(1)];
                const bShapeConst = builder.createConstant(
                    `c1x1_bsh_${op.id}`,
                    int64Vec(bBroadcastDims),
                );
                const [bReshaped] = builder.createOp("Reshape", [B, bShapeConst], {}, [
                    { type: dtype, shape: bBroadcastDims },
                ]);
                [result] = builder.createOp("Add", [result, bReshaped], {}, [
                    { type: Y.literalType || dtype, shape: outShapeKnown },
                ]);
            }
        }

        builder.replaceAllUsesWith(Y, result);
        builder.removeNode(op);
    }
}
