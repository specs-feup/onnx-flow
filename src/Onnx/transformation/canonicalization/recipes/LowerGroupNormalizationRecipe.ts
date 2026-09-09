import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, getIntAttr, makeTensorProto, toStaticShape } from "../../../Utils.js";

export class LowerGroupNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerGroupNormalization";
    public readonly targetOp = "GroupNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Reshape",
        "ReduceMean",
        "Sub",
        "Mul",
        "Add",
        "Sqrt",
        "Div",
        "Shape",
        "Gather",
        "Concat",
    ];

    match(op: OperationNode.Class): boolean {
        return op.type === "GroupNormalization";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as (ConcreteValueNode | undefined)[];
        const X = ins[0]!;
        const scale = ins.length > 1 ? ins[1] : undefined;
        const bias = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = Yshape ? (Y.shape as KnownShape) : (X.shape as KnownShape);
        const rank = outShape.length;

        const numGroups = getIntAttr(
            op,
            "num_groups",
            getIntAttr(op, "groups", getIntAttr(op, "group", 1)),
        );
        const eps = getFloatAttr(op, "epsilon", getFloatAttr(op, "eps", 1e-5));

        const xStaticShape = toStaticShape(X.shape);
        const isStatic =
            xStaticShape.length >= 2 &&
            !xStaticShape.includes(-1) &&
            xStaticShape.every((d) => d > 0);

        // 1. Build 3D Shape [N, G, (C/G) * spatial] and original shape reference
        let shape3DNode: ConcreteValueNode;
        let origShapeNode: ConcreteValueNode;
        let shape3DExpected: KnownShape = [-1, numGroups, -1];

        if (isStatic) {
            const N = xStaticShape[0];
            const C = xStaticShape[1];
            const spatialSize = xStaticShape.slice(2).reduce((a, b) => a * b, 1);
            const groupElements = Math.floor(C / numGroups) * spatialSize;
            const shape3D = [N, numGroups, groupElements];

            shape3DExpected = shape3D;
            shape3DNode = builder.createConstant(
                `gn_shape3d_${op.id}`,
                makeTensorProto(DataType.INT64, [3], shape3D),
            );
            origShapeNode = builder.createConstant(
                `gn_shape_orig_${op.id}`,
                makeTensorProto(DataType.INT64, [xStaticShape.length], xStaticShape),
            );
        } else {
            const [shapeX] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            origShapeNode = shapeX;

            const zeroIdx = builder.createConstant(
                `gn_zero_idx_${op.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            const [nDim] = builder.createOp("Gather", [shapeX, zeroIdx], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);
            const gAndMinusOne = builder.createConstant(
                `gn_g_m1_${op.id}`,
                makeTensorProto(DataType.INT64, [2], [numGroups, -1]),
            );
            [shape3DNode] = builder.createOp("Concat", [nDim, gAndMinusOne], { axis: 0 }, [
                { type: DataType.INT64, shape: [3] },
            ]);
        }

        // 2. Reshape input to 3D: [N, G, group_elements]
        const [x3D] = builder.createOp("Reshape", [X, shape3DNode], {}, [
            { type: dtype, shape: shape3DExpected },
        ]);

        // 3. Compute Mean across group elements (axis 2)
        const axes2Const = builder.createConstant(
            `gn_axes2_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [2]),
        );
        const meanShape: KnownShape = isStatic
            ? [shape3DExpected[0], numGroups, 1]
            : [-1, numGroups, 1];
        const meanExpected = [{ type: dtype, shape: meanShape }];

        const [mean] = builder.createOp(
            "ReduceMean",
            [x3D, axes2Const],
            { keepdims: 1 },
            meanExpected,
        );

        // 4. Compute Variance: Mean((X - Mean)^2)
        const x3DExpected = [{ type: dtype, shape: shape3DExpected }];
        const [diff] = builder.createOp("Sub", [x3D, mean], {}, x3DExpected);
        const [diffSq] = builder.createOp("Mul", [diff, diff], {}, x3DExpected);
        const [variance] = builder.createOp(
            "ReduceMean",
            [diffSq, axes2Const],
            { keepdims: 1 },
            meanExpected,
        );

        // 5. Standard Deviation: Sqrt(Variance + epsilon)
        const epsConst = builder.createConstant(
            `gn_eps_${op.id}`,
            makeTensorProto(dtype, [], [eps]),
        );
        const [varEps] = builder.createOp("Add", [variance, epsConst], {}, meanExpected);
        const [stdDev] = builder.createOp("Sqrt", [varEps], {}, meanExpected);

        // 6. Normalize in 3D: (X - Mean) / StdDev
        const [xNorm3D] = builder.createOp("Div", [diff, stdDev], {}, x3DExpected);

        // 7. Reshape back to original N-D shape: [N, C, H, W, ...]
        const [xNorm] = builder.createOp("Reshape", [xNorm3D, origShapeNode], {}, [
            { type: dtype, shape: outShape },
        ]);

        let curOut = xNorm;

        // 8. Scale and Bias: Reshape 1D [C] to [1, C, 1, ..., 1] for exact channel alignment
        if (scale || bias) {
            const C = isStatic ? (xStaticShape[1] as number) : -1;
            const paramShape: KnownShape = new Array(rank).fill(1);
            paramShape[1] = C;

            if (isStatic) {
                const paramShapeConst = builder.createConstant(
                    `gn_param_shape_${op.id}`,
                    makeTensorProto(DataType.INT64, [rank], paramShape as number[]),
                );

                if (scale) {
                    const [scaleReshaped] = builder.createOp(
                        "Reshape",
                        [scale, paramShapeConst],
                        {},
                        [{ type: dtype, shape: paramShape }],
                    );
                    [curOut] = builder.createOp("Mul", [curOut, scaleReshaped], {}, [
                        { type: dtype, shape: outShape },
                    ]);
                }

                if (bias) {
                    const [biasReshaped] = builder.createOp(
                        "Reshape",
                        [bias, paramShapeConst],
                        {},
                        [{ type: dtype, shape: paramShape }],
                    );
                    [curOut] = builder.createOp("Add", [curOut, biasReshaped], {}, [
                        { type: dtype, shape: outShape },
                    ]);
                }
            } else {
                // Dynamic fallback for non-static shapes
                const oneI64 = builder.createConstant(
                    `gn_one_dim_${op.id}`,
                    makeTensorProto(DataType.INT64, [1], [1]),
                );
                const oneIdx = builder.createConstant(
                    `gn_one_idx_${op.id}`,
                    makeTensorProto(DataType.INT64, [1], [1]),
                );
                const [cDim] = builder.createOp("Gather", [origShapeNode, oneIdx], { axis: 0 }, [
                    { type: DataType.INT64, shape: [1] },
                ]);

                const parts: ConcreteValueNode[] = [oneI64, cDim];
                for (let i = 2; i < rank; i++) {
                    parts.push(oneI64);
                }
                const [dynParamShape] = builder.createOp("Concat", parts, { axis: 0 }, [
                    { type: DataType.INT64, shape: [rank] },
                ]);

                if (scale) {
                    const [scaleReshaped] = builder.createOp(
                        "Reshape",
                        [scale, dynParamShape],
                        {},
                        [{ type: dtype, shape: paramShape }],
                    );
                    [curOut] = builder.createOp("Mul", [curOut, scaleReshaped], {}, [
                        { type: dtype, shape: outShape },
                    ]);
                }

                if (bias) {
                    const [biasReshaped] = builder.createOp("Reshape", [bias, dynParamShape], {}, [
                        { type: dtype, shape: paramShape },
                    ]);
                    [curOut] = builder.createOp("Add", [curOut, biasReshaped], {}, [
                        { type: dtype, shape: outShape },
                    ]);
                }
            }
        }

        // 9. Replace original GroupNormalization uses and remove the node
        builder.replaceAllUsesWith(Y, curOut);
        op.remove();
    }
}
