import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, getIntAttr, makeTensorProto } from "../../../Utils.js";

export class LowerGroupNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerGroupNormalization";
    public readonly targetOp = "GroupNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Unsqueeze", "Reshape", "ReduceMean", "Sub", "Mul", "Add", "Sqrt", "Div"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "GroupNormalization";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const scale = ins[1];
        const bias = ins[2];
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const xShape = X.shape as KnownShape;
        if (xShape.length !== 4) {
            throw new Error("LowerGroupNormalizationRecipe currently only supports rank-4 tensors.");
        }

        const N = Number(xShape[0]);
        const C = Number(xShape[1]);
        const H = Number(xShape[2]);
        const W = Number(xShape[3]);

        const epsilon = getFloatAttr(op, "epsilon", 1e-5);
        const numGroups = getIntAttr(op, "num_groups", 1);

        if (!Number.isFinite(C) || C % numGroups !== 0) {
            throw new Error("Channel dimension must be known and divisible by num_groups.");
        }

        const channelsPerGroup = C / numGroups;
        const groupSize = channelsPerGroup * H * W;

        const output4D = [{ type: dtype, shape: [N, C, H, W] as KnownShape }];
        const grouped3D = [{ type: dtype, shape: [N, numGroups, groupSize] as KnownShape }];
        const reduced3D = [{ type: dtype, shape: [N, numGroups, 1] as KnownShape }];
        const param4D = [{ type: dtype, shape: [1, C, 1, 1] as KnownShape }];

        const epsConst = builder.createConstant(
            `groupnorm_eps_${op.id}`,
            makeTensorProto(dtype, [], [epsilon]),
        );

        const unsqAxesConst = builder.createConstant(
            `groupnorm_unsq_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [3], [0, 2, 3]),
        );

        const reduceAxesConst = builder.createConstant(
            `groupnorm_reduce_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [2]),
        );

        const shapeGroupedConst = builder.createConstant(
            `groupnorm_shape_grouped_${op.id}`,
            makeTensorProto(DataType.INT64, [3], [N, numGroups, groupSize]),
        );

        const shapeOriginalConst = builder.createConstant(
            `groupnorm_shape_original_${op.id}`,
            makeTensorProto(DataType.INT64, [4], [N, C, H, W]),
        );

        const scaleU = builder.createOp("Unsqueeze", [scale, unsqAxesConst], {}, param4D)[0];
        const biasU = builder.createOp("Unsqueeze", [bias, unsqAxesConst], {}, param4D)[0];

        const XGrouped = builder.createOp("Reshape", [X, shapeGroupedConst], {}, grouped3D)[0];

        const mean = builder.createOp(
            "ReduceMean",
            [XGrouped, reduceAxesConst],
            { keepdims: 1 },
            reduced3D,
        )[0];

        const diff = builder.createOp("Sub", [XGrouped, mean], {}, grouped3D)[0];
        const diffSq = builder.createOp("Mul", [diff, diff], {}, grouped3D)[0];

        const vari = builder.createOp(
            "ReduceMean",
            [diffSq, reduceAxesConst],
            { keepdims: 1 },
            reduced3D,
        )[0];

        const varEps = builder.createOp("Add", [vari, epsConst], {}, reduced3D)[0];
        const stdDev = builder.createOp("Sqrt", [varEps], {}, reduced3D)[0];
        const normGrouped = builder.createOp("Div", [diff, stdDev], {}, grouped3D)[0];

        const norm = builder.createOp("Reshape", [normGrouped, shapeOriginalConst], {}, output4D)[0];
        const scaled = builder.createOp("Mul", [norm, scaleU], {}, output4D)[0];
        const finalOut = builder.createOp("Add", [scaled, biasU], {}, output4D)[0];

        builder.replaceAllUsesWith(Y, finalOut);
        op.remove();
    }
}