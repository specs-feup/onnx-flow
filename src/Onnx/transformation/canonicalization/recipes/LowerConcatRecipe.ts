import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, isNumeric, makeTensorProto, toStaticShape } from "../../../Utils.js";

export class LowerConcatRecipe implements DecompositionRecipe {
    public readonly name = "LowerConcat";
    public readonly targetOp = "Concat";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "Add",
        "Squeeze",
        "Unsqueeze",
        "ScatterElements",
        "Expand",
        "Range",
        "Identity",
    ];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "Concat") return false;
        const inputs = op.getInputs() as ConcreteValueNode[];
        if (inputs.length < 2) return false;

        const rank = inputs[0].shape.length;
        if (!inputs.every((t) => t.shape.length === rank)) return false;

        const dtype = inputs[0].literalType as DataType;
        if (!inputs.every((t) => t.literalType === dtype)) return false;

        const axisAttr = getIntAttr(op, "axis", 0);
        const axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
        if (axis < 0 || axis >= rank) return false;

        return isNumeric(dtype);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];
        const rank = inputs[0].shape.length;
        const dtype = inputs[0].literalType as DataType;

        let axis = getIntAttr(op, "axis", 0);
        if (axis < 0) axis += rank;

        const Yshape = toStaticShape(Y.shape);

        // 1. Extract shapes and sum sizes along the target axis
        const shape0 = builder.createOp("Shape", [inputs[0]], {}, [
            { type: DataType.INT64, shape: [rank] },
        ])[0];

        const sizeScalars: ConcreteValueNode[] = [];
        const axisIdxConst = builder.createConstant(
            `Concat_axis_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [axis]),
        );

        for (let i = 0; i < inputs.length; i++) {
            const shapeI = builder.createOp("Shape", [inputs[i]], {}, [
                { type: DataType.INT64, shape: [rank] },
            ])[0];
            const size1D = builder.createOp("Gather", [shapeI, axisIdxConst], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ])[0];
            const zeroConst = builder.createConstant(
                `Concat_sq_axes_${i}_${op.id}`,
                makeTensorProto(DataType.INT64, [1], [0]),
            );
            const sizeSc = builder.createOp("Squeeze", [size1D, zeroConst], {}, [
                { type: DataType.INT64, shape: [] },
            ])[0];
            sizeScalars.push(sizeSc);
        }

        let sumAxis: ConcreteValueNode = builder.createConstant(
            `Concat_sum_init_${op.id}`,
            makeTensorProto(DataType.INT64, [], [0]),
        );
        for (const sizeSc of sizeScalars) {
            sumAxis = builder.createOp("Add", [sumAxis, sizeSc])[0];
        }

        const unsqZeroConst = builder.createConstant(
            `Concat_unsq_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const sumAxis1D = builder.createOp("Unsqueeze", [sumAxis, unsqZeroConst], {}, [
            { type: DataType.INT64, shape: [1] },
        ])[0];
        const outShape1D = builder.createOp(
            "ScatterElements",
            [shape0, axisIdxConst, sumAxis1D],
            {
                axis: 0,
            },
            [{ type: DataType.INT64, shape: [rank] }],
        )[0];

        // 2. Initialize Y with Expand(0, out_shape)
        const zeroVal = builder.createConstant(
            `Concat_zero_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        let curY = builder.createOp("Expand", [zeroVal, outShape1D], {}, [
            { type: dtype, shape: Yshape },
        ])[0];

        // 3. Incrementally Scatter inputs into the constructed Y
        let offsetSc: ConcreteValueNode = builder.createConstant(
            `Concat_off_init_${op.id}`,
            makeTensorProto(DataType.INT64, [], [0]),
        );
        const oneSc = builder.createConstant(
            `Concat_one_${op.id}`,
            makeTensorProto(DataType.INT64, [], [1]),
        );

        for (let i = 0; i < inputs.length; i++) {
            const Xi = inputs[i];
            const sizeSc = sizeScalars[i];

            const endSc: ConcreteValueNode = builder.createOp("Add", [offsetSc, sizeSc])[0];

            const axisDim = Array.isArray(Xi.shape) ? Xi.shape[axis] : undefined;
            const range1D = builder.createOp("Range", [offsetSc, endSc, oneSc], {}, [
                { type: DataType.INT64, shape: [axisDim] as KnownShape },
            ])[0];

            const axesToUnsq = Array.from({ length: rank }, (_, idx) => idx).filter(
                (idx) => idx !== axis,
            );
            let idxRanked = range1D;

            if (axesToUnsq.length > 0) {
                const axesConst = builder.createConstant(
                    `Concat_unsq_idx_${i}_${op.id}`,
                    makeTensorProto(DataType.INT64, [axesToUnsq.length], axesToUnsq),
                );

                const idxShape = Array.isArray(Xi.shape)
                    ? [...Xi.shape]
                    : new Array(rank).fill(undefined);
                for (const d of axesToUnsq) {
                    idxShape[d] = 1;
                }

                idxRanked = builder.createOp("Unsqueeze", [range1D, axesConst], {}, [
                    { type: DataType.INT64, shape: idxShape as KnownShape },
                ])[0];
            }

            const shapeI = builder.createOp("Shape", [Xi])[0];
            const idxFull = builder.createOp("Expand", [idxRanked, shapeI], {}, [
                { type: DataType.INT64, shape: Xi.shape as KnownShape },
            ])[0];

            curY = builder.createOp("ScatterElements", [curY, idxFull, Xi], { axis })[0];

            offsetSc = endSc;
        }

        const finalId = builder.createOp("Identity", [curY], {}, [
            { type: dtype, shape: Yshape },
        ])[0];
        builder.replaceAllUsesWith(Y, finalId);
        op.remove();
    }
}
