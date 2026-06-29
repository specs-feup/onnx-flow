import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, isNumeric, makeTensorProto, toStaticShape } from "../../../Utils.js";
import { TransformationOpportunity } from "../../TransformationOpportunity.js";

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
        "Range",
        "Identity",
        "ConstantOfShape",
        "Cast",
    ];

    match(op: OperationNode.Class): TransformationOpportunity | null {
        if (op.type !== "Concat") return null;
        const inputs = op.getInputs() as ConcreteValueNode[];
        if (inputs.length < 2) return null;

        const rank = inputs[0].shape.length;
        if (!inputs.every((t) => t.shape.length === rank)) return null;

        const dtype = inputs[0].literalType as DataType;
        if (!inputs.every((t) => t.literalType === dtype)) return null;

        const axisAttr = getIntAttr(op, "axis", 0);
        const axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
        if (axis < 0 || axis >= rank) return null;

        return isNumeric(dtype) ? new TransformationOpportunity(
            this.name,
            op.id,
            "Lower Concat to ScatterElements",
            (builder: GraphBuilder) => this.apply(op, builder),
        ) : null;
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

        // 2. Initialize Y safely using ConstantOfShape to bypass Expand canonicalization bugs
        let curY = builder.createOp("ConstantOfShape", [outShape1D], {}, [
            { type: DataType.FLOAT, shape: Yshape },
        ])[0];

        // Explicitly cast the default float32 zeros to the required data type
        if (dtype !== DataType.FLOAT) {
            curY = builder.createOp("Cast", [curY], { to: dtype }, [
                { type: dtype, shape: Yshape },
            ])[0];
        }

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

            const axisDim =
                Array.isArray(Xi.shape) && Xi.shape[axis] !== undefined ? Xi.shape[axis] : -1;
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

                const idxShape = Array.isArray(Xi.shape) ? [...Xi.shape] : new Array(rank).fill(-1);
                for (const d of axesToUnsq) {
                    idxShape[d] = 1;
                }

                idxRanked = builder.createOp("Unsqueeze", [range1D, axesConst], {}, [
                    { type: DataType.INT64, shape: idxShape as KnownShape },
                ])[0];
            }

            const shapeI = builder.createOp("Shape", [Xi])[0];

            // Generate INT64 zeros of exact shapeI, then Add to idxRanked to force broadcast.
            // This safely bypasses the Expand canonicalization type mismatch.
            let idxFullZeros = builder.createOp("ConstantOfShape", [shapeI], {}, [
                { type: DataType.FLOAT, shape: Xi.shape as KnownShape },
            ])[0];

            idxFullZeros = builder.createOp("Cast", [idxFullZeros], { to: DataType.INT64 }, [
                { type: DataType.INT64, shape: Xi.shape as KnownShape },
            ])[0];

            const idxFull = builder.createOp("Add", [idxRanked, idxFullZeros], {}, [
                { type: DataType.INT64, shape: Xi.shape as KnownShape },
            ])[0];

            curY = builder.createOp("ScatterElements", [curY, idxFull, Xi], { axis }, [
                { type: dtype, shape: Yshape },
            ])[0];

            offsetSc = endSc;
        }

        const finalId = builder.createOp("Identity", [curY], {}, [
            { type: dtype, shape: Yshape },
        ])[0];
        builder.replaceAllUsesWith(Y, finalId);
        op.remove();
    }
}
