import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, normalizeAxis, scalarInt64, toStaticShape } from "../../../Utils.js";

export class LowerSplitDynamicRecipe implements DecompositionRecipe {
    public readonly name = "LowerSplitDynamic";
    public readonly targetOp = "Split";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["dynamic-split", "slice-based"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Shape",
        "Gather",
        "CumSum",
        "Cast",
        "Add",
        "Mul",
        "Div",
        "Min",
        "Slice",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Split") return false;
        const ins = op.getInputs();
        if (!ins || ins.length === 0 || !Array.isArray(ins[0].shape) || ins[0].shape.length === 0) {
            return false;
        }
        return op.getOutputs().length > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const outputs = op.getOutputs();
        const dtype = X.literalType as DataType;

        const inShape = toStaticShape(X.shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), inShape.length);
        const axesConst = builder.createConstant(`dsplit_axes_${op.id}`, int64Vec([axis]));
        const stepsConst = builder.createConstant(`dsplit_steps_${op.id}`, int64Vec([1]));

        const hasDynamicSplit = ins.length >= 2 && ins[1].id !== "";

        if (hasDynamicSplit) {
            const splitIn = ins[1];
            const split64: ConcreteValueNode =
                splitIn.literalType === DataType.INT64
                    ? splitIn
                    : builder.createOp("Cast", [splitIn], { to: DataType.INT64 })[0];

            const zeroAx = builder.createConstant(`dsplit_cax_${op.id}`, scalarInt64(0));
            const [endsAll] = builder.createOp(
                "CumSum",
                [split64, zeroAx],
                { exclusive: 0, reverse: 0 },
                [{ type: DataType.INT64, shape: [outputs.length] }],
            );
            const [startsAll] = builder.createOp(
                "CumSum",
                [split64, zeroAx],
                { exclusive: 1, reverse: 0 },
                [{ type: DataType.INT64, shape: [outputs.length] }],
            );

            for (let i = 0; i < outputs.length; i++) {
                const idx1D = builder.createConstant(`dsplit_idx_${op.id}_${i}`, int64Vec([i]));
                const [s_i] = builder.createOp("Gather", [startsAll, idx1D], { axis: 0 }, [
                    { type: DataType.INT64, shape: [1] },
                ]);
                const [e_i] = builder.createOp("Gather", [endsAll, idx1D], { axis: 0 }, [
                    { type: DataType.INT64, shape: [1] },
                ]);

                const outShape = [...inShape];
                outShape[axis] = -1;
                const [sliced] = builder.createOp(
                    "Slice",
                    [X, s_i, e_i, axesConst, stepsConst],
                    {},
                    [{ type: dtype, shape: outShape as KnownShape }],
                );
                builder.replaceAllUsesWith(outputs[i], sliced);
            }
        } else {
            const numOutputs = Math.max(
                getIntAttr(op, "num_outputs", outputs.length),
                outputs.length,
            );
            const [shapeX] = builder.createOp("Shape", [X], {}, [
                { type: DataType.INT64, shape: [inShape.length] },
            ]);
            const [dimSize] = builder.createOp("Gather", [shapeX, axesConst], { axis: 0 }, [
                { type: DataType.INT64, shape: [1] },
            ]);

            // ceil(dimSize / numOutputs) per ONNX Opset 18 Split spec
            const numOutC = builder.createConstant(`dsplit_n_${op.id}`, int64Vec([numOutputs]));
            const numOutM1 = builder.createConstant(
                `dsplit_nm1_${op.id}`,
                int64Vec([numOutputs - 1]),
            );
            const [chunkSize] = builder.createOp(
                "Div",
                [builder.createOp("Add", [dimSize, numOutM1])[0], numOutC],
                {},
                [{ type: DataType.INT64, shape: [1] }],
            );

            for (let i = 0; i < outputs.length; i++) {
                const iC = builder.createConstant(`dsplit_i_${op.id}_${i}`, int64Vec([i]));
                const i1C = builder.createConstant(`dsplit_i1_${op.id}_${i}`, int64Vec([i + 1]));
                const [rawS] = builder.createOp("Mul", [chunkSize, iC]);
                const [rawE] = builder.createOp("Mul", [chunkSize, i1C]);
                const [s_i] = builder.createOp("Min", [rawS, dimSize], {}, [
                    { type: DataType.INT64, shape: [1] },
                ]);
                const [e_i] = builder.createOp("Min", [rawE, dimSize], {}, [
                    { type: DataType.INT64, shape: [1] },
                ]);

                const outShape = [...inShape];
                outShape[axis] = -1;
                const [sliced] = builder.createOp(
                    "Slice",
                    [X, s_i, e_i, axesConst, stepsConst],
                    {},
                    [{ type: dtype, shape: outShape as KnownShape }],
                );
                builder.replaceAllUsesWith(outputs[i], sliced);
            }
        }

        builder.removeNode(op);
    }
}
