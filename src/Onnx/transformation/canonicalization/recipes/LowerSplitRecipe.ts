import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    normalizeAxis,
    readConstIntegerVectorFromTensorNode,
    toStaticShape,
} from "../../../Utils.js";

export class LowerSplitRecipe implements DecompositionRecipe {
    public readonly name = "LowerSplit";
    public readonly targetOp = "Split";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic-batch" as const;
    public readonly tags = ["slice-based"];
    public readonly producedOps = ["Slice"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Split") return false;
        const ins = op.getInputs();
        if (!ins || ins.length === 0 || !Array.isArray(ins[0].shape) || ins[0].shape.length === 0) {
            return false;
        }
        if (ins.length >= 2 && ins[1].id !== "") {
            return readConstIntegerVectorFromTensorNode(ins[1]) !== undefined;
        }
        const shape = toStaticShape(ins[0].shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), shape.length);
        return shape[axis] > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const outputs = op.getOutputs();
        const dtype = X.literalType as DataType;

        const inShape = toStaticShape(X.shape);
        const axis = normalizeAxis(getIntAttr(op, "axis", 0), inShape.length);

        let splitSizes = ins.length >= 2 ? readConstIntegerVectorFromTensorNode(ins[1]) : undefined;
        if (!splitSizes) {
            const numOutputs = Math.max(
                getIntAttr(op, "num_outputs", outputs.length),
                outputs.length,
            );
            const dimSize = inShape[axis];
            const chunkSize = Math.ceil(dimSize / numOutputs);
            splitSizes = [];
            let rem = dimSize;
            for (let i = 0; i < numOutputs; i++) {
                const sz = Math.min(chunkSize, Math.max(0, rem));
                splitSizes.push(sz);
                rem -= sz;
            }
        }

        const axesConst = builder.createConstant(`split_axes_${op.id}`, int64Vec([axis]));
        const stepsConst = builder.createConstant(`split_steps_${op.id}`, int64Vec([1]));

        let offset = 0;
        for (let i = 0; i < outputs.length; i++) {
            const size = splitSizes[i] ?? 0;
            const startsConst = builder.createConstant(
                `split_start_${op.id}_${i}`,
                int64Vec([offset]),
            );
            const endsConst = builder.createConstant(
                `split_end_${op.id}_${i}`,
                int64Vec([offset + size]),
            );

            const sliceShape = [...inShape];
            sliceShape[axis] = size;

            const [sliced] = builder.createOp(
                "Slice",
                [X, startsConst, endsConst, axesConst, stepsConst],
                {},
                [{ type: dtype, shape: sliceShape as KnownShape }],
            );

            builder.replaceAllUsesWith(outputs[i], sliced);
            offset += size;
        }

        builder.removeNode(op);
    }
}
