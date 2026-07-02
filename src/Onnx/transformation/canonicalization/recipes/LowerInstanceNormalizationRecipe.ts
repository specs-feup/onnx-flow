import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerInstanceNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerInstanceNormalization";
    public readonly targetOp = "InstanceNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Unsqueeze", "ReduceMean", "Sub", "Mul", "Add", "Sqrt", "Div"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "InstanceNormalization";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const input = ins[0];
        const scale = ins[1];
        const B = ins[2];
        const Y = op.getOutputs()[0];

        const dtype =
            (input.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (input.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const rank = input.shape.length;
        const epsilon = getFloatAttr(op, "epsilon", 1e-5);

        const reduceAxes: number[] = [];
        for (let i = 2; i < rank; i++) {
            reduceAxes.push(i);
        }

        const unsqAxes: number[] = [0];
        for (let i = 2; i < rank; i++) {
            unsqAxes.push(i);
        }

        const reduceAxesConst = builder.createConstant(
            `instancenorm_reduce_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [reduceAxes.length], reduceAxes),
        );

        const unsqAxesConst = builder.createConstant(
            `instancenorm_unsq_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [unsqAxes.length], unsqAxes),
        );

        const epsConst = builder.createConstant(
            `instancenorm_eps_${op.id}`,
            makeTensorProto(dtype, [], [epsilon]),
        );

        const scaleU = builder.createOp("Unsqueeze", [scale, unsqAxesConst], {}, output)[0];
        const BU = builder.createOp("Unsqueeze", [B, unsqAxesConst], {}, output)[0];

        const mean = builder.createOp(
            "ReduceMean",
            [input, reduceAxesConst],
            { keepdims: 1 },
            output,
        )[0];

        const diff = builder.createOp("Sub", [input, mean], {}, output)[0];
        const diffSq = builder.createOp("Mul", [diff, diff], {}, output)[0];

        const vari = builder.createOp(
            "ReduceMean",
            [diffSq, reduceAxesConst],
            { keepdims: 1 },
            output,
        )[0];

        const varEps = builder.createOp("Add", [vari, epsConst], {}, output)[0];
        const stdDev = builder.createOp("Sqrt", [varEps], {}, output)[0];
        const norm = builder.createOp("Div", [diff, stdDev], {}, output)[0];
        const scaled = builder.createOp("Mul", [norm, scaleU], {}, output)[0];
        const finalOut = builder.createOp("Add", [scaled, BU], {}, output)[0];

        builder.replaceAllUsesWith(Y, finalOut);
        op.remove();
    }
}