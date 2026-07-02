import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, makeTensorProto } from "../../../Utils.js";

export class LowerBatchNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerBatchNormalization";
    public readonly targetOp = "BatchNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Unsqueeze", "Add", "Sqrt", "Sub", "Div", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "BatchNormalization";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const scale = ins[1];
        const B = ins[2];
        const mean = ins[3];
        const vari = ins[4];
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (X.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const rank = X.shape.length;
        const unsqAxes: number[] = [];

        // We need to unsqueeze all dimensions except the channel dimension (1) for the parameters
        for (let i = 0; i < rank; i++) {
            if (i !== 1) unsqAxes.push(i);
        }

        const unsqAxesConst = builder.createConstant(
            `bn_unsq_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [unsqAxes.length], unsqAxes),
        );

        const eps = getFloatAttr(op, "epsilon", 1e-5);

        const epsConst = builder.createConstant(
            `bn_eps_${op.id}`,
            makeTensorProto(dtype, [], [eps]),
        );

        // Unsqueeze parameters
        const scaleU = builder.createOp("Unsqueeze", [scale, unsqAxesConst], {}, output)[0];
        const BU = builder.createOp("Unsqueeze", [B, unsqAxesConst], {}, output)[0];
        const meanU = builder.createOp("Unsqueeze", [mean, unsqAxesConst], {}, output)[0];
        const varU = builder.createOp("Unsqueeze", [vari, unsqAxesConst], {}, output)[0];

        // VarEps = var + epsilon
        const varEps = builder.createOp("Add", [varU, epsConst], {}, output)[0];

        // StdDev = sqrt(var + epsilon)
        const stdDev = builder.createOp("Sqrt", [varEps], {}, output)[0];

        // Diff = X - mean
        const diff = builder.createOp("Sub", [X, meanU], {}, output)[0];

        // Norm = (X - mean) / stdDev
        const norm = builder.createOp("Div", [diff, stdDev], {}, output)[0];

        // Scaled = Norm * scale
        const scaled = builder.createOp("Mul", [norm, scaleU], {}, output)[0];

        // Y = Scaled + B
        const finalOut = builder.createOp("Add", [scaled, BU], {}, output)[0];

        builder.replaceAllUsesWith(Y, finalOut);
        op.remove();
    }
}