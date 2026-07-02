import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, getIntAttr, makeTensorProto } from "../../../Utils.js";

export class LowerLayerNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerLayerNormalization";
    public readonly targetOp = "LayerNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["ReduceMean", "Sub", "Mul", "Add", "Sqrt", "Div"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "LayerNormalization";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Scale = ins[1];
        const B = ins[2];
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
        const axisAttr = getIntAttr(op, "axis", -1);
        const axis = axisAttr < 0 ? rank + axisAttr : axisAttr;
        const epsilon = getFloatAttr(op, "epsilon", 1e-5);

        const reduceAxes: number[] = [];
        for (let i = axis; i < rank; i++) {
            reduceAxes.push(i);
        }

        const reduceAxesConst = builder.createConstant(
            `layernorm_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [reduceAxes.length], reduceAxes),
        );

        const epsConst = builder.createConstant(
            `layernorm_eps_${op.id}`,
            makeTensorProto(dtype, [], [epsilon]),
        );

        const mean = builder.createOp(
            "ReduceMean",
            [X, reduceAxesConst],
            { keepdims: 1 },
            output,
        )[0];

        const diff = builder.createOp("Sub", [X, mean], {}, output)[0];
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
        const scaled = builder.createOp("Mul", [norm, Scale], {}, output)[0];

        let finalOut: ConcreteValueNode = scaled;
        if (B) {
            finalOut = builder.createOp("Add", [scaled, B], {}, output)[0];
        }

        builder.replaceAllUsesWith(Y, finalOut);
        op.remove();
    }
}