import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { getFloatAttr, getIntAttr, makeTensorProto } from "../../../utils/index.js";

export class LowerLayerNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerLayerNormalization";
    public readonly targetOp = "LayerNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["ReduceMean", "Sub", "Mul", "Add", "Sqrt", "Div"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LayerNormalization") return false;
        const ins = op.getInputs();
        return !!ins && ins.length >= 2 && ins[0].shape.length >= 1;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Scale = ins[1];
        const B = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape = Y.shape as KnownShape;

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
        if (B !== undefined) {
            finalOut = builder.createOp("Add", [scaled, B], {}, output)[0];
        }

        const outputs = op.getOutputs();
        builder.replaceAllUsesWith(outputs[0], finalOut);

        if (outputs.length > 1) {
            builder.replaceAllUsesWith(outputs[1], mean);
        }
        if (outputs.length > 2) {
            const oneConst = builder.createConstant(
                `layernorm_one_${op.id}`,
                makeTensorProto(dtype, [], [1]),
            );
            const [invStdDev] = builder.createOp("Div", [oneConst, stdDev], {}, [
                { type: dtype, shape: stdDev.shape as KnownShape },
            ]);
            builder.replaceAllUsesWith(outputs[2], invStdDev);
        }

        op.remove();
    }
}
