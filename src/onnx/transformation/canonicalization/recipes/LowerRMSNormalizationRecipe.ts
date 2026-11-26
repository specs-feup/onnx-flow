import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, getIntAttr, makeTensorProto } from "../../../utils/index.js";

/**
 * Canonicalizes RMSNormalization (Opset 23):
 *   Y = Cast(X_calc / Sqrt(ReduceMean(X_calc^2, axes=[axis..rank-1]) + eps), X.dtype) * scale
 */
export class LowerRMSNormalizationRecipe implements DecompositionRecipe {
    public readonly name = "LowerRMSNormalization";
    public readonly targetOp = "RMSNormalization";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Cast", "Mul", "ReduceMean", "Add", "Sqrt", "Div"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "RMSNormalization") return false;
        const ins = op.getInputs();
        return !!ins && ins.length >= 2 && ins[0].shape.length >= 1;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Scale = ins[1];
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const stashType = getIntAttr(op, "stash_type", DataType.FLOAT) as DataType;
        const calcDtype =
            (dtype === DataType.FLOAT16 || dtype === DataType.BFLOAT16) &&
            stashType !== DataType.UNDEFINED
                ? stashType
                : dtype;

        const rawOutShape = Array.isArray(Y.shape) && Y.shape.length > 0 ? Y.shape : X.shape;
        const outShape: KnownShape = rawOutShape.map((d) => d ?? -1);
        const rank = X.shape.length;
        const axisAttr = getIntAttr(op, "axis", -1);
        const axis = axisAttr < 0 ? rank + axisAttr : axisAttr;
        const epsilon = getFloatAttr(op, "epsilon", 1e-5);

        const reduceAxes: number[] = [];
        for (let i = axis; i < rank; i++) {
            reduceAxes.push(i);
        }

        const reducedShape: KnownShape = outShape.map((d, i) => (i >= axis ? 1 : d));
        const expectedCalc = [{ type: calcDtype, shape: outShape }];
        const expectedReduced = [{ type: calcDtype, shape: reducedShape }];

        const xCalc: ConcreteValueNode =
            calcDtype === dtype
                ? X
                : builder.createOp("Cast", [X], { to: calcDtype }, expectedCalc)[0];

        const axesConst = builder.createConstant(
            `rmsnorm_axes_${op.id}`,
            makeTensorProto(DataType.INT64, [reduceAxes.length], reduceAxes),
        );
        const epsConst = builder.createConstant(
            `rmsnorm_eps_${op.id}`,
            makeTensorProto(calcDtype, [], [epsilon]),
        );

        const [xSq] = builder.createOp("Mul", [xCalc, xCalc], {}, expectedCalc);
        const [meanSq] = builder.createOp(
            "ReduceMean",
            [xSq, axesConst],
            { keepdims: 1 },
            expectedReduced,
        );
        const [meanEps] = builder.createOp("Add", [meanSq, epsConst], {}, expectedReduced);
        const [rms] = builder.createOp("Sqrt", [meanEps], {}, expectedReduced);
        const [normCalc] = builder.createOp("Div", [xCalc, rms], {}, expectedCalc);

        const norm: ConcreteValueNode =
            calcDtype === dtype
                ? normCalc
                : builder.createOp("Cast", [normCalc], { to: dtype }, [
                      { type: dtype, shape: outShape },
                  ])[0];

        const [finalOut] = builder.createOp("Mul", [norm, Scale], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, finalOut);
        op.remove();
    }
}
