import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, makeTensorProto, normalizeAxis } from "../../../utils/index.js";

export class LowerLpNormalizationGeneralRecipe implements DecompositionRecipe {
    public readonly name = "LowerLpNormalizationGeneral";
    public readonly targetOp = "LpNormalization";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["normalization", "general-p"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Abs", "Pow", "ReduceSum", "Equal", "Where", "Div"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LpNormalization") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length === 0) return false;
        const p = getIntAttr(op, "p", 2);
        return p > 0 && p !== 1 && p !== 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];
        const dtype = (X.literalType || DataType.FLOAT) as DataType;
        const rank = X.shape.length;
        const axis = normalizeAxis(getIntAttr(op, "axis", -1), rank);
        const p = getIntAttr(op, "p", 2);

        const outShape = Y.shape as KnownShape;
        const reducedShape = [...(X.shape as KnownShape)];
        reducedShape[axis] = 1;

        const pConst = builder.createConstant(`lpnorm_p_${op.id}`, makeTensorProto(dtype, [], [p]));
        const invPConst = builder.createConstant(
            `lpnorm_invp_${op.id}`,
            makeTensorProto(dtype, [], [1 / p]),
        );
        const zeroConst = builder.createConstant(
            `lpnorm_z_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const oneConst = builder.createConstant(
            `lpnorm_one_${op.id}`,
            makeTensorProto(dtype, [], [1]),
        );
        const axesConst = builder.createConstant(`lpnorm_ax_${op.id}`, int64Vec([axis]));

        const [absX] = builder.createOp("Abs", [X], {}, [{ type: dtype, shape: outShape }]);
        const [powX] = builder.createOp("Pow", [absX, pConst], {}, [
            { type: dtype, shape: outShape },
        ]);
        const [sumPow] = builder.createOp("ReduceSum", [powX, axesConst], { keepdims: 1 }, [
            { type: dtype, shape: reducedShape },
        ]);
        const [norm] = builder.createOp("Pow", [sumPow, invPConst], {}, [
            { type: dtype, shape: reducedShape },
        ]);
        const [isZero] = builder.createOp("Equal", [norm, zeroConst], {}, [
            { type: DataType.BOOL, shape: reducedShape },
        ]);
        const [safeNorm] = builder.createOp("Where", [isZero, oneConst, norm], {}, [
            { type: dtype, shape: reducedShape },
        ]);
        const [out] = builder.createOp("Div", [X, safeNorm], {}, [
            { type: dtype, shape: outShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
