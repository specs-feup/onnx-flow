import type OperationNode from "../../../OperationNode.js";
import ConstantNode from "../../../ConstantNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, readTensorData, toStaticShape } from "../../../utils/index.js";

const FLOAT_DTYPES = new Set<DataType>([
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.FLOAT16,
    DataType.BFLOAT16,
]);

export function isExactPowerOfTwoFloat(v: number, dt: DataType): boolean {
    if (!Number.isFinite(v) || v === 0) return false;
    const exp = Math.log2(Math.abs(v));
    if (!Number.isInteger(exp)) return false;
    // Ensure both v and 1/v stay within the finite normal mantissa exponent range of dt
    const maxAbsExp =
        dt === DataType.DOUBLE
            ? 1022
            : dt === DataType.FLOAT16
              ? 14
              : dt === DataType.BFLOAT16
                ? 126
                : 126;
    return Math.abs(exp) <= maxAbsExp;
}

export function applyDivByReciprocalConstant(op: OperationNode.Class, builder: GraphBuilder): void {
    const ins = op.getInputs() as ConcreteValueNode[];
    const A = ins[0];
    const B = ins[1] as ConstantNode.Class;
    const Y = op.getOutputs()[0];

    const tv = B.constantValue;
    const dt = tv.dataType as DataType;
    const dims = (tv.dims ?? []).map(Number);
    const vals = readTensorData(B)!;
    const output = [
        {
            type: (Y.literalType || dt) as DataType,
            shape: toStaticShape(Y.shape) as KnownShape,
        },
    ];

    // Fast-path identity division by scalar 1.0
    if (dims.length === 0 && vals[0] === 1) {
        const idOut = builder.createOp("Identity", [A], {}, output)[0];
        builder.replaceAllUsesWith(Y, idOut);
        builder.removeNode(op);
        return;
    }

    const invVals = vals.map((v) => 1 / v);
    const invConst = builder.createConstant(`div_inv_${op.id}`, makeTensorProto(dt, dims, invVals));
    const mulOut = builder.createOp("Mul", [A, invConst], {}, output)[0];

    builder.replaceAllUsesWith(Y, mulOut);
    builder.removeNode(op);
}

export class LowerDivByConstantRecipe implements DecompositionRecipe {
    public readonly name = "LowerDivByConstant";
    public readonly targetOp = "Div";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "approximation" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["algebraic-strength-reduction", "constant-divisor"];
    public readonly producedOps = ["Mul", "Identity"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Div") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        const B = ins[1];
        if (!B.is(ConstantNode)) return false;

        const tv = B.as(ConstantNode).constantValue;
        const dt = tv.dataType as DataType;
        if (!FLOAT_DTYPES.has(dt)) return false;

        const vals = readTensorData(B);
        if (!vals || vals.length === 0) return false;

        const limit = dt === DataType.DOUBLE ? 1e300 : 3e38;
        const minMag = dt === DataType.DOUBLE ? 1e-300 : 1e-35;

        return vals.every(
            (v) =>
                Number.isFinite(v) &&
                v !== 0 &&
                Math.abs(v) >= minMag &&
                Number.isFinite(1 / v) &&
                Math.abs(1 / v) <= limit,
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        applyDivByReciprocalConstant(op, builder);
    }
}
