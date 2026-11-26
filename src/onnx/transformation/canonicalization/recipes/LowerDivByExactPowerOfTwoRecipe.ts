import type OperationNode from "../../../OperationNode.js";
import ConstantNode from "../../../ConstantNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import { DataType } from "../../../OnnxTypes.js";
import { readTensorData } from "../../../utils/index.js";
import {
    applyDivByReciprocalConstant,
    isExactPowerOfTwoFloat,
} from "./LowerDivByConstantRecipe.js";

const FLOAT_DTYPES = new Set<DataType>([
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.FLOAT16,
    DataType.BFLOAT16,
]);

/**
 * Exact strength reduction for floating-point division by normal power-of-two constants
 * (e.g., /2, /4, /0.5), where multiplying by 1/c is bit-exact without mantissa rounding.
 */
export class LowerDivByExactPowerOfTwoRecipe implements DecompositionRecipe {
    public readonly name = "LowerDivByExactPowerOfTwo";
    public readonly targetOp = "Div";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 110;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["algebraic-strength-reduction", "exact-power-of-two"];
    public readonly producedOps = ["Mul", "Identity"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Div") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;

        const B = ins[1];
        if (!B.is(ConstantNode)) return false;

        const dt = B.as(ConstantNode).constantValue.dataType as DataType;
        if (!FLOAT_DTYPES.has(dt)) return false;

        const vals = readTensorData(B);
        if (!vals || vals.length === 0) return false;
        return vals.every((v) => isExactPowerOfTwoFloat(v, dt));
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        applyDivByReciprocalConstant(op, builder);
    }
}
