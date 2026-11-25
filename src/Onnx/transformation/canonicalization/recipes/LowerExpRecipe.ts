import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerExpRecipe implements DecompositionRecipe {
    public readonly name = "LowerExp";
    public readonly targetOp = "Exp";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "approximation" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["range-reduced", "horner", "division-free"];
    public readonly producedOps = ["Add", "Sub", "Mul", "Floor", "Less", "Greater", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Exp";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype =
            (X.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        const c = (tag: string, val: number) =>
            builder.createConstant(`exp_${tag}_${op.id}`, makeTensorProto(dtype, [], [val]));

        // 1. Clamp X to [-88, 88]
        const minX = c("min_x", -88.0);
        const maxX = c("max_x", 88.0);
        const [belowMin] = builder.createOp("Less", [X, minX], {}, boolMeta);
        const [clampedLo] = builder.createOp("Where", [belowMin, minX, X], {}, outMeta);
        const [aboveMax] = builder.createOp("Greater", [clampedLo, maxX], {}, boolMeta);
        const [xc] = builder.createOp("Where", [aboveMax, maxX, clampedLo], {}, outMeta);

        // 2. Cody-Waite range reduction: k = Floor(xc * log2(e) + 0.5)
        //    r = (xc - k * LN2_HI) - k * LN2_LO  (exact 16-bit HI preserves FP32 accuracy)
        const invLn2 = c("inv_ln2", Math.LOG2E);
        const ln2Hi = c("ln2_hi", 0.693145751953125);
        const ln2Lo = c("ln2_lo", 1.428606765330187e-6);
        const half = c("half", 0.5);
        const one = c("one", 1.0);

        const [scaled] = builder.createOp("Mul", [xc, invLn2], {}, outMeta);
        const [shifted] = builder.createOp("Add", [scaled, half], {}, outMeta);
        const [k] = builder.createOp("Floor", [shifted], {}, outMeta);
        const [kLn2Hi] = builder.createOp("Mul", [k, ln2Hi], {}, outMeta);
        const [kLn2Lo] = builder.createOp("Mul", [k, ln2Lo], {}, outMeta);
        const [rHi] = builder.createOp("Sub", [xc, kLn2Hi], {}, outMeta);
        const [r] = builder.createOp("Sub", [rHi, kLn2Lo], {}, outMeta);

        // 3. Degree-7 Horner polynomial for exp(r) on [-ln(2)/2, ln(2)/2]
        const coeffs = [
            1.0 / 5040.0,
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            1.0 / 2.0,
            1.0,
            1.0,
        ];
        let poly: ConcreteValueNode = c("c0", coeffs[0]);
        for (let i = 1; i < coeffs.length; i++) {
            const [pr] = builder.createOp("Mul", [poly, r], {}, outMeta);
            [poly] = builder.createOp("Add", [pr, c(`c${i}`, coeffs[i])], {}, outMeta);
        }

        // 4. Reconstruct 2^k via 7-step binary cascade
        let kRem: ConcreteValueNode = k;
        let pow2: ConcreteValueNode | null = null;

        for (const bit of [64, 32, 16, 8, 4, 2, 1]) {
            const posThresh = c(`pos_t_${bit}`, bit - 0.5);
            const negThresh = c(`neg_t_${bit}`, -(bit - 0.5));
            const bitVal = c(`bit_${bit}`, bit);
            const twoPowPos = c(`pow2_p_${bit}`, Math.pow(2, bit));
            const twoPowNeg = c(`pow2_n_${bit}`, Math.pow(2, -bit));

            const [hasPos] = builder.createOp("Greater", [kRem, posThresh], {}, boolMeta);
            const [kSubBit] = builder.createOp("Sub", [kRem, bitVal], {}, outMeta);
            if (pow2 === null) {
                // First step: avoid Mul(scalar_1, scalar_2^64) with outMeta
                [pow2] = builder.createOp("Where", [hasPos, twoPowPos, one], {}, outMeta);
            } else {
                const [pMulPos] = builder.createOp("Mul", [pow2, twoPowPos], {}, outMeta);
                [pow2] = builder.createOp("Where", [hasPos, pMulPos, pow2], {}, outMeta);
            }
            [kRem] = builder.createOp("Where", [hasPos, kSubBit, kRem], {}, outMeta);

            const [hasNeg] = builder.createOp("Less", [kRem, negThresh], {}, boolMeta);
            const [pMulNeg] = builder.createOp("Mul", [pow2, twoPowNeg], {}, outMeta);
            const [kAddBit] = builder.createOp("Add", [kRem, bitVal], {}, outMeta);
            [pow2] = builder.createOp("Where", [hasNeg, pMulNeg, pow2], {}, outMeta);
            [kRem] = builder.createOp("Where", [hasNeg, kAddBit, kRem], {}, outMeta);
        }

        const [finalOut] = builder.createOp("Mul", [poly, pow2!], {}, outMeta);

        builder.replaceAllUsesWith(Y, finalOut);
        builder.removeNode(op);
    }
}
