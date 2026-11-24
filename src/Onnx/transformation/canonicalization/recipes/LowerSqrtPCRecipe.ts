import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerSqrtPCRecipe implements DecompositionRecipe {
    public readonly name = "LowerSqrtPC";
    public readonly targetOp = "Sqrt";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "approximation" as const;
    public readonly priority = 60;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["range-reduced", "newton-raphson", "division-free"];
    public readonly producedOps = ["Sub", "Mul", "Less", "Greater", "Where"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Sqrt";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const A = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype =
            (A.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];
        const boolMeta = [{ type: DataType.BOOL, shape: outShape }];

        const c = (tag: string, val: number) =>
            builder.createConstant(`sqrt_${tag}_${op.id}`, makeTensorProto(dtype, [], [val]));

        const zero = c("zero", 0.0);
        const one = c("one", 1.0);
        const half = c("half", 0.5);
        const quarter = c("quarter", 0.25);
        const two = c("two", 2.0);
        const three = c("three", 3.0);
        const oneHalf = c("one_half", 1.5);

        // Guard X <= 0 so range reduction stays finite and sqrt(0) = 0
        const [isNonPos] = builder.createOp("Less", [A, c("eps", 1e-38)], {}, boolMeta);
        let m: ConcreteValueNode = builder.createOp("Where", [isNonPos, one, A], {}, outMeta)[0];
        let scale: ConcreteValueNode | null = null;

        // 1. Power-of-4 range reduction into [0.25, 4.0]
        for (const p of [16, 8, 4, 2, 1]) {
            const hi = c(`hi_${p}`, Math.pow(4, p));
            const lo = c(`lo_${p}`, Math.pow(4, -p));
            const sUpVal = c(`sup_${p}`, Math.pow(2, p));
            const sDnVal = c(`sdn_${p}`, Math.pow(2, -p));

            const [tooBig] = builder.createOp("Greater", [m, hi], {}, boolMeta);
            const [mDown] = builder.createOp("Mul", [m, lo], {}, outMeta);
            [m] = builder.createOp("Where", [tooBig, mDown, m], {}, outMeta);
            if (scale === null) {
                [scale] = builder.createOp("Where", [tooBig, sUpVal, one], {}, outMeta);
            } else {
                const [sUp] = builder.createOp("Mul", [scale, sUpVal], {}, outMeta);
                [scale] = builder.createOp("Where", [tooBig, sUp, scale], {}, outMeta);
            }

            const [tooSmall] = builder.createOp("Less", [m, lo], {}, boolMeta);
            const [mUp] = builder.createOp("Mul", [m, hi], {}, outMeta);
            const [sDn] = builder.createOp("Mul", [scale, sDnVal], {}, outMeta);
            [m] = builder.createOp("Where", [tooSmall, mUp, m], {}, outMeta);
            [scale] = builder.createOp("Where", [tooSmall, sDn, scale], {}, outMeta);
        }

        // 2. Final fold [1.0, 4.0] -> [0.25, 1.0] (guarantees NR convergence!)
        const [gtOne] = builder.createOp("Greater", [m, one], {}, boolMeta);
        const [mQuarter] = builder.createOp("Mul", [m, quarter], {}, outMeta);
        const [sDouble] = builder.createOp("Mul", [scale!, two], {}, outMeta);
        [m] = builder.createOp("Where", [gtOne, mQuarter, m], {}, outMeta);
        [scale] = builder.createOp("Where", [gtOne, sDouble, scale!], {}, outMeta);

        // 3. Linear seed y0 = 1.5 - 0.5 * m (guaranteed outShape, exact at m = 1)
        const [halfM] = builder.createOp("Mul", [half, m], {}, outMeta);
        let y: ConcreteValueNode = builder.createOp("Sub", [oneHalf, halfM], {}, outMeta)[0];

        // 4. Five Newton-Raphson steps for y = 1 / sqrt(m): y = 0.5 * y * (3 - m * y * y)
        for (let i = 0; i < 5; i++) {
            const [y2] = builder.createOp("Mul", [y, y], {}, outMeta);
            const [my2] = builder.createOp("Mul", [m, y2], {}, outMeta);
            const [threeMinus] = builder.createOp("Sub", [three, my2], {}, outMeta);
            const [yStep] = builder.createOp("Mul", [y, threeMinus], {}, outMeta);
            [y] = builder.createOp("Mul", [half, yStep], {}, outMeta);
        }

        // 5. sqrt(A) = m * y * scale (or 0 if A <= 0)
        const [sqrtM] = builder.createOp("Mul", [m, y], {}, outMeta);
        const [rescaled] = builder.createOp("Mul", [sqrtM, scale!], {}, outMeta);
        const [finalOut] = builder.createOp("Where", [isNonPos, zero, rescaled], {}, outMeta);

        builder.replaceAllUsesWith(Y, finalOut);
        builder.removeNode(op);
    }
}
