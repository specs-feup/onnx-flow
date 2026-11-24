import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerReciprocalRecipe implements DecompositionRecipe {
    public readonly name = "LowerReciprocal";
    public readonly targetOp = "Reciprocal";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "approximation" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["range-reduced", "newton-raphson", "division-free"];
    public readonly producedOps = ["Less", "Greater", "Where", "Mul", "Sub"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Reciprocal";
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
            builder.createConstant(`rcp_${tag}_${op.id}`, makeTensorProto(dtype, [], [val]));

        const zero = c("zero", 0.0);
        const one = c("one", 1.0);
        const minusOne = c("neg1", -1.0);
        const two = c("two", 2.0);
        const half = c("half", 0.5);

        // 1. Separate sign and magnitude: |X| = X * sign
        const [isNeg] = builder.createOp("Less", [X, zero], {}, boolMeta);
        const [sign] = builder.createOp("Where", [isNeg, minusOne, one], {}, outMeta);
        let m: ConcreteValueNode = builder.createOp("Mul", [X, sign], {}, outMeta)[0];
        let scale: ConcreteValueNode | null = null;

        // 2. Binary power-of-2 range reduction into [0.5, 2.0)
        for (const p of [16, 8, 4, 2, 1]) {
            const hi = c(`hi_${p}`, Math.pow(2, p));
            const lo = c(`lo_${p}`, Math.pow(2, -p));

            const [tooBig] = builder.createOp("Greater", [m, hi], {}, boolMeta);
            const [mDown] = builder.createOp("Mul", [m, lo], {}, outMeta);
            [m] = builder.createOp("Where", [tooBig, mDown, m], {}, outMeta);
            if (scale === null) {
                [scale] = builder.createOp("Where", [tooBig, lo, one], {}, outMeta);
            } else {
                const [sDown] = builder.createOp("Mul", [scale, lo], {}, outMeta);
                [scale] = builder.createOp("Where", [tooBig, sDown, scale], {}, outMeta);
            }

            const [tooSmall] = builder.createOp("Less", [m, lo], {}, boolMeta);
            const [mUp] = builder.createOp("Mul", [m, hi], {}, outMeta);
            const [sUp] = builder.createOp("Mul", [scale, hi], {}, outMeta);
            [m] = builder.createOp("Where", [tooSmall, mUp, m], {}, outMeta);
            [scale] = builder.createOp("Where", [tooSmall, sUp, scale], {}, outMeta);
        }

        // Final fold [1.0, 2.0) -> [0.5, 1.0)
        const [geOne] = builder.createOp("Greater", [m, one], {}, boolMeta);
        const [mHalf] = builder.createOp("Mul", [m, half], {}, outMeta);
        const [sHalf] = builder.createOp("Mul", [scale!, half], {}, outMeta);
        [m] = builder.createOp("Where", [geOne, mHalf, m], {}, outMeta);
        [scale] = builder.createOp("Where", [geOne, sHalf, scale!], {}, outMeta);

        // 3. Minimax linear seed on [0.5, 1.0): y0 = 48/17 - (32/17) * m
        const a0 = c("seed_a", 48.0 / 17.0);
        const b0 = c("seed_b", 32.0 / 17.0);
        const [bm] = builder.createOp("Mul", [b0, m], {}, outMeta);
        let y: ConcreteValueNode = builder.createOp("Sub", [a0, bm], {}, outMeta)[0];

        // 4. Four Newton-Raphson iterations: y = y * (2 - m * y)
        for (let i = 0; i < 4; i++) {
            const [my] = builder.createOp("Mul", [m, y], {}, outMeta);
            const [twoMinusMy] = builder.createOp("Sub", [two, my], {}, outMeta);
            [y] = builder.createOp("Mul", [y, twoMinusMy], {}, outMeta);
        }

        // 5. Rescale and restore sign: sign * y * scale
        const [yScaled] = builder.createOp("Mul", [y, scale], {}, outMeta);
        const [finalOut] = builder.createOp("Mul", [yScaled, sign], {}, outMeta);

        builder.replaceAllUsesWith(Y, finalOut);
        builder.removeNode(op);
    }
}
