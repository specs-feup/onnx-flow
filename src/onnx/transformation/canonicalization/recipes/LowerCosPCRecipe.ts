import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerCosPCRecipe implements DecompositionRecipe {
    public readonly name = "LowerCosPC";
    public readonly targetOp = "Cos";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "approximation" as const;
    public readonly priority = 60;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["taylor", "division-free"];
    public readonly producedOps = ["Add", "Sub", "Mul"];

    private readonly ITERATIONS = 10;

    match(op: OperationNode.Class): boolean {
        return op.type === "Cos";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (A.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

        const output = [{ type: dtype, shape: outShape }];

        const oneConst = builder.createConstant(
            `cos_one_${op.id}`,
            makeTensorProto(dtype, [], [1.0]),
        );

        // x^2
        const x2 = builder.createOp("Mul", [A, A], {}, output)[0];

        // first term = 1
        let currentTerm: ConcreteValueNode = oneConst;
        let currentSum: ConcreteValueNode = oneConst;

        //start with n=0 and increment by 2
        for (let n = 0; n < this.ITERATIONS * 2; n += 2) {
            const invDenom = 1.0 / ((n + 1) * (n + 2));

            const invDenomConst = builder.createConstant(
                `cos_inv_denom_${op.id}_${n}`,
                makeTensorProto(dtype, [], [invDenom]),
            );

            const mulTerm: ConcreteValueNode = builder.createOp(
                "Mul",
                [currentTerm, x2],
                {},
                output,
            )[0];
            const nextTerm: ConcreteValueNode = builder.createOp(
                "Mul",
                [mulTerm, invDenomConst],
                {},
                output,
            )[0];

            if ((n / 2) % 2 === 0) {
                currentSum = builder.createOp("Sub", [currentSum, nextTerm], {}, output)[0];
            } else {
                currentSum = builder.createOp("Add", [currentSum, nextTerm], {}, output)[0];
            }

            currentTerm = nextTerm;
        }

        builder.replaceAllUsesWith(Y, currentSum);
        op.remove();
    }
}
