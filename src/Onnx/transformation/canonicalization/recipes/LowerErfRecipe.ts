import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerErfRecipe implements DecompositionRecipe {
    public readonly name = "LowerErf";
    public readonly targetOp = "Erf";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Add", "Sub", "Mul"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Erf";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype =
            (A.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const twoOverSqrtPiConst = builder.createConstant(
            `erf_two_over_sqrt_pi_${op.id}`,
            makeTensorProto(dtype, [], [1.1283791671]),
        );

        const oneThirdConst = builder.createConstant(
            `erf_one_third_${op.id}`,
            makeTensorProto(dtype, [], [1.0 / 3.0]),
        );

        const oneTenthConst = builder.createConstant(
            `erf_one_tenth_${op.id}`,
            makeTensorProto(dtype, [], [1.0 / 10.0]),
        );

        // x^2
        const x2: ConcreteValueNode = builder.createOp("Mul", [A, A], {}, output)[0];

        // x^3 = x^2 * x
        const x3: ConcreteValueNode = builder.createOp("Mul", [x2, A], {}, output)[0];

        // x^5 = x^3 * x^2
        const x5: ConcreteValueNode = builder.createOp("Mul", [x3, x2], {}, output)[0];

        // x^3 / 3
        const x3Term: ConcreteValueNode = builder.createOp("Mul", [oneThirdConst, x3], {}, output)[0];

        // x^5 / 10
        const x5Term: ConcreteValueNode = builder.createOp("Mul", [oneTenthConst, x5], {}, output)[0];

        // x - x^3/3
        const sub1: ConcreteValueNode = builder.createOp("Sub", [A, x3Term], {}, output)[0];

        // x - x^3/3 + x^5/10
        const poly: ConcreteValueNode = builder.createOp("Add", [sub1, x5Term], {}, output)[0];

        // 2/sqrt(pi) * (...)
        const erfApprox: ConcreteValueNode = builder.createOp(
            "Mul",
            [twoOverSqrtPiConst, poly],
            {},
            output,
        )[0];

        builder.replaceAllUsesWith(Y, erfApprox);
        op.remove();
    }
}