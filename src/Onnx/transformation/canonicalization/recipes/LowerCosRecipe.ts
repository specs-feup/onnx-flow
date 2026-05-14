import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerCosRecipe implements DecompositionRecipe {
    public readonly name = "LowerCos";
    public readonly targetOp = "Cos";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Add", "Sub", "Mul", "Div"];

    private readonly ITERATIONS = 10;

    canApply(op: OperationNode.Class): boolean {
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

        const outShape =
            (Y.shape as KnownShape) ??
            (A.shape as KnownShape);

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
            const denominator = (n + 1) * (n + 2);

            const denomConst = builder.createConstant(
                `cos_denom_${op.id}_${n}`,
                makeTensorProto(dtype, [], [denominator]),
            );

            const mulTerm: ConcreteValueNode = builder.createOp("Mul", [currentTerm, x2], {}, output)[0];
            const nextTerm: ConcreteValueNode = builder.createOp("Div", [mulTerm, denomConst], {}, output)[0];

            // Alternate adding and subtracting
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