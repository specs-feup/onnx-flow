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
    public readonly producedOps = ["Add", "Mul", "Neg"];
    
    private readonly ITERATIONS = 10;

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Reciprocal";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];

        const dtype = (A.literalType as DataType | undefined) ?? 
                      (Y.literalType as DataType | undefined) ??    
                      DataType.FLOAT;

        const outShape = (Y.shape as KnownShape) ??
                         (A.shape as KnownShape);

        const output = [{ type: dtype, shape: outShape }];

        const y0Const = builder.createConstant(
            `reciprocal_y0_${op.id}`,
            makeTensorProto(dtype, [], [1.0]),
        );

        const twoConst = builder.createConstant(
            `reciprocal_two_${op.id}`,
            makeTensorProto(dtype, [], [2.0]),
        );

        let currentTerm: ConcreteValueNode = y0Const;

        for (let i = 0; i < this.ITERATIONS; i++) {

            // Y1 = Mul(Y0, Add(2, Neg(Mul(X, Y0)))))
            const y1Mul: ConcreteValueNode = builder.createOp("Mul", [A, currentTerm], {}, output)[0];
            const y1Neg: ConcreteValueNode = builder.createOp("Neg", [y1Mul], {}, output)[0];
            const y1Add: ConcreteValueNode = builder.createOp("Add", [twoConst, y1Neg], {}, output)[0];
            const nextTerm: ConcreteValueNode = builder.createOp("Mul", [currentTerm, y1Add], {}, output)[0];

            currentTerm = nextTerm;
        }
        builder.replaceAllUsesWith(Y, currentTerm);
        op.remove();
    }
}

