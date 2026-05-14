import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";

export class LowerLogRecipe implements DecompositionRecipe {
    public readonly name = "LowerLog";
    public readonly targetOp = "Log";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Add", "Mul", "Div", "Sub"];
    
    private readonly ITERATIONS = 50;

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Log";
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

        const oneConst = builder.createConstant(
            `log_one_${op.id}`,
            makeTensorProto(dtype, [], [1.0]),
        );

        const twoConst = builder.createConstant(
            `log_two_${op.id}`,
            makeTensorProto(dtype, [], [2.0]),
        );

        // Z = Div(Sub(X, 1), Add(X, 1))
        const subZ = builder.createOp("Sub", [A, oneConst], {}, output)[0];
        const addZ = builder.createOp("Add", [A, oneConst], {}, output)[0];
        const Z = builder.createOp("Div", [subZ, addZ], {}, output)[0];

        // Z^2
        const z2 = builder.createOp("Mul", [Z, Z], {}, output)[0];

        let currentTerm: ConcreteValueNode = Z;
        let currentSum: ConcreteValueNode = Z;

        for (let n = 3; n < this.ITERATIONS; n += 2) {

            // currentTerm = currentTerm * Z^2
            currentTerm = builder.createOp("Mul", [currentTerm, z2], {}, output)[0];

            const nConst = builder.createConstant(
                `log_n_${op.id}_${n}`,
                makeTensorProto(dtype, [], [n]),
            );

            // nextTerm = currentTerm / n
            const nextTerm = builder.createOp("Div", [currentTerm, nConst], {}, output)[0];

            // currentSum = currentSum + nextTerm
            currentSum = builder.createOp("Add", [currentSum, nextTerm], {}, output)[0];

        }
        
        // MUL (currentSum, 2)
        const finalMul = builder.createOp("Mul", [currentSum, twoConst], {}, output)[0];

        builder.replaceAllUsesWith(Y, finalMul);
        op.remove();
    }
}

