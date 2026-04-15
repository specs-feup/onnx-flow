import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getFloatAttr, getIntAttr, makeTensorProto } from "../../../Utils.js";

export class LowerGemmRecipe implements DecompositionRecipe {
    public readonly name = "LowerGemm";
    public readonly targetOp = "Gemm";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Transpose", "MatMul", "Mul", "Add"];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Gemm";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as (ConcreteValueNode | undefined)[];
        const A = ins[0]!;
        const B = ins[1]!;
        const C = ins.length > 2 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const alpha = getFloatAttr(op, "alpha", 1.0);
        const beta = getFloatAttr(op, "beta", 1.0);
        const transA = getIntAttr(op, "transA", 0) === 1;
        const transB = getIntAttr(op, "transB", 0) === 1;

        const dtypeLeft = (A.literalType as DataType | undefined) ?? DataType.FLOAT;
        const dtypeRight = C ? (C.literalType as DataType) : dtypeLeft;

        let A_in = A;
        let B_in = B;

        if (transA) {
            A_in = builder.createOp("Transpose", [A_in], { perm: [1, 0] })[0];
        }
        if (transB) {
            B_in = builder.createOp("Transpose", [B_in], { perm: [1, 0] })[0];
        }

        // Base MatMul
        let left = builder.createOp("MatMul", [A_in, B_in])[0];

        // Apply alpha scaling
        if (alpha !== 1.0) {
            const aC = builder.createConstant(
                `Gemm_alpha_${op.id}`,
                makeTensorProto(dtypeLeft, [], [alpha]),
            );
            left = builder.createOp("Mul", [left, aC])[0];
        }

        // Apply beta * C addition
        if (C && beta !== 0.0) {
            let cTerm = C;
            if (beta !== 1.0) {
                const bC = builder.createConstant(
                    `Gemm_beta_${op.id}`,
                    makeTensorProto(dtypeRight, [], [beta]),
                );
                cTerm = builder.createOp("Mul", [C, bC])[0];
            }
            left = builder.createOp("Add", [left, cTerm])[0];
        }

        // Safely replace Y and clean up
        builder.replaceAllUsesWith(Y, left);
        op.remove();
    }
}
