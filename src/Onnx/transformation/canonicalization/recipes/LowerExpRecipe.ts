import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../Utils.js";
import { TransformationOpportunity } from "../../TransformationOpportunity.js";

export class LowerExpRecipe implements DecompositionRecipe {
    public readonly name = "LowerExpTaylor";
    public readonly targetOp = "Exp";

    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;

    // We declare the standard ONNX math ops we will produce
    public readonly producedOps = ["Add", "Mul", "Div"];

    // The constant defining the number of approximation iterations
    private readonly ITERATIONS = 20;

    match(op: OperationNode.Class): TransformationOpportunity | null {
        if (op.type !== "Exp") return null;
        return new TransformationOpportunity(
            this.name,
            op.id,
            "Lower Exp to Taylor Series",
            (builder: GraphBuilder) => this.apply(op, builder),
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const Y = op.getOutputs()[0];

        // Fallback to FLOAT if the input datatype is undefined
        const dtype = (X.literalType as DataType | undefined) ?? DataType.FLOAT;

        // 1. Initialize term_0 = 1.0 and sum = 1.0
        // We use scalars here, ONNX's Add/Mul will automatically broadcast these to X's shape
        const oneConst = builder.createConstant(
            `taylor_1_${op.id}`,
            makeTensorProto(dtype, [], [1.0]),
        );

        let currentTerm: ConcreteValueNode = oneConst;
        let currentSum: ConcreteValueNode = oneConst;

        // 2. Unroll the Taylor series into the graph
        // Instead of computing large factorials, we calculate the series iteratively:
        // term_n = term_{n-1} * (X / n)
        // sum = sum + term_n
        for (let i = 1; i <= this.ITERATIONS; i++) {
            // Constant for the current iteration 'n'
            const nConst = builder.createConstant(
                `taylor_n_${i}_${op.id}`,
                makeTensorProto(dtype, [], [i]),
            );

            // Step A: X_div_n = X / n
            const xDivN = builder.createOp("Div", [X, nConst])[0];

            // Step B: nextTerm = currentTerm * X_div_n
            const nextTerm: ConcreteValueNode = builder.createOp("Mul", [currentTerm, xDivN])[0];

            // Step C: currentSum = currentSum + nextTerm
            currentSum = builder.createOp("Add", [currentSum, nextTerm])[0];

            // Setup for the next iteration
            currentTerm = nextTerm;
        }

        // 3. Safely replace the original Exp output with our approximated sum
        builder.replaceAllUsesWith(Y, currentSum);

        // 4. Remove the original Exp operation from the graph
        op.remove();
    }
}
