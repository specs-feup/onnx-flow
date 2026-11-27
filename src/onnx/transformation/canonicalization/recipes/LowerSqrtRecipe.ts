import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerSqrtRecipe implements DecompositionRecipe {
    public readonly name = "LowerSqrt";
    public readonly targetOp = "Sqrt";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "approximation" as const;
    public readonly priority = 40;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["newton-raphson"];
    public readonly producedOps = ["Add", "Mul", "Div"];

    private readonly ITERATIONS = 50;

    match(op: OperationNode.Class): boolean {
        return op.type === "Sqrt";
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

        const halfConst = builder.createConstant(
            `sqrt_half_${op.id}`,
            makeTensorProto(dtype, [], [0.5]),
        );

        const y0Const = builder.createConstant(
            `sqrt_y0_${op.id}`,
            makeTensorProto(dtype, [], [1.0]),
        );

        let currentTerm: ConcreteValueNode = y0Const;

        for (let i = 0; i < this.ITERATIONS; i++) {
            // Y1 = Mul(0.5, Add(Y0, Div(A, Y0)))
            const y1Div: ConcreteValueNode = builder.createOp(
                "Div",
                [A, currentTerm],
                {},
                output,
            )[0];
            const y1Add: ConcreteValueNode = builder.createOp(
                "Add",
                [currentTerm, y1Div],
                {},
                output,
            )[0];
            const y1: ConcreteValueNode = builder.createOp(
                "Mul",
                [halfConst, y1Add],
                {},
                output,
            )[0];
            currentTerm = y1;
        }
        builder.replaceAllUsesWith(Y, currentTerm);
        op.remove();
    }
}
