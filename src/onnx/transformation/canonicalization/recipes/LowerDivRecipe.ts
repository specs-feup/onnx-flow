import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import { DataType } from "../../../ir/OnnxTypes.js";

const FLOAT_DTYPES = new Set<DataType>([
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.FLOAT16,
    DataType.BFLOAT16,
]);

export class LowerDivRecipe implements DecompositionRecipe {
    public readonly name = "LowerDiv";
    public readonly targetOp = "Div";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Mul", "Reciprocal"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Div") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const dtype = ins[0].literalType as DataType | undefined;
        return dtype === undefined || dtype === DataType.UNDEFINED || FLOAT_DTYPES.has(dtype);
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const B = ins[1];
        const Y = op.getOutputs()[0];

        const dtype =
            (Y.literalType as DataType | undefined) ??
            (A.literalType as DataType | undefined) ??
            (B.literalType as DataType | undefined) ??
            DataType.FLOAT;

        const reciprocalOutput = [
            {
                type: dtype,
                shape: B.shape as KnownShape,
            },
        ];

        const Yshape: KnownShape | undefined = Y.shape as KnownShape | undefined;
        const outShape = (Yshape ?? A.shape) as KnownShape;

        const finalOutput = [
            {
                type: dtype,
                shape: outShape,
            },
        ];

        // Reciprocal(B)
        const reciprocalB = builder.createOp("Reciprocal", [B], {}, reciprocalOutput)[0];

        // A * Reciprocal(B)
        const mulOut = builder.createOp("Mul", [A, reciprocalB], {}, finalOutput)[0];

        builder.replaceAllUsesWith(Y, mulOut);
        op.remove();
    }
}
