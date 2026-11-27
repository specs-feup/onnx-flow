import type OperationNode from "../../../OperationNode.js";
import ConstantNode from "../../../ConstantNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, readTensorData, toStaticShape } from "../../../Utils.js";

const FLOAT_DTYPES = new Set<DataType>([DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT16]);

const SUPPORTED_INT_EXPS = new Set([0, 1, 2, 3, 4, 5, 6, 8]);
const SUPPORTED_FLOAT_EXPS = new Set([0.5, -0.5, -1, -2]);

export class LowerPowSmallIntRecipe implements DecompositionRecipe {
    public readonly name = "LowerPowSmallInt";
    public readonly targetOp = "Pow";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 100;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["algebraic-strength-reduction", "small-exponent"];
    public readonly producedOps = [
        "Mul",
        "Identity",
        "Sqrt",
        "Reciprocal",
        "ConstantOfShape",
        "Shape",
    ];

    private readUniformExponent(op: OperationNode.Class): number | undefined {
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return undefined;
        const [A, B] = ins;
        if (!B.is(ConstantNode)) return undefined;

        const bShape = B.shape;
        if (bShape.length > 0) {
            if (!Array.isArray(A.shape) || bShape.length > A.shape.length) return undefined;
            if (!bShape.every((d) => d === 1)) return undefined;
        }

        const vals = readTensorData(B);
        if (!vals || vals.length === 0) return undefined;
        const first = vals[0];
        if (!Number.isFinite(first) || !vals.every((v) => v === first)) return undefined;

        if (SUPPORTED_INT_EXPS.has(first)) return first;

        const aDtype = A.literalType as DataType | undefined;
        const isFloat =
            aDtype === undefined || aDtype === DataType.UNDEFINED || FLOAT_DTYPES.has(aDtype);
        if (isFloat && SUPPORTED_FLOAT_EXPS.has(first)) return first;

        return undefined;
    }

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Pow") return false;
        return this.readUniformExponent(op) !== undefined;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const A = ins[0];
        const Y = op.getOutputs()[0];
        const p = this.readUniformExponent(op)!;

        const dtype =
            (A.literalType as DataType | undefined) ??
            (Y.literalType as DataType | undefined) ??
            DataType.FLOAT;
        const output = [
            {
                type: (Y.literalType || dtype) as DataType,
                shape: toStaticShape(Y.shape) as KnownShape,
            },
        ];

        let res: ConcreteValueNode;
        switch (p) {
            case 0: {
                const [shapeA] = builder.createOp("Shape", [A]);
                [res] = builder.createOp(
                    "ConstantOfShape",
                    [shapeA],
                    { value: makeTensorProto(dtype, [1], [1]) },
                    output,
                );
                break;
            }
            case 1:
                [res] = builder.createOp("Identity", [A], {}, output);
                break;
            case 2:
                [res] = builder.createOp("Mul", [A, A], {}, output);
                break;
            case 3: {
                const [a2] = builder.createOp("Mul", [A, A], {}, output);
                [res] = builder.createOp("Mul", [a2, A], {}, output);
                break;
            }
            case 4: {
                const [a2] = builder.createOp("Mul", [A, A], {}, output);
                [res] = builder.createOp("Mul", [a2, a2], {}, output);
                break;
            }
            case 5: {
                const [a2] = builder.createOp("Mul", [A, A], {}, output);
                const [a4] = builder.createOp("Mul", [a2, a2], {}, output);
                [res] = builder.createOp("Mul", [a4, A], {}, output);
                break;
            }
            case 6: {
                const [a2] = builder.createOp("Mul", [A, A], {}, output);
                const [a4] = builder.createOp("Mul", [a2, a2], {}, output);
                [res] = builder.createOp("Mul", [a4, a2], {}, output);
                break;
            }
            case 8: {
                const [a2] = builder.createOp("Mul", [A, A], {}, output);
                const [a4] = builder.createOp("Mul", [a2, a2], {}, output);
                [res] = builder.createOp("Mul", [a4, a4], {}, output);
                break;
            }
            case 0.5:
                [res] = builder.createOp("Sqrt", [A], {}, output);
                break;
            case -1:
                [res] = builder.createOp("Reciprocal", [A], {}, output);
                break;
            case -2: {
                const [a2] = builder.createOp("Mul", [A, A], {}, output);
                [res] = builder.createOp("Reciprocal", [a2], {}, output);
                break;
            }
            case -0.5: {
                const [sqrtA] = builder.createOp("Sqrt", [A], {}, output);
                [res] = builder.createOp("Reciprocal", [sqrtA], {}, output);
                break;
            }
            default:
                throw new Error(`Unsupported exponent in LowerPowSmallInt: ${p}`);
        }

        builder.replaceAllUsesWith(Y, res);
        builder.removeNode(op);
    }
}
