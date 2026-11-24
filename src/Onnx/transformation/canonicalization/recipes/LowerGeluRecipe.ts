import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { getStringAttr, makeTensorProto } from "../../../Utils.js";

export class LowerGeluRecipe implements DecompositionRecipe {
    public readonly name = "LowerGelu";
    public readonly targetOp = "Gelu";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Add", "Mul", "Erf", "Tanh"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Gelu";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];

        const c = (tag: string, val: number) =>
            builder.createConstant(`gelu_${tag}_${op.id}`, makeTensorProto(dtype, [], [val]));

        const half = c("half", 0.5);
        const one = c("one", 1.0);
        const approxMode = getStringAttr(op, "approximate", "none");

        let inner: ConcreteValueNode;
        if (approxMode === "tanh") {
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const [x2] = builder.createOp("Mul", [X, X], {}, outMeta);
            const [x3] = builder.createOp("Mul", [x2, X], {}, outMeta);
            const [cx3] = builder.createOp("Mul", [x3, c("coeff", 0.044715)], {}, outMeta);
            const [sum] = builder.createOp("Add", [X, cx3], {}, outMeta);
            const [arg] = builder.createOp(
                "Mul",
                [sum, c("sqrt_2_over_pi", Math.sqrt(2.0 / Math.PI))],
                {},
                outMeta,
            );
            [inner] = builder.createOp("Tanh", [arg], {}, outMeta);
        } else {
            // 0.5 * x * (1 + erf(x / sqrt(2)))
            const [scaled] = builder.createOp(
                "Mul",
                [X, c("inv_sqrt2", Math.SQRT1_2)],
                {},
                outMeta,
            );
            [inner] = builder.createOp("Erf", [scaled], {}, outMeta);
        }

        const [onePlus] = builder.createOp("Add", [one, inner], {}, outMeta);
        const [halfX] = builder.createOp("Mul", [X, half], {}, outMeta);
        const [out] = builder.createOp("Mul", [halfX, onePlus], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
