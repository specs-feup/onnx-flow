import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerHardSwishRecipe implements DecompositionRecipe {
    public readonly name = "LowerHardSwish";
    public readonly targetOp = "HardSwish";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Add", "Clip", "Mul"];

    match(op: OperationNode.Class): boolean {
        return op.type === "HardSwish";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];

        const invSix = builder.createConstant(
            `hswish_inv6_${op.id}`,
            makeTensorProto(dtype, [], [1.0 / 6.0]),
        );
        const half = builder.createConstant(
            `hswish_half_${op.id}`,
            makeTensorProto(dtype, [], [0.5]),
        );
        const zero = builder.createConstant(
            `hswish_zero_${op.id}`,
            makeTensorProto(dtype, [], [0.0]),
        );
        const one = builder.createConstant(
            `hswish_one_${op.id}`,
            makeTensorProto(dtype, [], [1.0]),
        );

        const [scaled] = builder.createOp("Mul", [X, invSix], {}, outMeta);
        const [shifted] = builder.createOp("Add", [scaled, half], {}, outMeta);
        const [clipped] = builder.createOp("Clip", [shifted, zero, one], {}, outMeta);
        const [out] = builder.createOp("Mul", [X, clipped], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
