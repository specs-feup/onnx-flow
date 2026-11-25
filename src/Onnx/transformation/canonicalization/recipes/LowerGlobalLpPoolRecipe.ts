import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, makeTensorProto } from "../../../Utils.js";

export class LowerGlobalLpPoolRecipe implements DecompositionRecipe {
    public readonly name = "LowerGlobalLpPool";
    public readonly targetOp = "GlobalLpPool";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["ReduceL1", "ReduceL2", "Abs", "Pow", "ReduceSum"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "GlobalLpPool") return false;
        const X = op.getInputs()?.[0];
        return !!X && Array.isArray(X.shape) && X.shape.length >= 3;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const p = getIntAttr(op, "p", 2);
        const spatialAxes: number[] = [];
        for (let i = 2; i < X.shape.length; i++) spatialAxes.push(i);
        const axesConst = builder.createConstant(`glp_axes_${op.id}`, int64Vec(spatialAxes));

        let out: ConcreteValueNode;
        if (p === 1) {
            [out] = builder.createOp("ReduceL1", [X, axesConst], { keepdims: 1 });
        } else if (p === 2) {
            [out] = builder.createOp("ReduceL2", [X, axesConst], { keepdims: 1 });
        } else {
            const pConst = builder.createConstant(
                `glp_p_${op.id}`,
                makeTensorProto(dtype, [], [p]),
            );
            const invPConst = builder.createConstant(
                `glp_invp_${op.id}`,
                makeTensorProto(dtype, [], [1.0 / p]),
            );
            const [absX] = builder.createOp("Abs", [X]);
            const [powP] = builder.createOp("Pow", [absX, pConst]);
            const [sumP] = builder.createOp("ReduceSum", [powP, axesConst], { keepdims: 1 });
            [out] = builder.createOp("Pow", [sumP, invPConst]);
        }

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
