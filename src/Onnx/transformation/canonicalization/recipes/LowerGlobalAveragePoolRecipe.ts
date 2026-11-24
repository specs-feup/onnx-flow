import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { int64Vec } from "../../../Utils.js";

export class LowerGlobalAveragePoolRecipe implements DecompositionRecipe {
    public readonly name = "LowerGlobalAveragePool";
    public readonly targetOp = "GlobalAveragePool";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["ReduceMean"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "GlobalAveragePool") return false;
        const X = op.getInputs()?.[0];
        return !!X && Array.isArray(X.shape) && X.shape.length >= 3;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const spatialAxes: number[] = [];
        for (let i = 2; i < X.shape.length; i++) spatialAxes.push(i);

        const axesConst = builder.createConstant(`gap_axes_${op.id}`, int64Vec(spatialAxes));
        const [out] = builder.createOp("ReduceMean", [X, axesConst], { keepdims: 1 });

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
