import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../../utils/index.js";

export class LowerMeanRecipe implements DecompositionRecipe {
    public readonly name = "LowerMean";
    public readonly targetOp = "Mean";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Add", "Mul", "Identity"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Mean" && (op.getInputs()?.length ?? 0) >= 1;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];
        const dtype = ins[0].literalType as DataType;
        const outShape = Y.shape as KnownShape;
        const outMeta = [{ type: dtype, shape: outShape }];

        if (ins.length === 1) {
            const [idOut] = builder.createOp("Identity", [ins[0]], {}, outMeta);
            builder.replaceAllUsesWith(Y, idOut);
            builder.removeNode(op);
            return;
        }

        let acc: ConcreteValueNode = ins[0];
        for (let i = 1; i < ins.length; i++) {
            const isLast = i === ins.length - 1;
            [acc] = builder.createOp("Add", [acc, ins[i]], {}, isLast ? outMeta : undefined);
        }

        const invN = builder.createConstant(
            `mean_inv_n_${op.id}`,
            makeTensorProto(dtype, [], [1.0 / ins.length]),
        );
        const [out] = builder.createOp("Mul", [acc, invN], {}, outMeta);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
