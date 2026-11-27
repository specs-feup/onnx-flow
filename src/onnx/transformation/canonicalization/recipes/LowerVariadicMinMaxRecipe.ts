import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import type { DataType } from "../../../ir/OnnxTypes.js";

abstract class BaseLowerVariadicMinMaxRecipe implements DecompositionRecipe {
    public abstract readonly name: string;
    public abstract readonly targetOp: "Min" | "Max";
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["variadic"] as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public abstract readonly producedOps: string[];

    match(op: OperationNode.Class): boolean {
        if (op.type !== this.targetOp) return false;
        const len = op.getInputs()?.length ?? 0;
        return len === 1 || len > 2;
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
            [acc] = builder.createOp(
                this.targetOp,
                [acc, ins[i]],
                {},
                isLast ? outMeta : undefined,
            );
        }

        builder.replaceAllUsesWith(Y, acc);
        builder.removeNode(op);
    }
}

export class LowerVariadicMinRecipe extends BaseLowerVariadicMinMaxRecipe {
    public readonly name = "LowerVariadicMin";
    public readonly targetOp = "Min" as const;
    public readonly producedOps = ["Min", "Identity"];
}

export class LowerVariadicMaxRecipe extends BaseLowerVariadicMinMaxRecipe {
    public readonly name = "LowerVariadicMax";
    public readonly targetOp = "Max" as const;
    public readonly producedOps = ["Max", "Identity"];
}
