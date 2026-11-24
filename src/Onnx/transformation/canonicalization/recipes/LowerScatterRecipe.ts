import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { getIntAttr } from "../../../Utils.js";

export class LowerScatterRecipe implements DecompositionRecipe {
    public readonly name = "LowerScatter";
    public readonly targetOp = "Scatter";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["alias"];
    public readonly producedOps = ["ScatterElements"];

    match(op: OperationNode.Class): boolean {
        return op.type === "Scatter" && (op.getInputs()?.length ?? 0) >= 3;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [data, indices, updates] = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];
        const axis = getIntAttr(op, "axis", 0);
        const dtype = data.literalType as DataType;

        const [out] = builder.createOp("ScatterElements", [data, indices, updates], { axis }, [
            { type: dtype, shape: Y.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
