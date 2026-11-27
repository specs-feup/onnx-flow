import type OperationNode from "../../../ir/OperationNode.js";
import type { GraphBuilder } from "../../../ir/GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../ir/OnnxTypes.js";
import type { DataType } from "../../../ir/OnnxTypes.js";

export class LowerQLinearMatMulRecipe implements DecompositionRecipe {
    public readonly name = "LowerQLinearMatMul";
    public readonly targetOp = "QLinearMatMul";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["DequantizeLinear", "MatMul", "QuantizeLinear"];

    match(op: OperationNode.Class): boolean {
        return op.type === "QLinearMatMul" && (op.getInputs()?.length ?? 0) >= 8;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const [a, aScale, aZp, b, bScale, bZp, yScale, yZp] = op.getInputs() as ConcreteValueNode[];
        const Y = op.getOutputs()[0];

        const floatDtype = aScale.literalType as DataType;
        const outDtype = Y.literalType as DataType;

        const aAttrs =
            aScale.shape.length === 1 && a.shape.length >= 2 ? { axis: a.shape.length - 2 } : {};
        const bAttrs =
            bScale.shape.length === 1 && b.shape.length >= 2 ? { axis: b.shape.length - 1 } : {};
        const yAttrs =
            yScale.shape.length === 1 && Y.shape.length >= 2 ? { axis: Y.shape.length - 1 } : {};

        const [aF] = builder.createOp("DequantizeLinear", [a, aScale, aZp], aAttrs, [
            { type: floatDtype, shape: a.shape as KnownShape },
        ]);
        const [bF] = builder.createOp("DequantizeLinear", [b, bScale, bZp], bAttrs, [
            { type: floatDtype, shape: b.shape as KnownShape },
        ]);
        const [mmF] = builder.createOp("MatMul", [aF, bF], {}, [
            { type: floatDtype, shape: Y.shape as KnownShape },
        ]);
        const [out] = builder.createOp("QuantizeLinear", [mmF, yScale, yZp], yAttrs, [
            { type: outDtype, shape: Y.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
