import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { AttributeMap, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { getAttr } from "../../../Utils.js";

const CONV_ATTRS = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"] as const;

export class LowerQLinearConvRecipe implements DecompositionRecipe {
    public readonly name = "LowerQLinearConv";
    public readonly targetOp = "QLinearConv";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["DequantizeLinear", "Cast", "Mul", "Conv", "QuantizeLinear"];

    match(op: OperationNode.Class): boolean {
        return op.type === "QLinearConv" && (op.getInputs()?.length ?? 0) >= 8;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const [x, xScale, xZp, w, wScale, wZp, yScale, yZp] = ins;
        const B = ins.length >= 9 ? ins[8] : undefined;
        const Y = op.getOutputs()[0];

        const floatDtype = xScale.literalType as DataType;
        const outDtype = Y.literalType as DataType;

        const wAttrs = wScale.shape.length === 1 ? { axis: 0 } : {};
        const [xF] = builder.createOp("DequantizeLinear", [x, xScale, xZp], {}, [
            { type: floatDtype, shape: x.shape as KnownShape },
        ]);
        const [wF] = builder.createOp("DequantizeLinear", [w, wScale, wZp], wAttrs, [
            { type: floatDtype, shape: w.shape as KnownShape },
        ]);

        const convInputs: ConcreteValueNode[] = [xF, wF];
        if (B) {
            const [bFloat] = builder.createOp("Cast", [B], { to: floatDtype }, [
                { type: floatDtype, shape: B.shape as KnownShape },
            ]);
            const [bScale] = builder.createOp("Mul", [xScale, wScale]);
            const [bDequant] = builder.createOp("Mul", [bFloat, bScale], {}, [
                { type: floatDtype, shape: B.shape as KnownShape },
            ]);
            convInputs.push(bDequant);
        }

        const convAttrs: AttributeMap = {};
        for (const key of CONV_ATTRS) {
            const val = getAttr(op, key);
            if (val !== undefined) convAttrs[key] = val as AttributeMap[string];
        }

        const [convF] = builder.createOp("Conv", convInputs, convAttrs, [
            { type: floatDtype, shape: Y.shape as KnownShape },
        ]);
        const [out] = builder.createOp("QuantizeLinear", [convF, yScale, yZp], {}, [
            { type: outDtype, shape: Y.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
