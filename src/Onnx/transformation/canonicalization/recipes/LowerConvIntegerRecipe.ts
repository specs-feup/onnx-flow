import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { AttributeMap, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { getAttr, int64Vec, toStaticShape } from "../../../Utils.js";

const CONV_ATTRS = ["auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"] as const;

export class LowerConvIntegerRecipe implements DecompositionRecipe {
    public readonly name = "LowerConvInteger";
    public readonly targetOp = "ConvInteger";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly producedOps = ["Cast", "Reshape", "Sub", "Conv"];

    match(op: OperationNode.Class): boolean {
        return op.type === "ConvInteger" && (op.getInputs()?.length ?? 0) >= 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const X = ins[0];
        const W = ins[1];
        const xZp = ins.length >= 3 ? ins[2] : undefined;
        const wZp = ins.length >= 4 ? ins[3] : undefined;
        const Y = op.getOutputs()[0];

        let [xEff] = builder.createOp("Cast", [X], { to: DataType.INT32 }, [
            { type: DataType.INT32, shape: X.shape as KnownShape },
        ]);
        if (xZp) {
            const [xzpI32] = builder.createOp("Cast", [xZp], { to: DataType.INT32 }, [
                { type: DataType.INT32, shape: xZp.shape as KnownShape },
            ]);
            [xEff] = builder.createOp("Sub", [xEff, xzpI32], {}, [
                { type: DataType.INT32, shape: X.shape as KnownShape },
            ]);
        }

        let [wEff] = builder.createOp("Cast", [W], { to: DataType.INT32 }, [
            { type: DataType.INT32, shape: W.shape as KnownShape },
        ]);
        if (wZp) {
            let [wzpI32] = builder.createOp("Cast", [wZp], { to: DataType.INT32 }, [
                { type: DataType.INT32, shape: wZp.shape as KnownShape },
            ]);
            if (wZp.shape.length === 1 && W.shape.length > 1) {
                const wShape = toStaticShape(W.shape);
                const bcastShape = [wShape[0], ...new Array<number>(W.shape.length - 1).fill(1)];
                const shapeConst = builder.createConstant(
                    `ci_wzp_s_${op.id}`,
                    int64Vec(bcastShape),
                );
                [wzpI32] = builder.createOp("Reshape", [wzpI32, shapeConst], {}, [
                    { type: DataType.INT32, shape: bcastShape as KnownShape },
                ]);
            }
            [wEff] = builder.createOp("Sub", [wEff, wzpI32], {}, [
                { type: DataType.INT32, shape: W.shape as KnownShape },
            ]);
        }

        const convAttrs: AttributeMap = {};
        for (const key of CONV_ATTRS) {
            const val = getAttr(op, key);
            if (val !== undefined) convAttrs[key] = val as AttributeMap[string];
        }

        const [out] = builder.createOp("Conv", [xEff, wEff], convAttrs, [
            { type: DataType.INT32, shape: Y.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
