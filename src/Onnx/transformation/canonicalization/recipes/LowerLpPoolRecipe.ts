import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { AttributeValue, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import {
    buildResolvedConvAttrs,
    getIntAttr,
    getIntsAttr,
    getStringAttr,
    makeTensorProto,
    toStaticShape,
} from "../../../Utils.js";

export class LowerLpPoolRecipe implements DecompositionRecipe {
    public readonly name = "LowerLpPool";
    public readonly targetOp = "LpPool";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["conv-based"];
    public readonly producedOps = ["Abs", "Mul", "Pow", "Conv", "AveragePool", "Sqrt"];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "LpPool") return false;
        const X = op.getInputs()?.[0];
        if (!X || !Array.isArray(X.shape) || X.shape.length < 3) return false;
        const inShape = toStaticShape(X.shape);
        const kernelShape = getIntsAttr(op, "kernel_shape", []);
        const ceilMode = getIntAttr(op, "ceil_mode", 0);
        return kernelShape.length === inShape.length - 2 && ceilMode === 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = X.literalType as DataType;
        const inShape = toStaticShape(X.shape);
        const outShape = Y.shape as KnownShape;
        const C = inShape[1];
        const kernelShape = getIntsAttr(op, "kernel_shape", []);
        const p = getIntAttr(op, "p", 2);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");

        const inMeta = [{ type: dtype, shape: inShape as KnownShape }];
        const outMeta = [{ type: dtype, shape: outShape }];

        // 1. Elementwise |X|^p
        let powered: ConcreteValueNode;
        if (p === 1) {
            [powered] = builder.createOp("Abs", [X], {}, inMeta);
        } else if (p === 2) {
            [powered] = builder.createOp("Mul", [X, X], {}, inMeta);
        } else {
            const pConst = builder.createConstant(
                `lppool_p_${op.id}`,
                makeTensorProto(dtype, [], [p]),
            );
            const [absX] = builder.createOp("Abs", [X], {}, inMeta);
            [powered] = builder.createOp("Pow", [absX, pConst], {}, inMeta);
        }

        // 2. Depthwise all-ones Conv when C is static & pads don't need dynamic spatial SAME_*;
        //    otherwise use AveragePool(count_include_pad=1) * kTotal (handles dynamic C & spatial).
        const useConvPath =
            C > 0 &&
            (autoPad === "NOTSET" || autoPad === "VALID" || !inShape.slice(2).includes(-1));

        let windowSum: ConcreteValueNode;
        if (useConvPath) {
            const weightShape = [C, 1, ...kernelShape];
            const totalWeights = weightShape.reduce((a, b) => a * b, 1);
            const onesKernel = builder.createConstant(
                `lppool_ones_${op.id}`,
                makeTensorProto(dtype, weightShape, new Array<number>(totalWeights).fill(1)),
            );

            const convAttrs = buildResolvedConvAttrs(op, inShape.slice(2), kernelShape, {
                fixedGroup: C,
            });

            [windowSum] = builder.createOp("Conv", [powered, onesKernel], convAttrs, outMeta);
        } else {
            const kTotal = kernelShape.reduce((a, b) => a * b, 1);
            const poolAttrs: Record<string, AttributeValue> = {
                kernel_shape: kernelShape,
                count_include_pad: 1,
                auto_pad: autoPad,
            };
            const strides = getIntsAttr(op, "strides", []);
            if (strides.length > 0) poolAttrs["strides"] = strides;
            const dilations = getIntsAttr(op, "dilations", []);
            if (dilations.length > 0) poolAttrs["dilations"] = dilations;
            const pads = getIntsAttr(op, "pads", []);
            if (autoPad === "NOTSET" && pads.length > 0) poolAttrs["pads"] = pads;

            const [avgOut] = builder.createOp("AveragePool", [powered], poolAttrs, outMeta);
            const kTotalConst = builder.createConstant(
                `lppool_ktot_${op.id}`,
                makeTensorProto(dtype, [], [kTotal]),
            );
            [windowSum] = builder.createOp("Mul", [avgOut, kTotalConst], {}, outMeta);
        }

        // 3. Raise window sum to 1/p
        let out: ConcreteValueNode;
        if (p === 1) {
            out = windowSum;
        } else if (p === 2) {
            [out] = builder.createOp("Sqrt", [windowSum], {}, outMeta);
        } else {
            const invPConst = builder.createConstant(
                `lppool_invp_${op.id}`,
                makeTensorProto(dtype, [], [1.0 / p]),
            );
            [out] = builder.createOp("Pow", [windowSum, invPConst], {}, outMeta);
        }

        builder.replaceAllUsesWith(Y, out);
        builder.removeNode(op);
    }
}
