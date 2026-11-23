import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getAttr,
    getStringAttr,
    int64Vec,
    makeTensorProto,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";

export class LowerNegativeLogLikelihoodLossRecipe implements DecompositionRecipe {
    public readonly name = "LowerNegativeLogLikelihoodLoss";
    public readonly targetOp = "NegativeLogLikelihoodLoss";
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = [
        "Cast",
        "Equal",
        "Not",
        "Where",
        "OneHot",
        "Reshape",
        "Mul",
        "ReduceSum",
        "Neg",
        "Div",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "NegativeLogLikelihoodLoss") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2 || !Array.isArray(ins[0].shape) || ins[0].shape.length < 2) {
            return false;
        }
        const inShape = toStaticShape(ins[0].shape);
        return inShape[1] > 0;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const input = ins[0];
        const target = ins[1];
        const weight = ins.length >= 3 ? ins[2] : undefined;
        const Y = op.getOutputs()[0];

        const dtype = input.literalType as DataType;
        const inShape = toStaticShape(input.shape);
        const rank = inShape.length;
        const C = inShape[1];
        const targetShape = [inShape[0], ...inShape.slice(2)];

        const reduction = getStringAttr(op, "reduction", "mean");
        const rawIgnore = getAttr(op, "ignore_index");
        const hasIgnore = typeof rawIgnore === "number";

        const inMeta = [{ type: dtype, shape: inShape as KnownShape }];
        const targetFloatMeta = [{ type: dtype, shape: targetShape as KnownShape }];
        const targetI64Meta = [{ type: DataType.INT64, shape: targetShape as KnownShape }];
        const targetBoolMeta = [{ type: DataType.BOOL, shape: targetShape as KnownShape }];
        const scalarMeta = [{ type: dtype, shape: [] as KnownShape }];

        // 1. Cast target to INT64 and mask ignore_index if present
        let target64: ConcreteValueNode = target;
        if (target.literalType !== DataType.INT64) {
            [target64] = builder.createOp("Cast", [target], { to: DataType.INT64 }, targetI64Meta);
        }

        let safeTarget: ConcreteValueNode = target64;
        let validFloat: ConcreteValueNode | undefined;
        if (hasIgnore) {
            const ignConst = builder.createConstant(
                `nll_ign_${op.id}`,
                scalarInt64(rawIgnore as number),
            );
            const zeroI64 = builder.createConstant(`nll_z64_${op.id}`, scalarInt64(0));
            const [isIgnored] = builder.createOp("Equal", [target64, ignConst], {}, targetBoolMeta);
            const [isValid] = builder.createOp("Not", [isIgnored], {}, targetBoolMeta);
            [safeTarget] = builder.createOp(
                "Where",
                [isIgnored, zeroI64, target64],
                {},
                targetI64Meta,
            );
            [validFloat] = builder.createOp("Cast", [isValid], { to: dtype }, targetFloatMeta);
        }

        // 2. Build one-hot class selector [N, C, d1, ...] along axis 1
        const depthConst = builder.createConstant(`nll_depth_${op.id}`, scalarInt64(C));
        const valuesConst = builder.createConstant(
            `nll_oh_vals_${op.id}`,
            makeTensorProto(dtype, [2], [0, 1]),
        );
        const [oneHot] = builder.createOp(
            "OneHot",
            [safeTarget, depthConst, valuesConst],
            { axis: 1 },
            inMeta,
        );

        // 3. Apply optional class weights along axis 1
        const axis1Const = builder.createConstant(`nll_ax1_${op.id}`, int64Vec([1]));
        let weightedSelector: ConcreteValueNode = oneHot;
        let sampleWeight: ConcreteValueNode | undefined;

        if (weight) {
            const wBcastShape = [1, C, ...new Array<number>(rank - 2).fill(1)];
            const wShapeConst = builder.createConstant(
                `nll_wshape_${op.id}`,
                int64Vec(wBcastShape),
            );
            const [wReshaped] = builder.createOp("Reshape", [weight, wShapeConst], {}, [
                { type: dtype, shape: wBcastShape as KnownShape },
            ]);
            [weightedSelector] = builder.createOp("Mul", [oneHot, wReshaped], {}, inMeta);
            [sampleWeight] = builder.createOp(
                "ReduceSum",
                [weightedSelector, axis1Const],
                { keepdims: 0 },
                targetFloatMeta,
            );
        }

        // 4. Per-element loss: -sum_c(input * weightedSelector)
        const [selected] = builder.createOp("Mul", [input, weightedSelector], {}, inMeta);
        const [gathered] = builder.createOp(
            "ReduceSum",
            [selected, axis1Const],
            { keepdims: 0 },
            targetFloatMeta,
        );
        let [loss] = builder.createOp("Neg", [gathered], {}, targetFloatMeta);

        if (validFloat) {
            [loss] = builder.createOp("Mul", [loss, validFloat], {}, targetFloatMeta);
            sampleWeight = sampleWeight
                ? builder.createOp("Mul", [sampleWeight, validFloat], {}, targetFloatMeta)[0]
                : validFloat;
        }

        // 5. Apply reduction ("none" | "sum" | "mean")
        const allAxes = builder.createConstant(
            `nll_all_axes_${op.id}`,
            int64Vec(targetShape.map((_, i) => i)),
        );

        let finalOut: ConcreteValueNode;
        if (reduction === "none") {
            finalOut = loss;
        } else if (reduction === "sum") {
            [finalOut] = builder.createOp(
                "ReduceSum",
                [loss, allAxes],
                { keepdims: 0 },
                scalarMeta,
            );
        } else if (sampleWeight) {
            const [totalLoss] = builder.createOp(
                "ReduceSum",
                [loss, allAxes],
                { keepdims: 0 },
                scalarMeta,
            );
            const [totalWeight] = builder.createOp(
                "ReduceSum",
                [sampleWeight, allAxes],
                { keepdims: 0 },
                scalarMeta,
            );
            [finalOut] = builder.createOp("Div", [totalLoss, totalWeight], {}, scalarMeta);
        } else {
            const [totalLoss] = builder.createOp(
                "ReduceSum",
                [loss, allAxes],
                { keepdims: 0 },
                scalarMeta,
            );
            const totalElems = targetShape.reduce((a, b) => a * b, 1);
            const invN = builder.createConstant(
                `nll_inv_n_${op.id}`,
                makeTensorProto(dtype, [], [1.0 / totalElems]),
            );
            [finalOut] = builder.createOp("Mul", [totalLoss, invN], {}, scalarMeta);
        }

        builder.replaceAllUsesWith(Y, finalOut);
        builder.removeNode(op);
    }
}
