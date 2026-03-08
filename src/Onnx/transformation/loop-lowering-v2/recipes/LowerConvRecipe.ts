import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asStaticDims,
    getIntsAttr,
    getStringAttr,
    getIntAttr,
    scalarInt64,
    int64Vec,
    uniq,
} from "../../../Utils.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput, squeezeIfLen1 } from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerConvRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        return op.type === "Conv";
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `conv_${op.id}`);
        const inputs = op.getInputs()!;
        const dtype = (op.getOutputs()[0].literalType as DataType) ?? DataType.FLOAT;

        // 1. Resolve inputs (Captured Tensors)
        const X = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
        );
        const W = resolveRecipeInput(
            builder,
            inputs[1],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
        );
        const B_bias =
            inputs.length > 2
                ? resolveRecipeInput(
                      builder,
                      inputs[2],
                      valueMap,
                      iter,
                      axes,
                      outShape,
                      false,
                      false,
                  )
                : null;

        const xShape = asStaticDims(inputs[0].shape);
        const wShape = asStaticDims(inputs[1].shape);

        const C = xShape[1],
            H = xShape[2],
            Win = xShape[3];
        const M = wShape[0];

        const strides = getIntsAttr(op, "strides", [1, 1]);
        const dilations = getIntsAttr(op, "dilations", [1, 1]);
        const kernelShape = getIntsAttr(op, "kernel_shape", [wShape[2], wShape[3]]);
        let pads = getIntsAttr(op, "pads", []);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const group = getIntAttr(op, "group", 1);

        const kH = kernelShape[0],
            kW = kernelShape[1];
        const sH = strides[0],
            sW = strides[1];
        const dH = dilations[0],
            dW = dilations[1];

        const kEffH = dH * (kH - 1) + 1;
        const kEffW = dW * (kW - 1) + 1;

        // 2. Handle Padding logic
        if (pads.length === 0) {
            if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
                const isLower = autoPad === "SAME_LOWER";
                const padH = Math.max((Math.ceil(H / sH) - 1) * sH + kEffH - H, 0);
                const padW = Math.max((Math.ceil(Win / sW) - 1) * sW + kEffW - Win, 0);
                const pT = isLower ? Math.ceil(padH / 2) : Math.floor(padH / 2);
                const pL = isLower ? Math.ceil(padW / 2) : Math.floor(padW / 2);
                pads = [pT, pL, padH - pT, padW - pL];
            } else {
                pads = [0, 0, 0, 0];
            }
        }

        const pT = pads[0] ?? 0,
            pL = pads[1] ?? 0,
            pB = pads[2] ?? 0,
            pR = pads[3] ?? 0;
        const H_out = Math.floor((H + pT + pB - kEffH) / sH + 1);
        const W_out = Math.floor((Win + pL + pR - kEffW) / sW + 1);

        let paddedX = X;
        if (pads.some((p) => p !== 0)) {
            const padsConst = builder.createConstant(`pads`, {
                dataType: DataType.INT64,
                dims: [8],
                int64Data: [0, 0, BigInt(pT), BigInt(pL), 0, 0, BigInt(pB), BigInt(pR)],
            });
            [paddedX] = builder.createOp("Pad", [X, padsConst]);
        }

        // 3. Decode iteration index into N, M, Y, X coordinates
        const [nIdx] = builder.createOp("Div", [
            iter,
            builder.createConstant(`M_HW`, scalarInt64(M * H_out * W_out)),
        ]);
        const [remN] = builder.createOp("Mod", [
            iter,
            builder.createConstant(`M_HW_rem`, scalarInt64(M * H_out * W_out)),
        ]);
        const [mIdx] = builder.createOp("Div", [
            remN,
            builder.createConstant(`HW`, scalarInt64(H_out * W_out)),
        ]);
        const [rem] = builder.createOp("Mod", [
            remN,
            builder.createConstant(`HW_rem`, scalarInt64(H_out * W_out)),
        ]);
        const [yIdx] = builder.createOp("Div", [
            rem,
            builder.createConstant(`Wout`, scalarInt64(W_out)),
        ]);
        const [xIdx] = builder.createOp("Mod", [
            rem,
            builder.createConstant(`Wout_rem`, scalarInt64(W_out)),
        ]);

        // 4. Calculate Slice bounds for the input patch
        const [yStart] = builder.createOp("Mul", [
            yIdx,
            builder.createConstant(`sH`, scalarInt64(sH)),
        ]);
        const [xStart] = builder.createOp("Mul", [
            xIdx,
            builder.createConstant(`sW`, scalarInt64(sW)),
        ]);
        const [yEnd] = builder.createOp("Add", [
            yStart,
            builder.createConstant(`keH`, scalarInt64(kEffH)),
        ]);
        const [xEnd] = builder.createOp("Add", [
            xStart,
            builder.createConstant(`keW`, scalarInt64(kEffW)),
        ]);
        const [nEnd] = builder.createOp("Add", [
            nIdx,
            builder.createConstant(`one`, scalarInt64(1)),
        ]);

        let starts: ValueNode, ends: ValueNode, sliceAxes: ValueNode, sliceSteps: ValueNode;

        if (group > 1) {
            const MperG = Math.floor(M / group),
                CperG = Math.floor(C / group);
            const [gIdx] = builder.createOp("Div", [
                mIdx,
                builder.createConstant(`MperG`, scalarInt64(MperG)),
            ]);
            const [cStart] = builder.createOp("Mul", [
                gIdx,
                builder.createConstant(`CperG`, scalarInt64(CperG)),
            ]);
            const [cEnd] = builder.createOp("Add", [
                cStart,
                builder.createConstant(`CperG_val`, scalarInt64(CperG)),
            ]);

            const coordsS = [nIdx, cStart, yStart, xStart].map(
                (c, i) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [starts] = builder.createOp("Concat", coordsS, { axis: 0 });

            const coordsE = [nEnd, cEnd, yEnd, xEnd].map(
                (c, i) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [ends] = builder.createOp("Concat", coordsE, { axis: 0 });

            sliceAxes = builder.createConstant(`sliceAxes`, int64Vec([0, 1, 2, 3]));
            sliceSteps = builder.createConstant(`sliceSteps`, int64Vec([1, 1, dH, dW]));
        } else {
            const coordsS = [nIdx, yStart, xStart].map(
                (c, i) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [starts] = builder.createOp("Concat", coordsS, { axis: 0 });

            const coordsE = [nEnd, yEnd, xEnd].map(
                (c, i) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [ends] = builder.createOp("Concat", coordsE, { axis: 0 });

            sliceAxes = builder.createConstant(`sliceAxes`, int64Vec([0, 2, 3]));
            sliceSteps = builder.createConstant(`sliceSteps`, int64Vec([1, dH, dW]));
        }

        // 5. Extract patch, gather kernel, and perform dot product
        const [patch] = builder.createOp("Slice", [paddedX, starts, ends, sliceAxes, sliceSteps]);
        const [mIdxUnsq] = builder.createOp("Unsqueeze", [mIdx, axes]);
        const [kernel] = builder.createOp("Gather", [W, mIdxUnsq], { axis: 0 });

        const [mulOut] = builder.createOp("Mul", [patch, kernel]);
        const [sumOut] = builder.createOp("ReduceSum", [mulOut], { keepdims: 0 });

        // 6. Final bias Addition
        let finalScalar = sumOut;
        if (B_bias) {
            const [gBiasOut] = builder.createOp("Gather", [B_bias, mIdxUnsq], { axis: 0 });
            const biasSq = squeezeIfLen1(builder, gBiasOut, axes, `sqBias`);
            [finalScalar] = builder.createOp("Add", [sumOut, biasSq]);
        }

        return finalScalar;
    }
}
