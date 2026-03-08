import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    toStaticShape,
    getIntsAttr,
    makeTensorProto,
    getStringAttr,
    getIntAttr,
} from "../../../Utils.js";

export class LowerConvRecipe implements DecompositionRecipe {
    public readonly name = "LowerConv";
    public readonly targetOp = "Conv";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Slice",
        "Gather",
        "Mul",
        "ReduceSum",
        "ScatterElements",
        "Pad",
    ];

    canApply(op: OperationNode.Class): boolean {
        return op.type === "Conv";
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const inputs = op.getInputs() as ConcreteValueNode[];
        const X = inputs[0];
        const W = inputs[1];
        const B_bias = inputs.length > 2 ? inputs[2] : null;

        const output = op.getOutputs()[0];

        let dtype = (output.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED)
            dtype = (X.literalType as unknown as DataType | undefined) ?? DataType.UNDEFINED;
        if (dtype === DataType.UNDEFINED) dtype = DataType.FLOAT;

        const xShape = toStaticShape(X.shape);
        const wShape = toStaticShape(W.shape);
        const N = xShape[0];
        const C = xShape[1];
        const H = xShape[2];
        const Win = xShape[3];
        const M = wShape[0];

        const strides = getIntsAttr(op, "strides", [1, 1]);
        const dilations = getIntsAttr(op, "dilations", [1, 1]);
        const kernelShape = getIntsAttr(op, "kernel_shape", [wShape[2], wShape[3]]);
        let pads = getIntsAttr(op, "pads", []);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");

        const group = getIntAttr(op, "group", 1);

        const kH = kernelShape[0];
        const kW = kernelShape[1];
        const sH = strides[0];
        const sW = strides[1];
        const dH = dilations[0];
        const dW = dilations[1];

        const kEffH = dH * (kH - 1) + 1;
        const kEffW = dW * (kW - 1) + 1;

        if (pads.length === 0) {
            if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
                const isLower = autoPad === "SAME_LOWER";
                const outH_temp = Math.ceil(H / sH);
                const outW_temp = Math.ceil(Win / sW);
                const padH = Math.max((outH_temp - 1) * sH + kEffH - H, 0);
                const padW = Math.max((outW_temp - 1) * sW + kEffW - Win, 0);

                const pT = isLower ? Math.ceil(padH / 2) : Math.floor(padH / 2);
                const pB = padH - pT;
                const pL = isLower ? Math.ceil(padW / 2) : Math.floor(padW / 2);
                const pR = padW - pL;
                pads = [pT, pL, pB, pR];
            } else {
                pads = [0, 0, 0, 0];
            }
        }

        const pT = pads[0] ?? 0,
            pL = pads[1] ?? 0,
            pB = pads[2] ?? 0,
            pR = pads[3] ?? 0;

        const H_padded = H + pT + pB;
        const W_padded = Win + pL + pR;

        const M_out = M;
        const H_out = Math.floor((H_padded - kEffH) / sH + 1);
        const W_out = Math.floor((W_padded - kEffW) / sW + 1);

        const outShape = [N, M_out, H_out, W_out];

        // Pad X if necessary
        let paddedX = X;
        if (pads.some((p) => p !== 0)) {
            const padVec = [0, 0, pT, pL, 0, 0, pB, pR];
            const padsConst = builder.createConstant(
                `pads_${op.id}`,
                makeTensorProto(DataType.INT64, [8], padVec),
            );
            paddedX = builder.createOp("Pad", [X, padsConst])[0];
        }

        const totalElements = outShape.reduce((a, b) => a * b, 1);
        const shapeConst = builder.createConstant(
            `shape_${op.id}`,
            makeTensorProto(DataType.INT64, [outShape.length], outShape),
        );

        const { innerBuilder, trip, vInitial, loopOutput, finalize } = builder.createForLoopRegion(
            builder,
            totalElements,
            dtype,
            [totalElements],
            `ConvLoop_${op.id}`,
        );

        const M_HW = innerBuilder.createConstant(
            `M_HW_${op.id}`,
            makeTensorProto(DataType.INT64, [], [M_out * H_out * W_out]),
        );
        const HW = innerBuilder.createConstant(
            `HW_${op.id}`,
            makeTensorProto(DataType.INT64, [], [H_out * W_out]),
        );
        const W_const = innerBuilder.createConstant(
            `Wout_${op.id}`,
            makeTensorProto(DataType.INT64, [], [W_out]),
        );

        const nIdx = innerBuilder.createOp("Div", [trip, M_HW])[0];
        const remN = innerBuilder.createOp("Mod", [trip, M_HW])[0];
        const mIdx = innerBuilder.createOp("Div", [remN, HW])[0];
        const rem = innerBuilder.createOp("Mod", [remN, HW])[0];
        const yIdx = innerBuilder.createOp("Div", [rem, W_const])[0];
        const xIdx = innerBuilder.createOp("Mod", [rem, W_const])[0];

        const strideYConst = innerBuilder.createConstant(
            `sY_${op.id}`,
            makeTensorProto(DataType.INT64, [], [sH]),
        );
        const strideXConst = innerBuilder.createConstant(
            `sX_${op.id}`,
            makeTensorProto(DataType.INT64, [], [sW]),
        );
        const kEffHConst = innerBuilder.createConstant(
            `kEffH_${op.id}`,
            makeTensorProto(DataType.INT64, [], [kEffH]),
        );
        const kEffWConst = innerBuilder.createConstant(
            `kEffW_${op.id}`,
            makeTensorProto(DataType.INT64, [], [kEffW]),
        );

        const yStart = innerBuilder.createOp("Mul", [yIdx, strideYConst])[0];
        const xStart = innerBuilder.createOp("Mul", [xIdx, strideXConst])[0];
        const yEnd = innerBuilder.createOp("Add", [yStart, kEffHConst])[0]; // Use Effective kernel sizes for Ends
        const xEnd = innerBuilder.createOp("Add", [xStart, kEffWConst])[0];

        const flatAxes = innerBuilder.createConstant(
            `flatAxes_${op.id}`,
            makeTensorProto(DataType.INT64, [1], [0]),
        );
        const oneConst = innerBuilder.createConstant(
            `one_${op.id}`,
            makeTensorProto(DataType.INT64, [], [1]),
        );
        const nEnd = innerBuilder.createOp("Add", [nIdx, oneConst])[0];

        let starts: ConcreteValueNode;
        let ends: ConcreteValueNode;
        let sliceAxes: ConcreteValueNode;
        let sliceSteps: ConcreteValueNode;

        if (group > 1) {
            // Group > 1 requires us to slice C to extract only the channels belonging to the current output channel (mIdx)
            const C_per_group = Math.floor(C / group);
            const M_per_group = Math.floor(M / group);

            const MperGroupConst = innerBuilder.createConstant(
                `MperG_${op.id}`,
                makeTensorProto(DataType.INT64, [], [M_per_group]),
            );
            const CperGroupConst = innerBuilder.createConstant(
                `CperG_${op.id}`,
                makeTensorProto(DataType.INT64, [], [C_per_group]),
            );

            const gIdx = innerBuilder.createOp("Div", [mIdx, MperGroupConst])[0];
            const cStart = innerBuilder.createOp("Mul", [gIdx, CperGroupConst])[0];
            const cEnd = innerBuilder.createOp("Add", [cStart, CperGroupConst])[0];

            starts = innerBuilder.createOp(
                "Concat",
                [
                    innerBuilder.createOp("Unsqueeze", [nIdx, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [cStart, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [yStart, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [xStart, flatAxes])[0],
                ],
                { axis: 0 },
            )[0];

            ends = innerBuilder.createOp(
                "Concat",
                [
                    innerBuilder.createOp("Unsqueeze", [nEnd, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [cEnd, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [yEnd, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [xEnd, flatAxes])[0],
                ],
                { axis: 0 },
            )[0];

            sliceAxes = innerBuilder.createConstant(
                `sliceAxes_${op.id}`,
                makeTensorProto(DataType.INT64, [4], [0, 1, 2, 3]),
            );
            sliceSteps = innerBuilder.createConstant(
                `sliceSteps_${op.id}`,
                makeTensorProto(DataType.INT64, [4], [1, 1, dH, dW]),
            );
        } else {
            starts = innerBuilder.createOp(
                "Concat",
                [
                    innerBuilder.createOp("Unsqueeze", [nIdx, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [yStart, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [xStart, flatAxes])[0],
                ],
                { axis: 0 },
            )[0];

            ends = innerBuilder.createOp(
                "Concat",
                [
                    innerBuilder.createOp("Unsqueeze", [nEnd, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [yEnd, flatAxes])[0],
                    innerBuilder.createOp("Unsqueeze", [xEnd, flatAxes])[0],
                ],
                { axis: 0 },
            )[0];

            sliceAxes = innerBuilder.createConstant(
                `sliceAxes_${op.id}`,
                makeTensorProto(DataType.INT64, [3], [0, 2, 3]),
            );
            // Apply Dilations naturally via Slice steps
            sliceSteps = innerBuilder.createConstant(
                `sliceSteps_${op.id}`,
                makeTensorProto(DataType.INT64, [3], [1, dH, dW]),
            );
        }

        const patch = innerBuilder.createOp("Slice", [
            paddedX,
            starts,
            ends,
            sliceAxes,
            sliceSteps,
        ])[0];

        const mUnsq = innerBuilder.createOp("Unsqueeze", [mIdx, flatAxes])[0];
        const kernel = innerBuilder.createOp("Gather", [W, mUnsq], { axis: 0 })[0];

        const mul = innerBuilder.createOp("Mul", [patch, kernel])[0];
        const sumScalar = innerBuilder.createOp("ReduceSum", [mul], { keepdims: 0 })[0];

        let finalScalar = sumScalar;
        if (B_bias) {
            let biasSq: ConcreteValueNode;
            const bShape = toStaticShape(B_bias.shape);
            if (bShape.length === 1 && bShape[0] === M) {
                const biasVal = innerBuilder.createOp("Gather", [B_bias, mUnsq], { axis: 0 })[0];
                biasSq = innerBuilder.createOp("Squeeze", [biasVal, flatAxes])[0];
            } else if (bShape.length === 4 && bShape[1] === M) {
                // [1, M, 1, 1]
                const biasVal = innerBuilder.createOp("Gather", [B_bias, mUnsq], { axis: 1 })[0];
                const squeezeAxes = innerBuilder.createConstant(
                    `sqAxes_${op.id}`,
                    makeTensorProto(DataType.INT64, [3], [0, 1, 2]),
                );
                biasSq = innerBuilder.createOp("Squeeze", [biasVal, squeezeAxes])[0];
            } else {
                const zeroIdx = innerBuilder.createConstant(
                    `zero_${op.id}`,
                    makeTensorProto(DataType.INT64, [1], [0]),
                );
                const biasVal = innerBuilder.createOp("Gather", [B_bias, zeroIdx], { axis: 0 })[0];
                biasSq = innerBuilder.createOp("Squeeze", [biasVal, flatAxes])[0];
            }
            finalScalar = innerBuilder.createOp("Add", [sumScalar, biasSq])[0];
        }

        const iterUnsq = innerBuilder.createOp("Unsqueeze", [trip, flatAxes])[0];
        const updateVal = innerBuilder.createOp("Unsqueeze", [finalScalar, flatAxes])[0];
        const scatterOut = innerBuilder.createOp(
            "ScatterElements",
            [vInitial, iterUnsq, updateVal],
            { axis: 0 },
        )[0];

        finalize([scatterOut]);

        const finalReshape = builder.createOp("Reshape", [loopOutput, shapeConst], {}, [
            { type: dtype, shape: outShape },
        ])[0];
        builder.replaceAllUsesWith(output, finalReshape);
        op.remove();
    }
}
