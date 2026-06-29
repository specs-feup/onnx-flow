import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { ConcreteValueNode, AttributeValue } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    getIntsAttr,
    getStringAttr,
    makeTensorProto,
    toStaticShape,
} from "../../../Utils.js";
import { TransformationOpportunity } from "../../TransformationOpportunity.js";

export class LowerAveragePoolRecipe implements DecompositionRecipe {
    public readonly name = "LowerAveragePool";
    public readonly targetOp = "AveragePool";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Conv", "Div", "Shape", "Expand"];

    match(op: OperationNode.Class): TransformationOpportunity | null {
        if (op.type !== "AveragePool") return null;
        if (op.getInputs() === undefined) return null;
        const X = op.getInputs()![0] as ConcreteValueNode;
        const xShape = toStaticShape(X.shape);

        const kernelShape = getIntsAttr(op, "kernel_shape", []);
        if (kernelShape.length !== 2) return null;

        if (xShape.length !== 4 || xShape[1] <= 0) return null;

        // Hoist the optimization check here!
        const [kH, kW] = kernelShape;
        const strides = getIntsAttr(op, "strides", [1, 1]);
        const padsAttr = getIntsAttr(op, "pads", [0, 0, 0, 0]);
        const pads = padsAttr.length === 4 ? padsAttr.map(Number) : [0, 0, 0, 0];
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const ceilMode = getIntAttr(op, "ceil_mode", 0);

        if (
            autoPad === "NOTSET" &&
            ceilMode === 0 &&
            pads.every((p) => p === 0) &&
            kH === strides[0] &&
            kW === strides[1]
        ) {
            return null;
        }

        return new TransformationOpportunity(
            this.name,
            op.id,
            "Lower AveragePool to Conv",
            (builder: GraphBuilder) => this.apply(op, builder),
        );
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const X = op.getInputs()![0] as ConcreteValueNode;
        const Y = op.getOutputs()[0];

        const dtype = (X.literalType as DataType | undefined) ?? DataType.FLOAT;
        const xShape = toStaticShape(X.shape);
        const C = xShape[1];

        const kernelShape = getIntsAttr(op, "kernel_shape", []);
        const [kH, kW] = kernelShape;
        const strides = getIntsAttr(op, "strides", [1, 1]);
        const padsAttr = getIntsAttr(op, "pads", [0, 0, 0, 0]);
        let pads = padsAttr.length === 4 ? padsAttr.map(Number) : [0, 0, 0, 0];
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const countIncludePad = getIntAttr(op, "count_include_pad", 0);
        const ceilMode = getIntAttr(op, "ceil_mode", 0);

        // --- Handle ceil_mode dynamically by augmenting right/bottom padding ---
        if (ceilMode === 1 && autoPad === "NOTSET") {
            const H = xShape[2];
            const W = xShape[3];

            if (typeof H === "number" && typeof W === "number") {
                const pT = pads[0];
                const pL = pads[1];
                let pB = pads[2];
                let pR = pads[3];
                const [sH, sW] = strides.map(Number);

                // Calculate if an extra step is required due to ceiling
                const outHFloor = Math.floor((H + pT + pB - kH) / sH) + 1;
                const outHCeil = Math.ceil((H + pT + pB - kH) / sH) + 1;
                if (outHCeil > outHFloor) {
                    pB += (outHCeil - outHFloor) * sH; // Inject enough padding to push floor to ceil
                }

                const outWFloor = Math.floor((W + pL + pR - kW) / sW) + 1;
                const outWCeil = Math.ceil((W + pL + pR - kW) / sW) + 1;
                if (outWCeil > outWFloor) {
                    pR += (outWCeil - outWFloor) * sW;
                }

                pads = [pT, pL, pB, pR];
            }
        }

        // 1. Create Weight Tensor (Ones) -> [C, 1, kH, kW]
        const wElements = C * kH * kW;
        const wData = new Array(wElements).fill(1);
        const wOnes = builder.createConstant(
            `AvgPool_W_${op.id}`,
            makeTensorProto(dtype, [C, 1, kH, kW], wData),
        );

        const convAttrs: Record<string, AttributeValue> = {
            group: C,
            strides: strides,
        };
        if (autoPad !== "NOTSET") convAttrs["auto_pad"] = autoPad;
        else convAttrs["pads"] = pads;

        const XShapeStatic = toStaticShape(X.shape);
        const YShapeStatic = toStaticShape(Y.shape);

        const expectedX = [{ type: dtype, shape: XShapeStatic }];
        const expectedY = [{ type: dtype, shape: YShapeStatic }];

        // 2. Compute Sum = Conv(X, Wones)
        const sumOut = builder.createOp("Conv", [X, wOnes], convAttrs, expectedY)[0];

        // 3. Compute Divisor
        let divisor: ConcreteValueNode;
        if (countIncludePad === 1 || autoPad === "VALID") {
            // Simple case: Divide by constant kernel area
            divisor = builder.createConstant(
                `AvgPool_Div_${op.id}`,
                makeTensorProto(dtype, [], [kH * kW]),
            );
        } else {
            // Complex case: Convolve a mask of 1s to count valid pixels
            const xShapeStatic = toStaticShape(X.shape);
            let shapeX: ConcreteValueNode;

            // Generate a static shape constant if possible to preserve shape inference downstream
            if (xShapeStatic.length > 0 && xShapeStatic.every((d) => d > 0)) {
                shapeX = builder.createConstant(
                    `AvgPool_shapeX_${op.id}`,
                    makeTensorProto(DataType.INT64, [xShapeStatic.length], xShapeStatic),
                );
            } else {
                shapeX = builder.createOp("Shape", [X])[0];
            }

            const oneSc = builder.createConstant(
                `AvgPool_One_${op.id}`,
                makeTensorProto(dtype, [], [1]),
            );

            // Force mask to inherit the exact shape of X
            const mask = builder.createOp("Expand", [oneSc, shapeX], {}, expectedX)[0];

            // Force divisor to inherit the exact shape of Y
            divisor = builder.createOp("Conv", [mask, wOnes], convAttrs, expectedY)[0];
        }

        // 4. Final Divide (Y shape)
        const finalY = builder.createOp("Div", [sumOut, divisor], {}, expectedY)[0];

        builder.replaceAllUsesWith(Y, finalY);
        op.remove();
    }
}
