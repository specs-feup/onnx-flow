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

export class LowerAveragePoolRecipe implements DecompositionRecipe {
    public readonly name = "LowerAveragePool";
    public readonly targetOp = "AveragePool";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly producedOps = ["Conv", "Div", "Shape", "Expand"];

    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "AveragePool") return false;
        if (op.getInputs() === undefined) return false;
        const X = op.getInputs()![0] as ConcreteValueNode;
        const xShape = toStaticShape(X.shape);

        const kernelShape = getIntsAttr(op, "kernel_shape", []);
        if (kernelShape.length !== 2) return false;

        if (xShape.length !== 4 || xShape[1] <= 0) return false;

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
            return false; // Skip rewriting, Orchestrator will leave it for TransformChain
        }

        return true;
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
        const pads = padsAttr.length === 4 ? padsAttr.map(Number) : [0, 0, 0, 0];
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const countIncludePad = getIntAttr(op, "count_include_pad", 0);

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

        // 2. Compute Sum = Conv(X, Wones)
        const sumOut = builder.createOp("Conv", [X, wOnes], convAttrs)[0];

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
            const shapeX = builder.createOp("Shape", [X])[0];

            const oneSc = builder.createConstant(
                `AvgPool_One_${op.id}`,
                makeTensorProto(dtype, [], [1]),
            );

            const mask = builder.createOp("Expand", [oneSc, shapeX])[0];

            divisor = builder.createOp("Conv", [mask, wOnes], convAttrs)[0];
        }

        // 4. Final Divide
        const finalY = builder.createOp("Div", [sumOut, divisor])[0];

        builder.replaceAllUsesWith(Y, finalY);
        op.remove();
    }
}
