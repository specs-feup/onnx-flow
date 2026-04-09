import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";
import { toStaticShape, inferPoolDim } from "@specs-feup/onnx-flow/Onnx/Utils";

export const QuantizeLinear: OpSchema = {
    opType: "QuantizeLinear",
    sinceVersion: 13, // Standard definition
    category: OpCategory.Other,
    broadcastable: true,
    hasState: false,
    inputs: [
        { name: "x", typeConstraint: "T1" },
        { name: "y_scale", typeConstraint: "T1" },
        { name: "y_zero_point", typeConstraint: "T2", optional: true },
    ],
    outputs: [{ name: "y", typeConstraint: "T2" }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 1 },
        // Saturate is a newer attribute (opset 19), usually defaults to 1 (true)
        saturate: { name: "saturate", type: AttributeType.INT, required: false, defaultValue: 1 },
    },

    inferShape: (inputs) => {
        const shape = inputs[0]?.shape?.slice() ?? [];
        // If zero_point is provided, output type matches it. Otherwise, defaults to UINT8.
        const dtype = inputs[2]?.dtype ?? DataType.UINT8;
        return [{ shape, dtype }];
    },
};

export const DequantizeLinear: OpSchema = {
    opType: "DequantizeLinear",
    sinceVersion: 13,
    category: OpCategory.Other,
    broadcastable: true,
    hasState: false,
    inputs: [
        { name: "x", typeConstraint: "T1" },
        { name: "x_scale", typeConstraint: "T2" },
        { name: "x_zero_point", typeConstraint: "T1", optional: true },
    ],
    outputs: [{ name: "y", typeConstraint: "T2" }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 1 },
    },

    inferShape: (inputs) => {
        const shape = inputs[0]?.shape?.slice() ?? [];
        // Output type matches the scale type (FLOAT, FLOAT16, etc.)
        const dtype = inputs[1]?.dtype ?? DataType.FLOAT;
        return [{ shape, dtype }];
    },
};

// --- Quantized Neural Network Ops ---

export const QLinearConv: OpSchema = {
    opType: "QLinearConv",
    sinceVersion: 10,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "x", typeConstraint: T_ANY },
        { name: "x_scale", typeConstraint: T_ANY },
        { name: "x_zero_point", typeConstraint: T_ANY },
        { name: "w", typeConstraint: T_ANY },
        { name: "w_scale", typeConstraint: T_ANY },
        { name: "w_zero_point", typeConstraint: T_ANY },
        { name: "y_scale", typeConstraint: T_ANY },
        { name: "y_zero_point", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "y", typeConstraint: T_ANY }],
    attributes: {
        auto_pad: {
            name: "auto_pad",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "NOTSET",
        },
        dilations: { name: "dilations", type: AttributeType.INTS, required: false },
        group: { name: "group", type: AttributeType.INT, required: false, defaultValue: 1 },
        kernel_shape: { name: "kernel_shape", type: AttributeType.INTS, required: false },
        pads: { name: "pads", type: AttributeType.INTS, required: false },
        strides: { name: "strides", type: AttributeType.INTS, required: false },
    },
    inferShape: (inputs, attrs) => {
        // Shape inference is mathematically identical to standard Conv
        const xShape = toStaticShape(inputs[0]?.shape);
        const wShape = toStaticShape(inputs[3]?.shape); // w is at index 3
        const dtype = inputs[7]?.dtype ?? DataType.UNDEFINED; // Output type matches y_zero_point type

        if (xShape.length < 4 || wShape.length < 4) {
            return [{ shape: xShape.slice(), dtype }];
        }

        const [N, , H, W] = xShape;
        const [M, , kH, kW] = wShape;

        const strides = "strides" in attrs ? (attrs["strides"] as number[]) : [1, 1];
        const dilations = "dilations" in attrs ? (attrs["dilations"] as number[]) : [1, 1];
        const pads = "pads" in attrs ? (attrs["pads"] as number[]) : [0, 0, 0, 0];
        const autoPad = "auto_pad" in attrs ? (attrs["auto_pad"] as string) : "NOTSET";

        let padTop = pads[0] ?? 0,
            padLeft = pads[1] ?? 0,
            padBottom = pads[2] ?? 0,
            padRight = pads[3] ?? 0;

        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            const kEffH = dilations[0] * (kH - 1) + 1;
            const kEffW = dilations[1] * (kW - 1) + 1;
            const totalPadH = Math.max(0, (Math.ceil(H / strides[0]) - 1) * strides[0] + kEffH - H);
            const totalPadW = Math.max(0, (Math.ceil(W / strides[1]) - 1) * strides[1] + kEffW - W);

            if (autoPad === "SAME_UPPER") {
                padTop = Math.floor(totalPadH / 2);
                padBottom = totalPadH - padTop;
                padLeft = Math.floor(totalPadW / 2);
                padRight = totalPadW - padLeft;
            } else {
                padBottom = Math.floor(totalPadH / 2);
                padTop = totalPadH - padBottom;
                padRight = Math.floor(totalPadW / 2);
                padLeft = totalPadW - padRight;
            }
        }

        const H_out = inferPoolDim(H, kH, strides[0], padTop, padBottom, dilations[0]);
        const W_out = inferPoolDim(W, kW, strides[1], padLeft, padRight, dilations[1]);

        return [{ shape: [N, M, H_out, W_out], dtype }];
    },
};

export const QLinearMatMul: OpSchema = {
    opType: "QLinearMatMul",
    sinceVersion: 10,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "a", typeConstraint: T_ANY },
        { name: "a_scale", typeConstraint: T_ANY },
        { name: "a_zero_point", typeConstraint: T_ANY },
        { name: "b", typeConstraint: T_ANY },
        { name: "b_scale", typeConstraint: T_ANY },
        { name: "b_zero_point", typeConstraint: T_ANY },
        { name: "y_scale", typeConstraint: T_ANY },
        { name: "y_zero_point", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "y", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        // Shape inference is identical to standard MatMul
        const a = inputs[0]?.shape ?? [];
        const b = inputs[3]?.shape ?? []; // b is at index 3
        const dtype = inputs[7]?.dtype ?? DataType.UNDEFINED; // Output type matches y_zero_point

        if (a.length >= 2 && b.length >= 2) {
            return [{ shape: [a[0], b[1]], dtype }];
        }
        return [{ shape: [], dtype }];
    },
};

// --- Integer Only Ops ---

export const ConvInteger: OpSchema = {
    opType: "ConvInteger",
    sinceVersion: 10,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "x", typeConstraint: T_ANY },
        { name: "w", typeConstraint: T_ANY },
        { name: "x_zero_point", typeConstraint: T_ANY, optional: true },
        { name: "w_zero_point", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "y", typeConstraint: T_ANY }],
    attributes: {
        auto_pad: {
            name: "auto_pad",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "NOTSET",
        },
        dilations: { name: "dilations", type: AttributeType.INTS, required: false },
        group: { name: "group", type: AttributeType.INT, required: false, defaultValue: 1 },
        kernel_shape: { name: "kernel_shape", type: AttributeType.INTS, required: false },
        pads: { name: "pads", type: AttributeType.INTS, required: false },
        strides: { name: "strides", type: AttributeType.INTS, required: false },
    },
    inferShape: (inputs, attrs) => {
        // Uses the exact same inference logic as QLinearConv, but forces output to INT32
        const shapes = QLinearConv.inferShape!(inputs, attrs);
        return [{ shape: shapes[0].shape, dtype: DataType.INT32 }];
    },
};

export const MatMulInteger: OpSchema = {
    opType: "MatMulInteger",
    sinceVersion: 10,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "A", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY },
        { name: "a_zero_point", typeConstraint: T_ANY, optional: true },
        { name: "b_zero_point", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        const a = inputs[0]?.shape ?? [];
        const b = inputs[1]?.shape ?? [];
        if (a.length >= 2 && b.length >= 2) {
            // MatMulInteger strictly outputs INT32
            return [{ shape: [a[0], b[1]], dtype: DataType.INT32 }];
        }
        return [{ shape: [], dtype: DataType.INT32 }];
    },
};

// --- Dynamic Quantization & Binarization ---

export const DynamicQuantizeLinear: OpSchema = {
    opType: "DynamicQuantizeLinear",
    sinceVersion: 11,
    category: OpCategory.ElementWise,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "x", typeConstraint: T_ANY }],
    outputs: [
        { name: "y", typeConstraint: T_ANY },
        { name: "y_scale", typeConstraint: T_ANY },
        { name: "y_zero_point", typeConstraint: T_ANY },
    ],
    attributes: {},
    inferShape: (inputs) => {
        const xShape = inputs[0]?.shape?.slice() ?? [];
        return [
            { shape: xShape, dtype: DataType.UINT8 }, // y
            { shape: [], dtype: DataType.FLOAT }, // y_scale (scalar)
            { shape: [], dtype: DataType.UINT8 }, // y_zero_point (scalar)
        ];
    },
};

export const Binarizer: OpSchema = {
    opType: "Binarizer",
    sinceVersion: 1, // Actually an ai.onnx.ml op but highly common in QNNs
    category: OpCategory.ElementWise,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        threshold: {
            name: "threshold",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 0.0,
        },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const QuantizationOps: OpSchema[] = [
    QuantizeLinear,
    DequantizeLinear,
    QLinearConv,
    QLinearMatMul,
    DynamicQuantizeLinear,
    ConvInteger,
    MatMulInteger,
    Binarizer,
];
