import { AttributeType, DataType } from "../../../ir/OnnxTypes.js";
import { toStaticShape, inferPoolDim, resolveAutoPads } from "../../../utils/index.js";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";

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

    inferShape: (inputs, attrs) => {
        const shape = inputs[0]?.shape?.slice() ?? [];
        const attrOutType =
            "output_dtype" in attrs && Number(attrs["output_dtype"]) > 0
                ? Number(attrs["output_dtype"])
                : undefined;
        // If output_dtype attr or zero_point is provided, use it; otherwise UINT8
        const dtype = attrOutType ?? (inputs.length > 2 ? inputs[2].dtype : DataType.UINT8);
        return [{ shape, dtype }];
    },
};

export const QuantizeLinear21: OpSchema = {
    ...QuantizeLinear,
    sinceVersion: 21,
    attributes: {
        ...QuantizeLinear.attributes,
        block_size: {
            name: "block_size",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        output_dtype: {
            name: "output_dtype",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        precision: { name: "precision", type: AttributeType.INT, required: false, defaultValue: 0 },
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

    inferShape: (inputs, attrs) => {
        const shape = inputs[0]?.shape?.slice() ?? [];
        const attrOutType =
            "output_dtype" in attrs && Number(attrs["output_dtype"]) > 0
                ? Number(attrs["output_dtype"])
                : undefined;
        // Output type matches output_dtype (if set) or scale type (FLOAT, FLOAT16, BFLOAT16)
        const dtype = attrOutType ?? (inputs.length > 1 ? inputs[1].dtype : DataType.FLOAT);
        return [{ shape, dtype }];
    },
};

export const DequantizeLinear21: OpSchema = {
    ...DequantizeLinear,
    sinceVersion: 21,
    attributes: {
        ...DequantizeLinear.attributes,
        block_size: {
            name: "block_size",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        output_dtype: {
            name: "output_dtype",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
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

        const [padTop, padLeft, padBottom, padRight] = resolveAutoPads(
            autoPad,
            [H, W],
            [kH, kW],
            strides,
            dilations,
            pads,
        );

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
    category: OpCategory.Other,
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
    QuantizeLinear21,
    DequantizeLinear,
    DequantizeLinear21,
    QLinearConv,
    QLinearMatMul,
    DynamicQuantizeLinear,
    ConvInteger,
    MatMulInteger,
    Binarizer,
];
