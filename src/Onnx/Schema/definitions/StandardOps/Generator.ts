import type { TensorProto } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { DataType, AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";

export const Range: OpSchema = {
    opType: "Range",
    sinceVersion: 11,
    category: OpCategory.Generator,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "start", typeConstraint: T_ANY },
        { name: "limit", typeConstraint: T_ANY },
        { name: "delta", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},

    inferShape: (inputs) => {
        const start = inputs[0]?.constantValue?.[0];
        const end = inputs[1]?.constantValue?.[0];
        const step = inputs[2]?.constantValue?.[0];
        const dtype = inputs[0]?.dtype ?? DataType.FLOAT;

        if (start !== undefined && end !== undefined && step !== undefined && step !== 0) {
            const len = Math.max(0, Math.ceil((end - start) / step));
            return [{ shape: [len], dtype }];
        }
        // Unknown 1D tensor fallback
        return [{ shape: [-1], dtype }];
    },
};

export const ConstantOfShape: OpSchema = {
    opType: "ConstantOfShape",
    sinceVersion: 9,
    category: OpCategory.Generator,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_INT, optional: true }], // The shape
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        value: { name: "value", type: AttributeType.TENSOR, required: false }, // defaults to float 0.0
    },

    inferShape: (inputs, attrs) => {
        let shape = inputs[0]?.constantValue ?? [];

        // If dynamic (not provided via constantValue), output shape is unknown 1D tensor
        if (
            shape.length === 0 &&
            inputs[0]?.shape?.length &&
            (inputs[0].shape as number[])[0] > 0
        ) {
            shape = Array((inputs[0].shape as number[])[0]).fill(-1);
        }

        // If the 'value' tensor attribute is provided, use its dtype. Otherwise FLOAT.
        const valAttr = attrs["value"];
        let dtype = DataType.FLOAT;
        if (valAttr && typeof valAttr === "object" && "dataType" in valAttr) {
            dtype = valAttr.dataType as DataType;
        }

        return [{ shape, dtype }];
    },
};

export const Constant: OpSchema = {
    opType: "Constant",
    sinceVersion: 13,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        value: { name: "value", type: AttributeType.TENSOR, required: false },
        value_float: { name: "value_float", type: AttributeType.FLOAT, required: false },
        value_floats: { name: "value_floats", type: AttributeType.FLOATS, required: false },
        value_int: { name: "value_int", type: AttributeType.INT, required: false },
        value_ints: { name: "value_ints", type: AttributeType.INTS, required: false },
        value_string: { name: "value_string", type: AttributeType.STRING, required: false },
        value_strings: { name: "value_strings", type: AttributeType.STRINGS, required: false },
    },
    inferShape: (_, attrs) => {
        // 1. Check if it's a tensor attribute
        if (attrs["value"]) {
            const tensor = attrs["value"] as TensorProto;
            return [{ shape: tensor.dims ?? [], dtype: tensor.dataType ?? DataType.UNDEFINED }];
        }
        // 2. Check for scalars
        if ("value_float" in attrs) return [{ shape: [], dtype: DataType.FLOAT }];
        if ("value_int" in attrs) return [{ shape: [], dtype: DataType.INT64 }];
        if ("value_string" in attrs) return [{ shape: [], dtype: DataType.STRING }];
        // 3. Check for 1D arrays
        if ("value_floats" in attrs)
            return [{ shape: [(attrs["value_floats"] as number[]).length], dtype: DataType.FLOAT }];
        if ("value_ints" in attrs)
            return [{ shape: [(attrs["value_ints"] as number[]).length], dtype: DataType.INT64 }];
        if ("value_strings" in attrs)
            return [
                { shape: [(attrs["value_strings"] as string[]).length], dtype: DataType.STRING },
            ];

        return [{ shape: [], dtype: DataType.UNDEFINED }];
    },
};

export const EyeLike: OpSchema = {
    opType: "EyeLike",
    sinceVersion: 9,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        dtype: { name: "dtype", type: AttributeType.INT, required: false },
        k: { name: "k", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape?.slice() ?? [];
        const dtype = (attrs["dtype"] as number) ?? inputs[0]?.dtype ?? DataType.UNDEFINED;
        return [{ shape: inputShape, dtype }];
    },
};

export const RandomNormal: OpSchema = {
    opType: "RandomNormal",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        dtype: {
            name: "dtype",
            type: AttributeType.INT,
            required: false,
            defaultValue: DataType.FLOAT,
        },
        mean: { name: "mean", type: AttributeType.FLOAT, required: false, defaultValue: 0.0 },
        scale: { name: "scale", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        seed: { name: "seed", type: AttributeType.FLOAT, required: false },
        shape: { name: "shape", type: AttributeType.INTS, required: true },
    },
    inferShape: (_, attrs) => {
        const shape = (attrs["shape"] as number[]) ?? [];
        const dtype = (attrs["dtype"] as number) ?? DataType.FLOAT;
        return [{ shape, dtype }];
    },
};

export const RandomNormalLike: OpSchema = {
    opType: "RandomNormalLike",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        dtype: { name: "dtype", type: AttributeType.INT, required: false },
        mean: { name: "mean", type: AttributeType.FLOAT, required: false, defaultValue: 0.0 },
        scale: { name: "scale", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        seed: { name: "seed", type: AttributeType.FLOAT, required: false },
    },
    inferShape: (inputs, attrs) => {
        const shape = inputs[0]?.shape?.slice() ?? [];
        const dtype = (attrs["dtype"] as number) ?? inputs[0]?.dtype ?? DataType.FLOAT;
        return [{ shape, dtype }];
    },
};

export const RandomUniform: OpSchema = {
    opType: "RandomUniform",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        dtype: {
            name: "dtype",
            type: AttributeType.INT,
            required: false,
            defaultValue: DataType.FLOAT,
        },
        high: { name: "high", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        low: { name: "low", type: AttributeType.FLOAT, required: false, defaultValue: 0.0 },
        seed: { name: "seed", type: AttributeType.FLOAT, required: false },
        shape: { name: "shape", type: AttributeType.INTS, required: true },
    },
    inferShape: (_, attrs) => {
        const shape = (attrs["shape"] as number[]) ?? [];
        const dtype = (attrs["dtype"] as number) ?? DataType.FLOAT;
        return [{ shape, dtype }];
    },
};

export const RandomUniformLike: OpSchema = {
    opType: "RandomUniformLike",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        dtype: { name: "dtype", type: AttributeType.INT, required: false },
        high: { name: "high", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        low: { name: "low", type: AttributeType.FLOAT, required: false, defaultValue: 0.0 },
        seed: { name: "seed", type: AttributeType.FLOAT, required: false },
    },
    inferShape: (inputs, attrs) => {
        const shape = inputs[0]?.shape?.slice() ?? [];
        const dtype = (attrs["dtype"] as number) ?? inputs[0]?.dtype ?? DataType.FLOAT;
        return [{ shape, dtype }];
    },
};

export const Multinomial: OpSchema = {
    opType: "Multinomial",
    sinceVersion: 7,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: "T2" }],
    attributes: {
        dtype: {
            name: "dtype",
            type: AttributeType.INT,
            required: false,
            defaultValue: DataType.INT32,
        },
        sample_size: {
            name: "sample_size",
            type: AttributeType.INT,
            required: false,
            defaultValue: 1,
        },
        seed: { name: "seed", type: AttributeType.FLOAT, required: false },
    },
    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const sampleSize = (attrs["sample_size"] as number) ?? 1;
        const dtype = (attrs["dtype"] as number) ?? DataType.INT32;
        // Output shape is [batch_size, sample_size]
        const batchSize = inputShape.length > 0 ? inputShape[0] : -1;
        return [{ shape: [batchSize, sampleSize], dtype }];
    },
};

export const GeneratorOps: OpSchema[] = [
    Range,
    ConstantOfShape,
    Constant,
    EyeLike,
    RandomNormal,
    RandomNormalLike,
    RandomUniform,
    RandomUniformLike,
    Multinomial,
];
