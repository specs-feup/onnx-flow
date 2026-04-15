import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";

export const BatchNormalization: OpSchema = {
    opType: "BatchNormalization",
    sinceVersion: 9,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "scale", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY },
        { name: "mean", typeConstraint: T_ANY },
        { name: "var", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        epsilon: {
            name: "epsilon",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 1e-5,
        },
        momentum: {
            name: "momentum",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 0.9,
        },
    },

    inferShape: (inputs) => {
        // Output shape and type exactly matches the primary input (X)
        return [
            {
                shape: inputs[0]?.shape ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
};

export const Softmax: OpSchema = {
    opType: "Softmax",
    sinceVersion: 13, // Axis handling stabilized
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
    },

    inferShape: (inputs) => {
        // Output shape and type exactly match the primary input
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
};

export const LayerNormalization: OpSchema = {
    opType: "LayerNormalization",
    sinceVersion: 17,
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "Scale", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [
        { name: "Y", typeConstraint: T_ANY },
        { name: "Mean", typeConstraint: T_ANY, optional: true },
        { name: "InvStdDev", typeConstraint: T_ANY, optional: true },
    ],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
        epsilon: {
            name: "epsilon",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 1e-5,
        },
        stash_type: {
            name: "stash_type",
            type: AttributeType.INT,
            required: false,
            defaultValue: 1,
        },
    },
    inferShape: (inputs) => {
        const xShape = inputs[0]?.shape?.slice() ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        // Y shape matches X. Mean and InvStdDev will have reduced dimensions based on 'axis'.
        // For simplicity in standard ops, we pass X shape (you can expand this logic later if needed).
        return [
            { shape: xShape, dtype },
            { shape: xShape.map(() => -1), dtype }, // Mean
            { shape: xShape.map(() => -1), dtype }, // InvStdDev
        ];
    },
};

export const InstanceNormalization: OpSchema = {
    opType: "InstanceNormalization",
    sinceVersion: 6,
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "scale", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        epsilon: {
            name: "epsilon",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 1e-5,
        },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const GroupNormalization: OpSchema = {
    opType: "GroupNormalization",
    sinceVersion: 18,
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "scale", typeConstraint: T_ANY },
        { name: "bias", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        epsilon: {
            name: "epsilon",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 1e-5,
        },
        num_groups: { name: "num_groups", type: AttributeType.INT, required: true },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const LpNormalization: OpSchema = {
    opType: "LpNormalization",
    sinceVersion: 1,
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
        p: { name: "p", type: AttributeType.INT, required: false, defaultValue: 2 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const NormalizationOps: OpSchema[] = [
    BatchNormalization,
    Softmax,
    LayerNormalization,
    InstanceNormalization,
    GroupNormalization,
    LpNormalization,
];
