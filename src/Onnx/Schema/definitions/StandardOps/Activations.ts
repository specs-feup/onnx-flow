// --- Activations ---

import { DataType, AttributeType } from "../../../OnnxTypes.js";
import { toStaticShape, broadcastShapes } from "../../../Utils.js";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";

export const PRelu: OpSchema = {
    opType: "PRelu",
    sinceVersion: 16,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "slope", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        const xShape = toStaticShape(inputs[0]?.shape);
        const slopeShape = toStaticShape(inputs[1]?.shape);
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        return [{ shape: broadcastShapes(xShape, slopeShape), dtype }];
    },
};

export const ThresholdedRelu: OpSchema = {
    opType: "ThresholdedRelu",
    sinceVersion: 10,
    category: OpCategory.ElementWise,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        alpha: { name: "alpha", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const Shrink: OpSchema = {
    opType: "Shrink",
    sinceVersion: 9,
    category: OpCategory.ElementWise,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        bias: { name: "bias", type: AttributeType.FLOAT, required: false, defaultValue: 0.0 },
        lambd: { name: "lambd", type: AttributeType.FLOAT, required: false, defaultValue: 0.5 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const LogSoftmax: OpSchema = {
    opType: "LogSoftmax",
    sinceVersion: 13,
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const Hardmax: OpSchema = {
    opType: "Hardmax",
    sinceVersion: 13,
    category: OpCategory.Normalization,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

// --- Loss Functions ---

export const SoftmaxCrossEntropyLoss: OpSchema = {
    opType: "SoftmaxCrossEntropyLoss",
    sinceVersion: 13,
    category: OpCategory.Other,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "scores", typeConstraint: T_ANY },
        { name: "labels", typeConstraint: T_ANY },
        { name: "weights", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [
        { name: "output", typeConstraint: T_ANY },
        { name: "log_prob", typeConstraint: T_ANY, optional: true },
    ],
    attributes: {
        reduction: {
            name: "reduction",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "mean",
        },
        ignore_index: { name: "ignore_index", type: AttributeType.INT, required: false },
    },
    inferShape: (inputs, attrs) => {
        const reduction = "reduction" in attrs ? (attrs["reduction"] as string) : "mean";
        const scoresShape = inputs[0]?.shape ?? [];
        const labelsShape = inputs[1]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        // Output is scalar unless reduction is "none"
        const outShape = reduction === "none" ? labelsShape : [];
        return [
            { shape: outShape, dtype },
            { shape: scoresShape, dtype }, // log_prob output matches scores
        ];
    },
};

export const NegativeLogLikelihoodLoss: OpSchema = {
    opType: "NegativeLogLikelihoodLoss",
    sinceVersion: 13,
    category: OpCategory.Other,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "target", typeConstraint: T_ANY },
        { name: "weight", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "loss", typeConstraint: T_ANY }],
    attributes: {
        reduction: {
            name: "reduction",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "mean",
        },
        ignore_index: { name: "ignore_index", type: AttributeType.INT, required: false },
    },
    inferShape: (inputs, attrs) => {
        const reduction = "reduction" in attrs ? (attrs["reduction"] as string) : "mean";
        const targetShape = inputs[1]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        const outShape = reduction === "none" ? targetShape : [];
        return [{ shape: outShape, dtype }];
    },
};

// --- Group Exports ---

export const LossOps: OpSchema[] = [SoftmaxCrossEntropyLoss, NegativeLogLikelihoodLoss];

export const ActivationNonLossOps: OpSchema[] = [
    PRelu,
    ThresholdedRelu,
    Shrink,
    LogSoftmax,
    Hardmax,
];

export const ActivationOps: OpSchema[] = [...LossOps, ...ActivationNonLossOps];
