import { DataType, AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";

export const SequenceConstruct: OpSchema = {
    opType: "SequenceConstruct",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "inputs", typeConstraint: T_ANY, variadic: true }],
    outputs: [{ name: "output_sequence", typeConstraint: "seq(T)" }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const SequenceEmpty: OpSchema = {
    opType: "SequenceEmpty",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [],
    outputs: [{ name: "output_sequence", typeConstraint: "seq(T)" }],
    attributes: {
        dtype: { name: "dtype", type: AttributeType.INT, required: false },
    },
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const SequenceAt: OpSchema = {
    opType: "SequenceAt",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input_sequence", typeConstraint: "seq(T)" },
        { name: "position", typeConstraint: T_INT },
    ],
    outputs: [{ name: "tensor", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }], // Dependent on the contents of the sequence
};

export const SequenceInsert: OpSchema = {
    opType: "SequenceInsert",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input_sequence", typeConstraint: "seq(T)" },
        { name: "tensor", typeConstraint: T_ANY },
        { name: "position", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "output_sequence", typeConstraint: "seq(T)" }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const SequenceErase: OpSchema = {
    opType: "SequenceErase",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input_sequence", typeConstraint: "seq(T)" },
        { name: "position", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "output_sequence", typeConstraint: "seq(T)" }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const SequenceLength: OpSchema = {
    opType: "SequenceLength",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input_sequence", typeConstraint: "seq(T)" }],
    outputs: [{ name: "length", typeConstraint: T_INT }],
    attributes: {},
    inferShape: () => [{ shape: [], dtype: DataType.INT64 }], // Output is always a scalar 0D tensor
};

export const ConcatFromSequence: OpSchema = {
    opType: "ConcatFromSequence",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input_sequence", typeConstraint: "seq(T)" }],
    outputs: [{ name: "concat_result", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: true },
        new_axis: { name: "new_axis", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }], // Dependent on the contents of the sequence
};

export const SplitToSequence: OpSchema = {
    opType: "SplitToSequence",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "split", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "split_sequence", typeConstraint: "seq(T)" }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
        keepdims: { name: "keepdims", type: AttributeType.INT, required: false, defaultValue: 1 },
    },
    inferShape: () => [{ shape: [], dtype: DataType.UNDEFINED }],
};

export const SequenceOps: OpSchema[] = [
    SequenceConstruct,
    SequenceEmpty,
    SequenceAt,
    SequenceInsert,
    SequenceErase,
    SequenceLength,
    ConcatFromSequence,
    SplitToSequence,
];
