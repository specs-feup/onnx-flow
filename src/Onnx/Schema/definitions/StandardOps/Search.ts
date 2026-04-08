import { DataType, AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";

export const NonZero: OpSchema = {
    opType: "NonZero",
    sinceVersion: 13,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_INT }],
    attributes: {},
    inferShape: (inputs) => {
        const xShape = inputs[0]?.shape ?? [];
        const rank = xShape.length;
        // Output is always a 2D tensor of shape [rank, num_non_zero_elements]
        // Since we don't know num_non_zero_elements at compile time, we use -1
        return [{ shape: [rank, -1], dtype: DataType.INT64 }];
    },
};

export const Unique: OpSchema = {
    opType: "Unique",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [
        { name: "Y", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT, optional: true },
        { name: "inverse_indices", typeConstraint: T_INT, optional: true },
        { name: "counts", typeConstraint: T_INT, optional: true },
    ],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false },
        sorted: { name: "sorted", type: AttributeType.INT, required: false, defaultValue: 1 },
    },
    inferShape: (inputs, attrs) => {
        const xShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        if ("axis" in attrs) {
            const axisRaw = attrs["axis"] as number;
            const rank = xShape.length;
            const axis = rank > 0 ? ((axisRaw % rank) + rank) % rank : 0;

            const yShape = [...xShape];
            yShape[axis] = -1; // The targeted dimension is now dynamic

            return [
                { shape: yShape, dtype }, // Y
                { shape: [-1], dtype: DataType.INT64 }, // indices
                { shape: [xShape[axis] ?? -1], dtype: DataType.INT64 }, // inverse_indices
                { shape: [-1], dtype: DataType.INT64 }, // counts
            ];
        } else {
            // Flattened case
            return [
                { shape: [-1], dtype }, // Y
                { shape: [-1], dtype: DataType.INT64 }, // indices
                { shape: xShape, dtype: DataType.INT64 }, // inverse_indices
                { shape: [-1], dtype: DataType.INT64 }, // counts
            ];
        }
    },
};

export const NonMaxSuppression: OpSchema = {
    opType: "NonMaxSuppression",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "boxes", typeConstraint: T_ANY },
        { name: "scores", typeConstraint: T_ANY },
        { name: "max_output_boxes_per_class", typeConstraint: T_INT, optional: true },
        { name: "iou_threshold", typeConstraint: T_ANY, optional: true },
        { name: "score_threshold", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "selected_indices", typeConstraint: T_INT }],
    attributes: {
        center_point_box: {
            name: "center_point_box",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
    inferShape: () => {
        // Output is always [num_selected_indices, 3], where num_selected is dynamic (-1)
        // The 3 indices are [batch_index, class_index, box_index]
        return [{ shape: [-1, 3], dtype: DataType.INT64 }];
    },
};

export const ReverseSequence: OpSchema = {
    opType: "ReverseSequence",
    sinceVersion: 14,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "sequence_lens", typeConstraint: T_INT },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        batch_axis: {
            name: "batch_axis",
            type: AttributeType.INT,
            required: false,
            defaultValue: 1,
        },
        time_axis: { name: "time_axis", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
    inferShape: (inputs) => {
        // Output shape exactly matches input shape
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
};

export const Compress: OpSchema = {
    opType: "Compress",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "condition", typeConstraint: "tensor(bool)" },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false },
    },
    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        if ("axis" in attrs) {
            const axisRaw = attrs["axis"] as number;
            const rank = inputShape.length;
            const axis = rank > 0 ? ((axisRaw % rank) + rank) % rank : 0;
            const outShape = [...inputShape];
            outShape[axis] = -1; // The compressed dimension size is dynamic
            return [{ shape: outShape, dtype }];
        }

        // If axis is not provided, the input is flattened before compression
        return [{ shape: [-1], dtype }];
    },
};

export const SearchOps: OpSchema[] = [
    NonZero,
    Unique,
    NonMaxSuppression,
    ReverseSequence,
    Compress,
];
