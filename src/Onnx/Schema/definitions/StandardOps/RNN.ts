import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY } from "../../OpSchema.js";
import { BatchNormalization } from "./Normalization.js";

export const LSTM: OpSchema = {
    opType: "LSTM",
    sinceVersion: 14, // Stable version with layout support (optional)
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: true,
    inputs: [
        { name: "X", typeConstraint: T_ANY }, // [seq_length, batch_size, input_size]
        { name: "W", typeConstraint: T_ANY }, // [num_directions, hidden_size, input_size]
        { name: "R", typeConstraint: T_ANY }, // [num_directions, hidden_size, hidden_size]
        { name: "B", typeConstraint: T_ANY, optional: true }, // [num_directions, 8*hidden_size]
        { name: "sequence_lens", typeConstraint: "T1", optional: true }, // [batch_size]
        { name: "initial_h", typeConstraint: T_ANY, optional: true }, // [num_directions, batch_size, hidden_size]
        { name: "initial_c", typeConstraint: T_ANY, optional: true }, // [num_directions, batch_size, hidden_size]
        { name: "P", typeConstraint: T_ANY, optional: true }, // [num_directions, 3*hidden_size] (Peepholes)
    ],
    outputs: [
        { name: "Y", typeConstraint: T_ANY, optional: true }, // [seq_length, num_directions, batch_size, hidden_size]
        { name: "Y_h", typeConstraint: T_ANY, optional: true }, // [num_directions, batch_size, hidden_size]
        { name: "Y_c", typeConstraint: T_ANY, optional: true }, // [num_directions, batch_size, hidden_size]
    ],
    attributes: {
        activation_alpha: { name: "activation_alpha", type: AttributeType.FLOATS, required: false },
        activation_beta: { name: "activation_beta", type: AttributeType.FLOATS, required: false },
        activations: { name: "activations", type: AttributeType.STRINGS, required: false },
        clip: { name: "clip", type: AttributeType.FLOAT, required: false },
        direction: {
            name: "direction",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "forward",
        },
        hidden_size: { name: "hidden_size", type: AttributeType.INT, required: true }, // The only required attribute
        input_forget: {
            name: "input_forget",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        layout: { name: "layout", type: AttributeType.INT, required: false, defaultValue: 0 },
    },

    inferShape: (inputs, attrs) => {
        const xShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        const hidden_size = attrs["hidden_size"] as number;
        const direction = (attrs["direction"] as string) ?? "forward";
        const num_directions = direction === "bidirectional" ? 2 : 1;
        const layout = (attrs["layout"] as number) ?? 0;

        let seq_length: number | string = -1;
        let batch_size: number | string = -1;

        if (xShape.length >= 2) {
            seq_length = layout === 0 ? (xShape[0] ?? -1) : (xShape[1] ?? -1);
            batch_size = layout === 0 ? (xShape[1] ?? -1) : (xShape[0] ?? -1);
        }

        // Y output layout depends on the layout attribute
        const yShape =
            layout === 0
                ? [seq_length, num_directions, batch_size, hidden_size]
                : [batch_size, seq_length, num_directions, hidden_size];

        const yhShape = [num_directions, batch_size, hidden_size];
        const ycShape = [num_directions, batch_size, hidden_size];

        return [
            { shape: yShape, dtype },
            { shape: yhShape, dtype },
            { shape: ycShape, dtype },
        ];
    },
};

export const GRU: OpSchema = {
    opType: "GRU",
    sinceVersion: 14,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: true,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "W", typeConstraint: T_ANY },
        { name: "R", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY, optional: true },
        { name: "sequence_lens", typeConstraint: "T1", optional: true },
        { name: "initial_h", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [
        { name: "Y", typeConstraint: T_ANY, optional: true },
        { name: "Y_h", typeConstraint: T_ANY, optional: true },
    ],
    attributes: {
        activation_alpha: { name: "activation_alpha", type: AttributeType.FLOATS, required: false },
        activation_beta: { name: "activation_beta", type: AttributeType.FLOATS, required: false },
        activations: { name: "activations", type: AttributeType.STRINGS, required: false },
        clip: { name: "clip", type: AttributeType.FLOAT, required: false },
        direction: {
            name: "direction",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "forward",
        },
        hidden_size: { name: "hidden_size", type: AttributeType.INT, required: true },
        layout: { name: "layout", type: AttributeType.INT, required: false, defaultValue: 0 },
        linear_before_reset: {
            name: "linear_before_reset",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
    inferShape: (inputs, attrs) => {
        // Shape inference is identical to LSTM, just missing the Y_c output
        const lstmShapes = LSTM.inferShape!(inputs, attrs);
        return [lstmShapes[0], lstmShapes[1]];
    },
};

export const RNN: OpSchema = {
    opType: "RNN",
    sinceVersion: 14,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: true,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "W", typeConstraint: T_ANY },
        { name: "R", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY, optional: true },
        { name: "sequence_lens", typeConstraint: "T1", optional: true },
        { name: "initial_h", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [
        { name: "Y", typeConstraint: T_ANY, optional: true },
        { name: "Y_h", typeConstraint: T_ANY, optional: true },
    ],
    attributes: {
        activation_alpha: { name: "activation_alpha", type: AttributeType.FLOATS, required: false },
        activation_beta: { name: "activation_beta", type: AttributeType.FLOATS, required: false },
        activations: { name: "activations", type: AttributeType.STRINGS, required: false },
        clip: { name: "clip", type: AttributeType.FLOAT, required: false },
        direction: {
            name: "direction",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "forward",
        },
        hidden_size: { name: "hidden_size", type: AttributeType.INT, required: true },
        layout: { name: "layout", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
    inferShape: (inputs, attrs) => {
        // Shape inference is identical to GRU
        return GRU.inferShape!(inputs, attrs);
    },
};

export const RNNOps: OpSchema[] = [LSTM, GRU, RNN];
