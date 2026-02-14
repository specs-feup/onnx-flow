import { AttributeType } from "../../OnnxTypes.js";
import { OpSchema } from "../OpSchema.js";

// --- Helper for common types ---
const T_FLOAT = "tensor(float)";
const T_INT = "tensor(int64)";
const T_BOOL = "tensor(bool)";
const T_ANY = "T"; // Generic type constraint

export const ElementWiseOps: OpSchema[] = [
    "Add", "Sub", "Mul", "Div", "Pow", "Min", "Max", 
    "And", "Or", "Xor", "Greater", "Less", "GreaterOrEqual", 
    "LessOrEqual", "Equal", "NotEqual"
].map(opType => ({
    opType,
    sinceVersion: 7, // Stable baseline for elementwise
    inputs: [
        { name: "A", typeConstraint: "T" },
        { name: "B", typeConstraint: "T" }
    ],
    outputs: [{ name: "C", typeConstraint: "T" }],
    attributes: {}, // Elementwise ops usually have no attributes (except specific version quirks)
    typeConstraints: { "T": ["tensor(float)", "tensor(int32)", "tensor(int64)"] }
}));

// Fix: Bitwise/Logical ops return BOOL
["And", "Or", "Xor", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal", "NotEqual"].forEach(op => {
    const schema = ElementWiseOps.find(s => s.opType === op);
    if (schema) {
        schema.outputs[0].typeConstraint = "tensor(bool)";
    }
});

export const UnaryOps: OpSchema[] = [
    "Relu", "Sigmoid", "Tanh", "Exp", "Sqrt", "Abs", "Neg", "Floor", "Ceil"
].map(opType => ({
    opType,
    sinceVersion: 6,
    inputs: [{ name: "X", typeConstraint: "T" }],
    outputs: [{ name: "Y", typeConstraint: "T" }],
    attributes: {}
}));

// Specific Unary with Attributes
export const LeakyRelu: OpSchema = {
    opType: "LeakyRelu",
    sinceVersion: 6,
    inputs: [{ name: "X", typeConstraint: "T" }],
    outputs: [{ name: "Y", typeConstraint: "T" }],
    attributes: {
        "alpha": { name: "alpha", type: AttributeType.FLOAT, required: false, defaultValue: 0.01 }
    }
};

export const Clip: OpSchema = {
    opType: "Clip",
    sinceVersion: 11, // In 11, min/max moved to inputs
    inputs: [
        { name: "input", typeConstraint: "T" },
        { name: "min", typeConstraint: "T", optional: true },
        { name: "max", typeConstraint: "T", optional: true }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {}
};

export const MatMul: OpSchema = {
    opType: "MatMul",
    sinceVersion: 1,
    inputs: [
        { name: "A", typeConstraint: "T" },
        { name: "B", typeConstraint: "T" }
    ],
    outputs: [{ name: "Y", typeConstraint: "T" }],
    attributes: {}
};

export const Gemm: OpSchema = {
    opType: "Gemm",
    sinceVersion: 11, // beta/transA/transB attributes support
    inputs: [
        { name: "A", typeConstraint: "T" },
        { name: "B", typeConstraint: "T" },
        { name: "C", typeConstraint: "T", optional: true }
    ],
    outputs: [{ name: "Y", typeConstraint: "T" }],
    attributes: {
        "alpha": { name: "alpha", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        "beta": { name: "beta", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        "transA": { name: "transA", type: AttributeType.INT, required: false, defaultValue: 0 },
        "transB": { name: "transB", type: AttributeType.INT, required: false, defaultValue: 0 }
    }
};

export const Conv: OpSchema = {
    opType: "Conv",
    sinceVersion: 11,
    inputs: [
        { name: "X", typeConstraint: "T" },
        { name: "W", typeConstraint: "T" },
        { name: "B", typeConstraint: "T", optional: true }
    ],
    outputs: [{ name: "Y", typeConstraint: "T" }],
    attributes: {
        "auto_pad": { name: "auto_pad", type: AttributeType.STRING, required: false, defaultValue: "NOTSET" },
        "dilations": { name: "dilations", type: AttributeType.INTS, required: false },
        "group": { name: "group", type: AttributeType.INT, required: false, defaultValue: 1 },
        "kernel_shape": { name: "kernel_shape", type: AttributeType.INTS, required: false },
        "pads": { name: "pads", type: AttributeType.INTS, required: false },
        "strides": { name: "strides", type: AttributeType.INTS, required: false }
    }
};

export const PoolingOps: OpSchema[] = ["MaxPool", "AveragePool"].map(opType => ({
    opType,
    sinceVersion: 12, // Stable pooling
    inputs: [{ name: "X", typeConstraint: "T" }],
    outputs: [
        { name: "Y", typeConstraint: "T" },
        { name: "Indices", typeConstraint: "I", optional: true } // MaxPool only
    ],
    attributes: {
        "auto_pad": { name: "auto_pad", type: AttributeType.STRING, required: false, defaultValue: "NOTSET" },
        "ceil_mode": { name: "ceil_mode", type: AttributeType.INT, required: false, defaultValue: 0 },
        "dilations": { name: "dilations", type: AttributeType.INTS, required: false },
        "kernel_shape": { name: "kernel_shape", type: AttributeType.INTS, required: true },
        "pads": { name: "pads", type: AttributeType.INTS, required: false },
        "strides": { name: "strides", type: AttributeType.INTS, required: false },
        // AveragePool specific
        "count_include_pad": { name: "count_include_pad", type: AttributeType.INT, required: false, defaultValue: 0 }
    }
}));

export const BatchNormalization: OpSchema = {
    opType: "BatchNormalization",
    sinceVersion: 9,
    inputs: [
        { name: "X", typeConstraint: "T" },
        { name: "scale", typeConstraint: "T" },
        { name: "B", typeConstraint: "T" },
        { name: "mean", typeConstraint: "T" },
        { name: "var", typeConstraint: "T" }
    ],
    outputs: [{ name: "Y", typeConstraint: "T" }],
    attributes: {
        "epsilon": { name: "epsilon", type: AttributeType.FLOAT, required: false, defaultValue: 1e-5 },
        "momentum": { name: "momentum", type: AttributeType.FLOAT, required: false, defaultValue: 0.9 }
    }
};

export const Reshape: OpSchema = {
    opType: "Reshape",
    sinceVersion: 5,
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "shape", typeConstraint: "tensor(int64)" }
    ],
    outputs: [{ name: "reshaped", typeConstraint: "T" }],
    attributes: {
        "allowzero": { name: "allowzero", type: AttributeType.INT, required: false, defaultValue: 0 }
    }
};

export const Transpose: OpSchema = {
    opType: "Transpose",
    sinceVersion: 1,
    inputs: [{ name: "data", typeConstraint: "T" }],
    outputs: [{ name: "transposed", typeConstraint: "T" }],
    attributes: {
        "perm": { name: "perm", type: AttributeType.INTS, required: false }
    }
};

export const Softmax: OpSchema = {
    opType: "Softmax",
    sinceVersion: 13, // Axis handling stabilized
    inputs: [{ name: "input", typeConstraint: "T" }],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 }
    }
};

export const Slice: OpSchema = {
    opType: "Slice",
    sinceVersion: 13, // Inputs: data, starts, ends, axes, steps
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "starts", typeConstraint: "tensor(int64)" },
        { name: "ends", typeConstraint: "tensor(int64)" },
        { name: "axes", typeConstraint: "tensor(int64)", optional: true },
        { name: "steps", typeConstraint: "tensor(int64)", optional: true }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {}
};

export const Pad: OpSchema = {
    opType: "Pad",
    sinceVersion: 11, // Pads moved to input
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "pads", typeConstraint: "tensor(int64)" },
        { name: "constant_value", typeConstraint: "T", optional: true }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {
        "mode": { name: "mode", type: AttributeType.STRING, required: false, defaultValue: "constant" }
    }
};

export const Concat: OpSchema = {
    opType: "Concat",
    sinceVersion: 11,
    inputs: [
        { name: "inputs", typeConstraint: "T", variadic: true }
    ],
    outputs: [{ name: "concat_result", typeConstraint: "T" }],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: true }
    }
};

export const ReductionOps: OpSchema[] = [
    "ReduceSum", "ReduceMean", "ReduceProd", "ReduceMin", "ReduceMax", 
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceSumSquare", "ReduceLogSumExp"
].map(opType => ({
    opType,
    sinceVersion: 13, // Axes moved to input (mostly) - Note: ONNX 13 moved axes to input for ReduceSum but others followed later. 
                      // This schema assumes the modern 'Input' approach or 'Attribute' approach based on specific op history.
                      // For simplicity in your Phase 1, we define axes as Attribute (older) or Input (newer). 
                      // To support *all*, we mark it as Input, and your Adapter (Phase 2) will move it there.
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "axes", typeConstraint: "tensor(int64)", optional: true }
    ],
    outputs: [{ name: "reduced", typeConstraint: "T" }],
    attributes: {
        "keepdims": { name: "keepdims", type: AttributeType.INT, required: false, defaultValue: 1 },
        "noop_with_empty_axes": { name: "noop_with_empty_axes", type: AttributeType.INT, required: false, defaultValue: 0 }
    }
}));

export const Gather: OpSchema = {
    opType: "Gather",
    sinceVersion: 11,
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "indices", typeConstraint: "tensor(int64)" }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 }
    }
};

export const Unsqueeze: OpSchema = {
    opType: "Unsqueeze",
    sinceVersion: 13, // Axes moved to input
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "axes", typeConstraint: "tensor(int64)" }
    ],
    outputs: [{ name: "expanded", typeConstraint: "T" }],
    attributes: {}
};

export const Squeeze: OpSchema = {
    opType: "Squeeze",
    sinceVersion: 13, // Axes moved to input
    inputs: [
        { name: "data", typeConstraint: "T" },
        { name: "axes", typeConstraint: "tensor(int64)", optional: true }
    ],
    outputs: [{ name: "squeezed", typeConstraint: "T" }],
    attributes: {}
};

export const Cast: OpSchema = {
    opType: "Cast",
    sinceVersion: 9,
    inputs: [{ name: "input", typeConstraint: "T1" }],
    outputs: [{ name: "output", typeConstraint: "T2" }],
    attributes: {
        "to": { name: "to", type: AttributeType.INT, required: true } // DataType enum
    }
};

export const Range: OpSchema = {
    opType: "Range",
    sinceVersion: 11,
    inputs: [
        { name: "start", typeConstraint: "T" },
        { name: "limit", typeConstraint: "T" },
        { name: "delta", typeConstraint: "T" }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {}
};

export const Where: OpSchema = {
    opType: "Where",
    sinceVersion: 9,
    inputs: [
        { name: "condition", typeConstraint: "B" },
        { name: "X", typeConstraint: "T" },
        { name: "Y", typeConstraint: "T" }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {}
};

export const Flatten: OpSchema = {
    opType: "Flatten",
    sinceVersion: 11,
    inputs: [{ name: "input", typeConstraint: "T" }],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: false, defaultValue: 1 }
    }
};

export const Expand: OpSchema = {
    opType: "Expand",
    sinceVersion: 8,
    inputs: [
        { name: "input", typeConstraint: "T" },
        { name: "shape", typeConstraint: "tensor(int64)" }
    ],
    outputs: [{ name: "output", typeConstraint: "T" }],
    attributes: {}
};

export const Shape: OpSchema = {
    opType: "Shape",
    sinceVersion: 15, // Added 'start' and 'end'
    inputs: [{ name: "data", typeConstraint: "T" }],
    outputs: [{ name: "shape", typeConstraint: "tensor(int64)" }],
    attributes: {
        "start": { name: "start", type: AttributeType.INT, required: false, defaultValue: 0 },
        "end": { name: "end", type: AttributeType.INT, required: false }, // Optional, defaults to rank
    }
};

export const OneHot: OpSchema = {
    opType: "OneHot",
    sinceVersion: 9,
    inputs: [
        { name: "indices", typeConstraint: "T1" },
        { name: "depth", typeConstraint: "T2" },
        { name: "values", typeConstraint: "T3" }
    ],
    outputs: [{ name: "output", typeConstraint: "T3" }],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 }
    }
};

export const Loop: OpSchema = {
    opType: "Loop",
    sinceVersion: 13, // Use 13 or 16
    inputs: [
        { name: "trip_count", typeConstraint: "I", optional: true },
        { name: "cond", typeConstraint: "B", optional: true },
        { name: "v_initial", typeConstraint: "V", variadic: true }
    ],
    outputs: [
        { name: "v_final_and_scan_outputs", typeConstraint: "V", variadic: true }
    ],
    attributes: {
        "body": { name: "body", type: AttributeType.GRAPH, required: true }
    }
};

export const Scan: OpSchema = {
    opType: "Scan",
    sinceVersion: 9,
    inputs: [
        { name: "sequence_lens", typeConstraint: "I", optional: true },
        { name: "initial_state_and_inputs", typeConstraint: "V", variadic: true }
    ],
    outputs: [
        { name: "final_state_and_scan_outputs", typeConstraint: "V", variadic: true }
    ],
    attributes: {
        "body": { name: "body", type: AttributeType.GRAPH, required: true },
        "num_scan_inputs": { name: "num_scan_inputs", type: AttributeType.INT, required: true },
        "scan_input_directions": { name: "scan_input_directions", type: AttributeType.INTS, required: false },
        "scan_output_directions": { name: "scan_output_directions", type: AttributeType.INTS, required: false },
        "scan_input_axes": { name: "scan_input_axes", type: AttributeType.INTS, required: false },
        "scan_output_axes": { name: "scan_output_axes", type: AttributeType.INTS, required: false }
    }
};

export const LSTM: OpSchema = {
    opType: "LSTM",
    sinceVersion: 14, // Stable version with layout support (optional)
    inputs: [
        { name: "X", typeConstraint: "T" }, // [seq_length, batch_size, input_size]
        { name: "W", typeConstraint: "T" }, // [num_directions, hidden_size, input_size]
        { name: "R", typeConstraint: "T" }, // [num_directions, hidden_size, hidden_size]
        { name: "B", typeConstraint: "T", optional: true }, // [num_directions, 8*hidden_size]
        { name: "sequence_lens", typeConstraint: "T1", optional: true }, // [batch_size]
        { name: "initial_h", typeConstraint: "T", optional: true }, // [num_directions, batch_size, hidden_size]
        { name: "initial_c", typeConstraint: "T", optional: true }, // [num_directions, batch_size, hidden_size]
        { name: "P", typeConstraint: "T", optional: true }  // [num_directions, 3*hidden_size] (Peepholes)
    ],
    outputs: [
        { name: "Y", typeConstraint: "T", optional: true }, // [seq_length, num_directions, batch_size, hidden_size]
        { name: "Y_h", typeConstraint: "T", optional: true }, // [num_directions, batch_size, hidden_size]
        { name: "Y_c", typeConstraint: "T", optional: true }  // [num_directions, batch_size, hidden_size]
    ],
    attributes: {
        "activation_alpha": { name: "activation_alpha", type: AttributeType.FLOATS, required: false },
        "activation_beta": { name: "activation_beta", type: AttributeType.FLOATS, required: false },
        "activations": { name: "activations", type: AttributeType.STRINGS, required: false },
        "clip": { name: "clip", type: AttributeType.FLOAT, required: false },
        "direction": { name: "direction", type: AttributeType.STRING, required: false, defaultValue: "forward" },
        "hidden_size": { name: "hidden_size", type: AttributeType.INT, required: true }, // The only required attribute
        "input_forget": { name: "input_forget", type: AttributeType.INT, required: false, defaultValue: 0 },
        "layout": { name: "layout", type: AttributeType.INT, required: false, defaultValue: 0 }
    }
};

export const QuantizeLinear: OpSchema = {
    opType: "QuantizeLinear",
    sinceVersion: 13, // Standard definition
    inputs: [
        { name: "x", typeConstraint: "T1" },
        { name: "y_scale", typeConstraint: "T1" },
        { name: "y_zero_point", typeConstraint: "T2", optional: true }
    ],
    outputs: [
        { name: "y", typeConstraint: "T2" }
    ],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: false, defaultValue: 1 },
        // Saturate is a newer attribute (opset 19), usually defaults to 1 (true)
        "saturate": { name: "saturate", type: AttributeType.INT, required: false, defaultValue: 1 }
    }
};

export const DequantizeLinear: OpSchema = {
    opType: "DequantizeLinear",
    sinceVersion: 13,
    inputs: [
        { name: "x", typeConstraint: "T1" },
        { name: "x_scale", typeConstraint: "T2" },
        { name: "x_zero_point", typeConstraint: "T1", optional: true }
    ],
    outputs: [
        { name: "y", typeConstraint: "T2" }
    ],
    attributes: {
        "axis": { name: "axis", type: AttributeType.INT, required: false, defaultValue: 1 }
    }
};

// --- Aggregate all into one export ---
export const StandardOps: OpSchema[] = [
    ...ElementWiseOps,
    ...UnaryOps,
    LeakyRelu, Clip,
    MatMul, Gemm,
    Conv, ...PoolingOps, BatchNormalization, LSTM,
    Reshape, Transpose, Softmax, Flatten, Expand,
    Slice, Pad, Concat, Gather, Unsqueeze, Squeeze, OneHot,
    ...ReductionOps,
    Cast, Range, Where, Shape,
    Loop, Scan,
    QuantizeLinear, DequantizeLinear
];