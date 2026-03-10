import { AttributeType } from "../../OnnxTypes.js";
import { OpCategory, type OpSchema } from "../OpSchema.js";

// --- Helper for common types ---
const T_FLOAT = "tensor(float)";
const T_INT = "tensor(int64)";
const T_BOOL = "tensor(bool)";
const T_ANY = "T"; // Generic type constraint

export const ElementWiseOps: OpSchema[] = [
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Min",
    "Max",
    "And",
    "Or",
    "Xor",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "Equal",
    "NotEqual",
    "Mod", 
    "BitwiseAnd", 
    "BitwiseOr", 
    "BitwiseXor", 
    "BitShift"
].map((opType) => ({
    opType,
    sinceVersion: 7, // Stable baseline for elementwise
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [
        { name: "A", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "C", typeConstraint: T_ANY }],
    attributes: {}, // Elementwise ops usually have no attributes (except specific version quirks)
    typeConstraints: { T: [T_FLOAT, "tensor(int32)", T_INT] },
}));

// Bitwise/Logical ops return BOOL
[
    "And",
    "Or",
    "Xor",
    "Greater",
    "Less",
    "GreaterOrEqual",
    "LessOrEqual",
    "Equal",
    "NotEqual",
].forEach((op) => {
    const schema = ElementWiseOps.find((s) => s.opType === op)!;
    schema.outputs[0].typeConstraint = T_BOOL;
});

export const Not: OpSchema = {
    opType: "Not",
    sinceVersion: 1,
    category: OpCategory.ElementWise,
    broadcastable: true, // While technically unary, it follows elementwise propagation rules
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_BOOL }],
    outputs: [{ name: "Y", typeConstraint: T_BOOL }],
    attributes: {},
};

export const UnaryOps: OpSchema[] = [
    "Relu",
    "Sigmoid",
    "Tanh",
    "Exp",
    "Sqrt",
    "Abs",
    "Neg",
    "Floor",
    "Ceil",
    "Round",
    "Log", 
    "Sign", 
    "Reciprocal", 
    "Erf", 
    "IsNaN", 
    "IsInf",
    "Sin", 
    "Cos", 
    "Tan", 
    "Asin", 
    "Acos", 
    "Atan", 
    "Sinh", 
    "Cosh", 
    "Asinh", 
    "Acosh", 
    "Atanh",
    "Elu", 
    "Celu", 
    "Selu", 
    "Gelu", 
    "HardSigmoid", 
    "HardSwish", 
    "Mish", 
    "Softplus", 
    "Softsign"
].map((opType) => ({
    opType,
    sinceVersion: 6,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {},
}));

// Specific Unary with Attributes
export const LeakyRelu: OpSchema = {
    opType: "LeakyRelu",
    sinceVersion: 6,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        alpha: { name: "alpha", type: AttributeType.FLOAT, required: false, defaultValue: 0.01 },
    },
};

export const Clip: OpSchema = {
    opType: "Clip",
    sinceVersion: 11, // In 11, min/max moved to inputs
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "min", typeConstraint: T_ANY, optional: true },
        { name: "max", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

export const MatMul: OpSchema = {
    opType: "MatMul",
    sinceVersion: 1,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "A", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {},
};

export const Gemm: OpSchema = {
    opType: "Gemm",
    sinceVersion: 11, // beta/transA/transB attributes support
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "A", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY },
        { name: "C", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        alpha: { name: "alpha", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        beta: { name: "beta", type: AttributeType.FLOAT, required: false, defaultValue: 1.0 },
        transA: { name: "transA", type: AttributeType.INT, required: false, defaultValue: 0 },
        transB: { name: "transB", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
};

export const Conv: OpSchema = {
    opType: "Conv",
    sinceVersion: 11,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "W", typeConstraint: T_ANY },
        { name: "B", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
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
};

export const PoolingOps: OpSchema[] = ["MaxPool", "AveragePool"].map((opType) => ({
    opType,
    sinceVersion: 12, // Stable pooling
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [
        { name: "Y", typeConstraint: T_ANY },
        { name: "Indices", typeConstraint: "I", optional: true }, // MaxPool only
    ],
    attributes: {
        auto_pad: {
            name: "auto_pad",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "NOTSET",
        },
        ceil_mode: { name: "ceil_mode", type: AttributeType.INT, required: false, defaultValue: 0 },
        dilations: { name: "dilations", type: AttributeType.INTS, required: false },
        kernel_shape: { name: "kernel_shape", type: AttributeType.INTS, required: true },
        pads: { name: "pads", type: AttributeType.INTS, required: false },
        strides: { name: "strides", type: AttributeType.INTS, required: false },
        // AveragePool specific
        count_include_pad: {
            name: "count_include_pad",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
}));

export const GlobalPoolingOps: OpSchema[] = ["GlobalAveragePool", "GlobalMaxPool"].map(
    (opType) => ({
        opType,
        sinceVersion: 1,
        category: OpCategory.Spatial,
        broadcastable: false,
        hasState: false,
        inputs: [{ name: "X", typeConstraint: T_ANY }],
        outputs: [{ name: "Y", typeConstraint: T_ANY }],
        attributes: {},
    }),
);

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
};

export const Reshape: OpSchema = {
    opType: "Reshape",
    sinceVersion: 5,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "shape", typeConstraint: T_INT },
    ],
    outputs: [{ name: "reshaped", typeConstraint: T_ANY }],
    attributes: {
        allowzero: { name: "allowzero", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
};

export const Transpose: OpSchema = {
    opType: "Transpose",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "data", typeConstraint: T_ANY }],
    outputs: [{ name: "transposed", typeConstraint: T_ANY }],
    attributes: {
        perm: { name: "perm", type: AttributeType.INTS, required: false },
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
};

export const Slice: OpSchema = {
    opType: "Slice",
    sinceVersion: 13, // Inputs: data, starts, ends, axes, steps
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "starts", typeConstraint: T_INT },
        { name: "ends", typeConstraint: T_INT },
        { name: "axes", typeConstraint: T_INT, optional: true },
        { name: "steps", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

export const Pad: OpSchema = {
    opType: "Pad",
    sinceVersion: 11, // Pads moved to input
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "pads", typeConstraint: T_INT },
        { name: "constant_value", typeConstraint: T_ANY, optional: true },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        mode: {
            name: "mode",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "constant",
        },
    },
};

export const Concat: OpSchema = {
    opType: "Concat",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "inputs", typeConstraint: T_ANY, variadic: true }],
    outputs: [{ name: "concat_result", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: true },
    },
};

export const Split: OpSchema = {
    opType: "Split",
    sinceVersion: 13, // In opset 13, 'split' moved from attribute to input
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "split", typeConstraint: T_INT, optional: true }, // 1D Tensor of lengths
    ],
    outputs: [{ name: "outputs", typeConstraint: T_ANY, variadic: true }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
        num_outputs: { name: "num_outputs", type: AttributeType.INT, required: false }, // Useful if 'split' input is omitted
    },
};

export const Tile: OpSchema = {
    opType: "Tile",
    sinceVersion: 6,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "repeats", typeConstraint: T_INT }, // 1D Tensor specifying replication
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

export const ReductionOps: OpSchema[] = [
    "ReduceSum",
    "ReduceMean",
    "ReduceProd",
    "ReduceMin",
    "ReduceMax",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceSumSquare",
    "ReduceLogSumExp",
].map((opType) => ({
    opType,
    sinceVersion: 13, // Axes moved to input (mostly) - Note: ONNX 13 moved axes to input for ReduceSum but others followed later.
    category: OpCategory.Reduction,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "axes", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "reduced", typeConstraint: T_ANY }],
    attributes: {
        keepdims: { name: "keepdims", type: AttributeType.INT, required: false, defaultValue: 1 },
        noop_with_empty_axes: {
            name: "noop_with_empty_axes",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
}));

export const Gather: OpSchema = {
    opType: "Gather",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
};

export const GatherElements: OpSchema = {
    opType: "GatherElements",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
};

export const GatherND: OpSchema = {
    opType: "GatherND",
    sinceVersion: 12,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        batch_dims: {
            name: "batch_dims",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
};

export const Scatter: OpSchema = {
    opType: "Scatter",
    sinceVersion: 9,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT },
        { name: "updates", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
};

export const ScatterElements: OpSchema = {
    opType: "ScatterElements",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT },
        { name: "updates", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
};

export const ScatterND: OpSchema = {
    opType: "ScatterND",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "indices", typeConstraint: T_INT },
        { name: "updates", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        reduction: {
            name: "reduction",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "none",
        },
    },
};

export const Unsqueeze: OpSchema = {
    opType: "Unsqueeze",
    sinceVersion: 13, // Axes moved to input
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "axes", typeConstraint: T_INT },
    ],
    outputs: [{ name: "expanded", typeConstraint: T_ANY }],
    attributes: {},
};

export const Squeeze: OpSchema = {
    opType: "Squeeze",
    sinceVersion: 13, // Axes moved to input
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "data", typeConstraint: T_ANY },
        { name: "axes", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "squeezed", typeConstraint: T_ANY }],
    attributes: {},
};

export const Cast: OpSchema = {
    opType: "Cast",
    sinceVersion: 9,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: "T1" }],
    outputs: [{ name: "output", typeConstraint: "T2" }],
    attributes: {
        to: { name: "to", type: AttributeType.INT, required: true }, // DataType enum
    },
};

export const Range: OpSchema = {
    opType: "Range",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "start", typeConstraint: T_ANY },
        { name: "limit", typeConstraint: T_ANY },
        { name: "delta", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

export const Where: OpSchema = {
    opType: "Where",
    sinceVersion: 9,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [
        { name: "condition", typeConstraint: "B" },
        { name: "X", typeConstraint: T_ANY },
        { name: "Y", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

export const Flatten: OpSchema = {
    opType: "Flatten",
    sinceVersion: 11,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 1 },
    },
};

export const Expand: OpSchema = {
    opType: "Expand",
    sinceVersion: 8,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "shape", typeConstraint: T_INT },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

export const Shape: OpSchema = {
    opType: "Shape",
    sinceVersion: 15, // Added 'start' and 'end'
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "data", typeConstraint: T_ANY }],
    outputs: [{ name: "shape", typeConstraint: T_INT }],
    attributes: {
        start: { name: "start", type: AttributeType.INT, required: false, defaultValue: 0 },
        end: { name: "end", type: AttributeType.INT, required: false }, // Optional, defaults to rank
    },
};

export const ConstantOfShape: OpSchema = {
    opType: "ConstantOfShape",
    sinceVersion: 9,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_INT, optional: true }], // The shape
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {
        value: { name: "value", type: AttributeType.TENSOR, required: false }, // defaults to float 0.0
    },
};

export const OneHot: OpSchema = {
    opType: "OneHot",
    sinceVersion: 9,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "indices", typeConstraint: "T1" },
        { name: "depth", typeConstraint: "T2" },
        { name: "values", typeConstraint: "T3" },
    ],
    outputs: [{ name: "output", typeConstraint: "T3" }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
    },
};

export const Loop: OpSchema = {
    opType: "Loop",
    sinceVersion: 13, // Use 13 or 16
    category: OpCategory.ControlFlow,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "trip_count", typeConstraint: "I", optional: true },
        { name: "cond", typeConstraint: "B", optional: true },
        { name: "v_initial", typeConstraint: "V", variadic: true },
    ],
    outputs: [{ name: "v_final_and_scan_outputs", typeConstraint: "V", variadic: true }],
    attributes: {
        body: { name: "body", type: AttributeType.GRAPH, required: true },
    },
};

export const If: OpSchema = {
    opType: "If",
    sinceVersion: 13, // Opset 13 aligns with Loop/Scan graph semantics
    category: OpCategory.ControlFlow,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "cond", typeConstraint: "B" }, // Boolean scalar
    ],
    outputs: [{ name: "outputs", typeConstraint: T_ANY, variadic: true }],
    attributes: {
        then_branch: { name: "then_branch", type: AttributeType.GRAPH, required: true },
        else_branch: { name: "else_branch", type: AttributeType.GRAPH, required: true },
    },
};

export const Scan: OpSchema = {
    opType: "Scan",
    sinceVersion: 9,
    category: OpCategory.ControlFlow,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "sequence_lens", typeConstraint: "I", optional: true },
        { name: "initial_state_and_inputs", typeConstraint: "V", variadic: true },
    ],
    outputs: [{ name: "final_state_and_scan_outputs", typeConstraint: "V", variadic: true }],
    attributes: {
        body: { name: "body", type: AttributeType.GRAPH, required: true },
        num_scan_inputs: { name: "num_scan_inputs", type: AttributeType.INT, required: true },
        scan_input_directions: {
            name: "scan_input_directions",
            type: AttributeType.INTS,
            required: false,
        },
        scan_output_directions: {
            name: "scan_output_directions",
            type: AttributeType.INTS,
            required: false,
        },
        scan_input_axes: { name: "scan_input_axes", type: AttributeType.INTS, required: false },
        scan_output_axes: { name: "scan_output_axes", type: AttributeType.INTS, required: false },
    },
};

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
};

export const QuantizeLinear: OpSchema = {
    opType: "QuantizeLinear",
    sinceVersion: 13, // Standard definition
    category: OpCategory.ElementWise,
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
};

export const DequantizeLinear: OpSchema = {
    opType: "DequantizeLinear",
    sinceVersion: 13,
    category: OpCategory.ElementWise,
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
};

export const Identity: OpSchema = {
    opType: "Identity",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "input", typeConstraint: T_ANY }],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
};

// --- Aggregate all into one export ---
export const StandardOps: OpSchema[] = [
    ...ElementWiseOps,
    ...UnaryOps,
    LeakyRelu,
    Clip,
    MatMul,
    Gemm,
    Conv,
    ...PoolingOps,
    BatchNormalization,
    LSTM,
    Reshape,
    Transpose,
    Softmax,
    Flatten,
    Expand,
    Slice,
    Pad,
    Concat,
    Gather,
    Unsqueeze,
    Squeeze,
    OneHot,
    ...ReductionOps,
    Cast,
    Range,
    Where,
    Shape,
    Loop,
    Scan,
    QuantizeLinear,
    DequantizeLinear,
    Identity,
    ScatterElements,
    ConstantOfShape,
    If,
    GatherElements,
    GatherND,
    Scatter,
    ScatterND,
    Split,
    Tile,
    Not,
    ...GlobalPoolingOps,
];
