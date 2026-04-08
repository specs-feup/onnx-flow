import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";
import { toStaticShape, broadcastShapes } from "@specs-feup/onnx-flow/Onnx/Utils";

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

    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const target = inputs[1]?.constantValue ?? [];
        const allowZero = (attrs["allowzero"] as number) ?? 0; // Grab the attribute

        if (target.length > 0) {
            if (inputShape.length > 0) {
                const inNums = inputShape.map((d) => (typeof d === "number" ? d : 1));
                const prodIn = inNums.reduce((a, b) => a * b, 1);

                let inferIndex = -1;
                let knownProd = 1;
                const resolved = [...target];

                resolved.forEach((d, i) => {
                    // If allowZero is 1, a 0 dimension actually means 0, not "copy".
                    if (d === 0 && allowZero === 0) {
                        resolved[i] = inNums[i] ?? 1;
                    }
                });
                resolved.forEach((d, i) => {
                    if (d === -1) {
                        inferIndex = i;
                    } else {
                        knownProd *= d || 1;
                    }
                });

                if (inferIndex !== -1) resolved[inferIndex] = prodIn / (knownProd || 1);
                return [{ shape: resolved, dtype }];
            } else {
                return [{ shape: target.map((d) => (d === 0 || d === -1 ? -1 : d)), dtype }];
            }
        }
        const targetShapeInput = inputs[1]?.shape ?? [];
        const rank = targetShapeInput[0];
        if (typeof rank === "number" && rank > 0) {
            return [{ shape: Array(rank).fill(-1), dtype }];
        }
        return [{ shape: [-1], dtype }];
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

    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const perm = (attrs["perm"] as number[]) ?? inputShape.map((_, i) => i).reverse();
        const outShape = perm.map((p) => inputShape[p] ?? 1);
        return [{ shape: outShape, dtype }];
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

    inferShape: (inputs) => {
        const dataShape = inputs[0]?.shape ?? [];
        const rank = dataShape.length;
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        if (rank === 0) return [{ shape: [], dtype }];

        const starts = inputs[1]?.constantValue ?? [];
        const ends = inputs[2]?.constantValue ?? [];
        let axes = inputs[3]?.constantValue ?? [];
        const steps = inputs[4]?.constantValue ?? [];

        if (!axes.length) {
            axes = Array.from({ length: starts.length || rank }, (_, i) => i);
        } else {
            axes = axes.map((a) => (a < 0 ? ((a % rank) + rank) % rank : a));
        }

        const out = [...dataShape];

        for (let i = 0; i < axes.length; i++) {
            const ax = axes[i];
            const len = typeof dataShape[ax] === "number" ? (dataShape[ax] as number) : 0;
            if (len === 0) {
                out[ax] = -1;
                continue;
            }

            let s = starts[i] ?? 0;
            let e = ends[i] ?? len;
            const step = steps[i] ?? 1;
            if (step === 0) continue;

            const normPos = (pos: number) =>
                pos < 0 ? Math.max(0, len + pos) : Math.min(len, pos);

            s = normPos(s);
            e = normPos(e);

            out[ax] = Math.max(0, Math.ceil((e - s) / step));
        }
        return [{ shape: out, dtype }];
    },
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

    inferShape: (inputs) => {
        const dataShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const pads = inputs[1]?.constantValue ?? [];

        const rank = dataShape.length;
        const outShape = [...dataShape];

        if (pads.length === 2 * rank) {
            for (let i = 0; i < rank; i++) {
                const dim = typeof outShape[i] === "number" ? (outShape[i] as number) : -1;
                if (dim < 0) {
                    outShape[i] = -1;
                } else {
                    outShape[i] = dim + (pads[i] ?? 0) + (pads[i + rank] ?? 0);
                }
            }
        }
        return [{ shape: outShape, dtype }];
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

    inferShape: (inputs, attrs) => {
        const axisRaw = (attrs["axis"] as number) ?? 0;
        const inputShapes = inputs.map((i) => i.shape);
        const ref = inputShapes.find((s) => s.length > 0) ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        if (ref.length === 0) return [{ shape: [], dtype }];

        const rank = ref.length;
        const axis = axisRaw < 0 ? axisRaw + rank : axisRaw;

        const outShape = [...ref];
        let sum = 0;
        for (const s of inputShapes) {
            if (s.length <= axis) {
                sum = -1;
                break;
            }
            const dim = typeof s[axis] === "number" ? (s[axis] as number) : -1;
            if (dim < 0) {
                sum = -1;
                break;
            }
            sum += dim;
        }
        outShape[axis] = sum;
        return [{ shape: outShape, dtype }];
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

    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const rank = inputShape.length;
        const axisRaw = (attrs["axis"] as number) ?? 0;
        const axis = rank > 0 ? ((axisRaw % rank) + rank) % rank : 0;

        const split = inputs[1]?.constantValue ?? [];
        const numOutputs = attrs["num_outputs"] as number | undefined;

        if (split.length > 0) {
            return split.map((s) => {
                const out = [...inputShape];
                out[axis] = s;
                return { shape: out, dtype };
            });
        } else if (numOutputs !== undefined && numOutputs > 0) {
            const dim = typeof inputShape[axis] === "number" ? (inputShape[axis] as number) : -1;
            const splitDim = dim > 0 ? Math.floor(dim / numOutputs) : -1;
            return Array(numOutputs)
                .fill(null)
                .map(() => {
                    const out = [...inputShape];
                    out[axis] = splitDim;
                    return { shape: out, dtype };
                });
        }
        return [{ shape: [], dtype }]; // Variadic fallback
    },
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

    inferShape: (inputs, attrs) => {
        return [
            {
                shape: inputs[0]?.shape ?? [],
                dtype: (attrs["to"] as number) ?? DataType.UNDEFINED,
            },
        ];
    },
};

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

    inferShape: (inputs, attrs) => {
        const dataShape = inputs[0]?.shape ?? [];
        const indicesShape = inputs[1]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const axisRaw = (attrs["axis"] as number) ?? 0;
        const rank = dataShape.length;
        const axis = rank > 0 ? ((axisRaw % rank) + rank) % rank : 0;

        const outShape = [
            ...dataShape.slice(0, axis),
            ...indicesShape,
            ...dataShape.slice(axis + 1),
        ];
        return [{ shape: outShape, dtype }];
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

    inferShape: (inputs) => {
        return [
            {
                shape: inputs[1]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
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

    inferShape: (inputs, attrs) => {
        const dataShape = inputs[0]?.shape ?? [];
        const indicesShape = inputs[1]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const batchDims = (attrs["batch_dims"] as number) ?? 0;

        if (dataShape.length === 0 || indicesShape.length === 0) {
            return [{ shape: [], dtype }];
        }

        const lastIndexDim = indicesShape[indicesShape.length - 1];
        if (typeof lastIndexDim !== "number") return [{ shape: [], dtype }];

        // output_shape = indices_shape[:-1] + data_shape[batch_dims + indices_shape[-1]:]
        const outShape = [
            ...indicesShape.slice(0, -1),
            ...dataShape.slice(batchDims + lastIndexDim),
        ];

        return [{ shape: outShape, dtype }];
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

    inferShape: (inputs) => {
        // Output shape is identical to the first input (data)
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
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

    inferShape: (inputs) => {
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
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

    inferShape: (inputs) => {
        // Output shape is identical to the first input (data)
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
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

    inferShape: (inputs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const repeats = inputs[1]?.constantValue ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        if (inputShape.length > 0 && repeats.length === inputShape.length) {
            const outShape = inputShape.map((d, i) => {
                if (typeof d === "number" && typeof repeats[i] === "number") return d * repeats[i];
                return -1;
            });
            return [{ shape: outShape, dtype }];
        }
        return [{ shape: inputShape.map(() => -1), dtype }];
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

    inferShape: (inputs) => {
        const tensorShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const axes = [...(inputs[1]?.constantValue ?? [])].sort((a, b) => a - b);

        const outShape = [...tensorShape];
        for (const ax of axes) outShape.splice(ax, 0, 1);
        return [{ shape: outShape, dtype }];
    },
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

    inferShape: (inputs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const axes = inputs[1]?.constantValue ?? [];

        if (axes.length === 0) {
            return [{ shape: inputShape.filter((d) => d !== 1), dtype }];
        } else {
            const rank = inputShape.length;
            const norm = new Set(axes.map((a) => (a < 0 ? ((a % rank) + rank) % rank : a)));
            return [{ shape: inputShape.filter((dim, idx) => !norm.has(idx) || dim !== 1), dtype }];
        }
    },
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

    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const axis = (attrs["axis"] as number) ?? 1;
        const d0 = inputShape
            .slice(0, axis)
            .reduce((a, b) => (a as number) * (b as number), 1) as number;
        const d1 = inputShape
            .slice(axis)
            .reduce((a, b) => (a as number) * (b as number), 1) as number;
        return [{ shape: [d0, d1], dtype }];
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

    inferShape: (inputs) => {
        const inputShape = toStaticShape(inputs[0]?.shape);
        const targetShape = inputs[1]?.constantValue ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        if (targetShape.length > 0) {
            // ONNX Expand broadcasts the input shape to the target shape
            return [{ shape: broadcastShapes(inputShape, targetShape), dtype }];
        }
        return [{ shape: [-1], dtype }];
    },
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

    inferShape: (inputs, attrs) => {
        const inputShape = inputs[0]?.shape ?? [];
        const rank = inputShape.length;
        if (rank === 0) return [{ shape: [0], dtype: DataType.INT64 }];

        const hasStart = "start" in attrs;
        const hasEnd = "end" in attrs;
        let start = hasStart ? (attrs["start"] as number) : 0;
        let end = hasEnd ? (attrs["end"] as number) : rank;

        const norm = (idx: number, r: number) => (r > 0 ? ((idx % r) + r) % r : 0);
        start = Math.max(0, Math.min(norm(start, rank), rank));
        end = Math.max(0, Math.min(norm(end, rank), rank));

        let length = Math.max(0, end - start);
        if (!hasStart && !hasEnd && length === 0 && rank > 0) length = rank;

        return [{ shape: [length], dtype: DataType.INT64 }];
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

    inferShape: (inputs) => {
        const indicesShape = inputs[0]?.shape ?? [];
        const depth = inputs[1]?.constantValue?.[0] ?? 0;
        const dtype = inputs[2]?.dtype ?? DataType.FLOAT;

        if (indicesShape.length > 0) {
            return [{ shape: [...indicesShape, depth > 0 ? depth : 1], dtype }];
        }
        return [{ shape: depth > 0 ? [depth] : [], dtype }];
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

    inferShape: (inputs) => {
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
};

export const Size: OpSchema = {
    opType: "Size",
    sinceVersion: 1,
    category: OpCategory.DataMovement,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "data", typeConstraint: T_ANY }],
    outputs: [{ name: "size", typeConstraint: T_INT }],
    attributes: {},

    inferShape: () => [{ shape: [], dtype: DataType.INT64 }], // Size always outputs a 0D scalar
};

export const DataMovementOps: OpSchema[] = [
    Reshape,
    Transpose,
    Slice,
    Concat,
    Split,
    Pad,
    Cast,
    Gather,
    GatherElements,
    GatherND,
    Scatter,
    ScatterElements,
    ScatterND,
    Tile,
    Unsqueeze,
    Squeeze,
    Flatten,
    Expand,
    Shape,
    OneHot,
    Identity,
    Size,
];
