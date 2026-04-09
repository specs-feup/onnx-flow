import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";

export const ReduceOps: OpSchema[] = [
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

        // ReduceLogSumExp uses axes as attribute in some older opsets, but we'll read both.
        axes: { name: "axes", type: AttributeType.INTS, required: false },
    },

    inferShape: (inputs, attrs) => {
        const inShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const keepdims = "keepdims" in attrs ? (attrs["keepdims"] as number) !== 0 : true;

        let axes: number[] | undefined;
        const axesAttr = attrs["axes"];
        if (Array.isArray(axesAttr)) axes = axesAttr as number[];
        else if (typeof axesAttr === "number") axes = [axesAttr];
        else if (inputs[1]?.constantValue) axes = inputs[1].constantValue;

        // ReduceLogSumExp defaults to reducing all axes if not provided
        if (!axes || axes.length === 0) {
            return [{ shape: keepdims ? inShape.map(() => 1) : [], dtype }];
        }

        const rank = inShape.length;
        const norm = new Set(axes.map((a) => (a < 0 ? ((a % rank) + rank) % rank : a)));
        const outShape = keepdims
            ? inShape.map((d, i) => (norm.has(i) ? 1 : d))
            : inShape.filter((_, i) => !norm.has(i));

        return [{ shape: outShape, dtype }];
    },
}));

export const ArgOps: OpSchema[] = ["ArgMax", "ArgMin"].map((opType) => ({
    opType,
    sinceVersion: 12,
    category: OpCategory.Reduction,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "data", typeConstraint: T_ANY }],
    outputs: [{ name: "reduced", typeConstraint: T_INT }],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: 0 },
        keepdims: { name: "keepdims", type: AttributeType.INT, required: false, defaultValue: 1 },
        select_last_index: {
            name: "select_last_index",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },

    inferShape: (inputs, attrs) => {
        const inShape = inputs[0]?.shape ?? [];
        const keepdims = "keepdims" in attrs ? (attrs["keepdims"] as number) !== 0 : true;
        const rank = inShape.length;
        const axisRaw = "axis" in attrs ? (attrs["axis"] as number) : 0;
        const axis = rank > 0 ? ((axisRaw % rank) + rank) % rank : 0;

        const outShape = [...inShape];
        if (rank > 0) {
            if (keepdims) {
                outShape[axis] = 1;
            } else {
                outShape.splice(axis, 1);
            }
        }
        return [{ shape: outShape, dtype: DataType.INT64 }];
    },
}));

export const TopK: OpSchema = {
    opType: "TopK",
    sinceVersion: 10, // Opset 10 moved K to input
    category: OpCategory.Reduction,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "K", typeConstraint: T_INT }, // 1D tensor containing a single integer
    ],
    outputs: [
        { name: "Values", typeConstraint: T_ANY },
        { name: "Indices", typeConstraint: T_INT },
    ],
    attributes: {
        axis: { name: "axis", type: AttributeType.INT, required: false, defaultValue: -1 },
        largest: { name: "largest", type: AttributeType.INT, required: false, defaultValue: 1 },
        sorted: { name: "sorted", type: AttributeType.INT, required: false, defaultValue: 1 },
    },

    inferShape: (inputs, attrs) => {
        const xShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const kVal = inputs[1]?.constantValue?.[0]; // Usually K is a 1-element 1D tensor

        const rank = xShape.length;
        const axisRaw = "axis" in attrs ? (attrs["axis"] as number) : -1;
        const axis = rank > 0 ? ((axisRaw % rank) + rank) % rank : 0;

        const outShape = [...xShape];
        if (rank > 0) {
            outShape[axis] = typeof kVal === "number" ? kVal : -1;
        }

        return [
            { shape: outShape, dtype }, // Values
            { shape: outShape, dtype: DataType.INT64 }, // Indices
        ];
    },
};

export const ReductionOps: OpSchema[] = [...ReduceOps, ...ArgOps, TopK];
