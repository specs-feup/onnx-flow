import { DataType, AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { toStaticShape, broadcastShapes } from "@specs-feup/onnx-flow/Onnx/Utils";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_FLOAT, T_INT, T_BOOL } from "../../OpSchema.js";

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
    "Mod",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BitShift",
].map((opType) => ({
    opType: opType,
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

    inferShape: (inputs) => {
        const shapes = inputs.map((i) => toStaticShape(i.shape));
        const outShape = broadcastShapes(...shapes);
        const isBool = [
            "Greater",
            "Less",
            "GreaterOrEqual",
            "LessOrEqual",
            "Equal",
            "NotEqual",
            "And",
            "Or",
            "Xor",
        ].includes(opType);

        return [
            {
                shape: outShape,
                dtype: isBool ? DataType.BOOL : (inputs[0]?.dtype ?? DataType.UNDEFINED),
            },
        ];
    },
}));

// Bitwise/Logical ops return BOOL
["And", "Or", "Xor", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Equal"].forEach((op) => {
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

    inferShape: (inputs) => {
        return [
            {
                shape: inputs[0]?.shape ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
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
    "Softsign",
].map((opType) => ({
    opType,
    sinceVersion: 6,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {},

    inferShape: (inputs) => {
        return [
            {
                shape: inputs[0]?.shape ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
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

    inferShape: (inputs) => {
        return [
            {
                shape: inputs[0]?.shape ?? [],
                dtype: inputs[0]?.dtype ?? DataType.UNDEFINED,
            },
        ];
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

    inferShape: (inputs) => {
        const a = inputs[0]?.shape ?? [];
        const b = inputs[1]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        const rankA = a.length;
        const rankB = b.length;

        if (rankA >= 2 && rankB >= 2) {
            // N-Dimensional MatMul: Broadcast the batch dimensions
            const aStatic = toStaticShape(a);
            const bStatic = toStaticShape(b);
            const aBatch = aStatic.slice(0, -2);
            const bBatch = bStatic.slice(0, -2);

            const batchOut = broadcastShapes(aBatch, bBatch);

            // Preserve original dynamic/symbolic markers if possible,
            // or just use the static broadcasted numbers (-1 for unknown).
            return [{ shape: [...batchOut, a[rankA - 2], b[rankB - 1]], dtype }];
        } else if (rankA === 1 && rankB >= 2) {
            // 1D x N-D: Prepend 1 to A, multiply, then remove the prepended 1
            const bStatic = toStaticShape(b);
            const batchOut = bStatic.slice(0, -2);
            return [{ shape: [...batchOut, b[rankB - 1]], dtype }];
        } else if (rankA >= 2 && rankB === 1) {
            // N-D x 1D: Append 1 to B, multiply, then remove the appended 1
            const aStatic = toStaticShape(a);
            const batchOut = aStatic.slice(0, -2);
            return [{ shape: [...batchOut, a[rankA - 2]], dtype }];
        } else if (rankA === 1 && rankB === 1) {
            // 1D x 1D: Dot product produces a scalar
            return [{ shape: [], dtype }];
        }

        return [{ shape: [], dtype }];
    },
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

    inferShape: (inputs, attrs) => {
        let a = inputs[0]?.shape ?? [];
        let b = inputs[1]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        // Handle transA / transB for shape inference
        const transA = "transA" in attrs ? (attrs["transA"] as number) : 0;
        const transB = "transB" in attrs ? (attrs["transB"] as number) : 0;
        if (transA) a = [...a].reverse();
        if (transB) b = [...b].reverse();

        if (a.length === 2 && b.length === 2) {
            const mm = [a[0], b[1]];
            const c = inputs[2]?.shape ?? [];
            const outShape = c.length ? broadcastShapes(toStaticShape(mm), toStaticShape(c)) : mm;
            return [{ shape: outShape, dtype }];
        }
        return [{ shape: [], dtype }];
    },
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

    inferShape: (inputs) => {
        const sc = toStaticShape(inputs[0]?.shape);
        const sx = toStaticShape(inputs[1]?.shape);
        const sy = toStaticShape(inputs[2]?.shape);
        const dtype = inputs[1]?.dtype !== DataType.UNDEFINED ? inputs[1]?.dtype : inputs[2]?.dtype;
        return [{ shape: broadcastShapes(sc, sx, sy), dtype }];
    },
};

export const Sum: OpSchema = {
    opType: "Sum",
    sinceVersion: 13,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [{ name: "data_0", typeConstraint: T_ANY, variadic: true }],
    outputs: [{ name: "sum", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        const shapes = inputs.map((i) => toStaticShape(i.shape));
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        return [{ shape: broadcastShapes(...shapes), dtype }];
    },
};

export const Mean: OpSchema = {
    opType: "Mean",
    sinceVersion: 13,
    category: OpCategory.ElementWise,
    broadcastable: true,
    hasState: false,
    inputs: [{ name: "data_0", typeConstraint: T_ANY, variadic: true }],
    outputs: [{ name: "mean", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        const shapes = inputs.map((i) => toStaticShape(i.shape));
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        return [{ shape: broadcastShapes(...shapes), dtype }];
    },
};

export const CastLike: OpSchema = {
    opType: "CastLike",
    sinceVersion: 19,
    category: OpCategory.ElementWise,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "input", typeConstraint: T_ANY },
        { name: "target_type", typeConstraint: T_ANY },
    ],
    outputs: [{ name: "output", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        // Shape matches input[0], but dtype matches input[1]!
        return [
            {
                shape: inputs[0]?.shape?.slice() ?? [],
                dtype: inputs[1]?.dtype ?? DataType.UNDEFINED,
            },
        ];
    },
};

export const Trilu: OpSchema = {
    opType: "Trilu",
    sinceVersion: 14,
    category: OpCategory.Math,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "k", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        upper: { name: "upper", type: AttributeType.INT, required: false, defaultValue: 1 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const Det: OpSchema = {
    opType: "Det",
    sinceVersion: 11,
    category: OpCategory.Math,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {},
    inferShape: (inputs) => {
        const xShape = inputs[0]?.shape ?? [];
        // Det reduces the last two dimensions [..., M, M] -> [...]
        const outShape = xShape.length >= 2 ? xShape.slice(0, -2) : [];
        return [{ shape: outShape, dtype: inputs[0]?.dtype ?? DataType.UNDEFINED }];
    },
};

export const CumSum: OpSchema = {
    opType: "CumSum",
    sinceVersion: 14,
    category: OpCategory.Math,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "x", typeConstraint: T_ANY },
        { name: "axis", typeConstraint: T_INT },
    ],
    outputs: [{ name: "y", typeConstraint: T_ANY }],
    attributes: {
        exclusive: { name: "exclusive", type: AttributeType.INT, required: false, defaultValue: 0 },
        reverse: { name: "reverse", type: AttributeType.INT, required: false, defaultValue: 0 },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const Einsum: OpSchema = {
    opType: "Einsum",
    sinceVersion: 12,
    category: OpCategory.Math,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "Inputs", typeConstraint: T_ANY, variadic: true }],
    outputs: [{ name: "Output", typeConstraint: T_ANY }],
    attributes: {
        equation: { name: "equation", type: AttributeType.STRING, required: true },
    },
    inferShape: (inputs) => {
        // Full Einsum equation parsing is complex. For now, we return dynamic shapes.
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        return [{ shape: inputs[0]?.shape?.map(() => -1) ?? [], dtype }];
    },
};

export const MathOps: OpSchema[] = [
    ...ElementWiseOps,
    ...UnaryOps,
    LeakyRelu,
    Clip,
    MatMul,
    Gemm,
    Where,
    Sum,
    Mean,
    CastLike,
    Trilu,
    Det,
    CumSum,
    Einsum,
];
