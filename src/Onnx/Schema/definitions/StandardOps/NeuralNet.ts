import { AttributeType, DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import { toStaticShape, inferPoolDim } from "@specs-feup/onnx-flow/Onnx/Utils";
import type { OpSchema } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";

// --- Helper for Shared Pooling Inference ---
function inferPoolingShape(inputs: any[], attrs: Record<string, any>, isMaxPool: boolean) {
    const xShape = toStaticShape(inputs[0]?.shape);
    const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

    // Pooling requires at least [Batch, Channel, Spatial_1]
    if (xShape.length < 3) return [{ shape: xShape.slice(), dtype }];

    const rank = xShape.length;
    const spatialRank = rank - 2;
    const [N, C] = xShape;

    const kernel = ((attrs["kernel_shape"] as number[]) ?? Array(spatialRank).fill(1)).map(Number);
    const strides = ((attrs["strides"] as number[]) ?? Array(spatialRank).fill(1)).map(Number);
    const ceilMode = Number(attrs["ceil_mode"] ?? 0);
    const autoPad = (attrs["auto_pad"] as string) ?? "NOTSET";
    let pads = ((attrs["pads"] as number[]) ?? Array(spatialRank * 2).fill(0)).map(Number);

    // MaxPool supports dilations natively. AveragePool does not (until Opset 19).
    const dilations = isMaxPool
        ? ((attrs["dilations"] as number[]) ?? Array(spatialRank).fill(1)).map(Number)
        : Array(spatialRank).fill(1);

    // Calculate effective padding if auto_pad is used
    if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
        pads = Array(spatialRank * 2).fill(0);
        for (let i = 0; i < spatialRank; i++) {
            const dIn = xShape[i + 2] as number;
            if (dIn === -1) continue;

            const dEffK = dilations[i] * (kernel[i] - 1) + 1;
            const totalPad = Math.max(
                0,
                (Math.ceil(dIn / strides[i]) - 1) * strides[i] + dEffK - dIn,
            );

            if (autoPad === "SAME_UPPER") {
                pads[i] = Math.floor(totalPad / 2); // pad_low
                pads[i + spatialRank] = totalPad - pads[i]; // pad_high
            } else {
                pads[i + spatialRank] = Math.floor(totalPad / 2); // pad_low
                pads[i] = totalPad - pads[i + spatialRank]; // pad_high
            }
        }
    }

    const outSpatial = [];
    for (let i = 0; i < spatialRank; i++) {
        const dIn = xShape[i + 2];
        if (typeof dIn === "number" && dIn >= 0) {
            outSpatial.push(
                inferPoolDim(
                    dIn,
                    kernel[i],
                    strides[i],
                    pads[i],
                    pads[i + spatialRank],
                    dilations[i],
                    ceilMode,
                ),
            );
        } else {
            outSpatial.push(-1);
        }
    }

    const results = [{ shape: [N, C, ...outSpatial], dtype }];

    // MaxPool has an optional second output for Indices
    if (isMaxPool) {
        results.push({ shape: [N, C, ...outSpatial], dtype: DataType.INT64 });
    }

    return results;
}

// --- Schemas ---
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

    inferShape: (inputs, attrs) => {
        const xShape = toStaticShape(inputs[0]?.shape);
        const wShape = toStaticShape(inputs[1]?.shape);
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        // Conv requires at least [Batch, Channel, Spatial_1]
        if (xShape.length < 3 || wShape.length < 3) {
            return [{ shape: xShape.slice(), dtype }];
        }

        const rank = xShape.length;
        const spatialRank = rank - 2;
        const N = xShape[0];
        const M = wShape[0]; // Output channels

        const strides = (attrs["strides"] as number[]) ?? Array(spatialRank).fill(1);
        const dilations = (attrs["dilations"] as number[]) ?? Array(spatialRank).fill(1);
        let pads = (attrs["pads"] as number[]) ?? Array(spatialRank * 2).fill(0);
        const autoPad = (attrs["auto_pad"] as string) ?? "NOTSET";

        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            pads = Array(spatialRank * 2).fill(0);
            for (let i = 0; i < spatialRank; i++) {
                const dIn = xShape[i + 2];
                const kDim = wShape[i + 2];
                if (dIn === -1 || kDim === -1) continue;

                const kEff = dilations[i] * (kDim - 1) + 1;
                const totalPad = Math.max(
                    0,
                    (Math.ceil(dIn / strides[i]) - 1) * strides[i] + kEff - dIn,
                );

                if (autoPad === "SAME_UPPER") {
                    pads[i] = Math.floor(totalPad / 2); // pad_low
                    pads[i + spatialRank] = totalPad - pads[i]; // pad_high
                } else {
                    pads[i + spatialRank] = Math.floor(totalPad / 2); // pad_low
                    pads[i] = totalPad - pads[i + spatialRank]; // pad_high
                }
            }
        }

        const outSpatial = [];
        for (let i = 0; i < spatialRank; i++) {
            const dIn = xShape[i + 2];
            const kDim = wShape[i + 2];
            if (dIn >= 0 && kDim >= 0) {
                outSpatial.push(
                    inferPoolDim(
                        dIn,
                        kDim,
                        strides[i],
                        pads[i],
                        pads[i + spatialRank],
                        dilations[i],
                        0, // ceilMode is 0 for standard Conv
                    ),
                );
            } else {
                outSpatial.push(-1);
            }
        }

        return [{ shape: [N, M, ...outSpatial], dtype }];
    },
};

export const AveragePool: OpSchema = {
    opType: "AveragePool",
    sinceVersion: 11,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }], // CRITICAL FIX: Only one output port allowed
    attributes: {
        auto_pad: {
            name: "auto_pad",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "NOTSET",
        },
        ceil_mode: { name: "ceil_mode", type: AttributeType.INT, required: false, defaultValue: 0 },
        kernel_shape: { name: "kernel_shape", type: AttributeType.INTS, required: true },
        pads: { name: "pads", type: AttributeType.INTS, required: false },
        strides: { name: "strides", type: AttributeType.INTS, required: false },
        count_include_pad: {
            name: "count_include_pad",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
    inferShape: (inputs, attrs) => inferPoolingShape(inputs, attrs, false),
};

export const MaxPool: OpSchema = {
    opType: "MaxPool",
    sinceVersion: 12,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [
        { name: "Y", typeConstraint: T_ANY },
        { name: "Indices", typeConstraint: "I", optional: true }, // Indices allowed
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
    },
    inferShape: (inputs, attrs) => inferPoolingShape(inputs, attrs, true),
};

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

        inferShape: (inputs) => {
            const xShape = toStaticShape(inputs[0]?.shape);
            const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
            if (xShape.length < 2) return [{ shape: xShape.slice(), dtype }];

            // Global pooling reduces all spatial dimensions to 1
            const outShape = xShape.map((dim, i) => (i < 2 ? dim : 1));
            return [{ shape: outShape, dtype }];
        },
    }),
);

export const Resize: OpSchema = {
    opType: "Resize",
    sinceVersion: 11, // Opset 11 moved sizes/scales to inputs
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "roi", typeConstraint: T_ANY, optional: true },
        { name: "scales", typeConstraint: T_ANY, optional: true },
        { name: "sizes", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        coordinate_transformation_mode: {
            name: "coordinate_transformation_mode",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "half_pixel",
        },
        mode: {
            name: "mode",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "nearest",
        },
        nearest_mode: {
            name: "nearest_mode",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "round_prefer_floor",
        },
    },

    inferShape: (inputs) => {
        const xShape = inputs[0]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        const scales = inputs[2]?.constantValue;
        const sizes = inputs[3]?.constantValue;

        if (sizes && sizes.length > 0) {
            return [{ shape: sizes, dtype }];
        }

        if (scales && scales.length > 0 && xShape.length === scales.length) {
            const outShape = xShape.map((d, i) => {
                if (typeof d === "number") return Math.floor(d * scales[i]);
                return -1;
            });
            return [{ shape: outShape, dtype }];
        }

        // Unknown shape fallback
        return [{ shape: xShape.map(() => -1), dtype }];
    },
};

export const ConvTranspose: OpSchema = {
    opType: "ConvTranspose",
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
        output_padding: { name: "output_padding", type: AttributeType.INTS, required: false },
        output_shape: { name: "output_shape", type: AttributeType.INTS, required: false },
        pads: { name: "pads", type: AttributeType.INTS, required: false },
        strides: { name: "strides", type: AttributeType.INTS, required: false },
    },
    inferShape: (inputs, attrs) => {
        const xShape = toStaticShape(inputs[0]?.shape);
        const wShape = toStaticShape(inputs[1]?.shape);
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        if (xShape.length < 4 || wShape.length < 4) return [{ shape: [], dtype }];

        const [N, , H, W] = xShape;
        const [, M_over_group, kH, kW] = wShape; // W is [C, M/group, kH, kW]
        const group = (attrs["group"] as number) ?? 1;
        const M = M_over_group * group;

        // If output_shape is explicitly provided, use it directly
        const outShapeAttr = attrs["output_shape"] as number[] | undefined;
        if (outShapeAttr && outShapeAttr.length === 2) {
            return [{ shape: [N, M, outShapeAttr[0], outShapeAttr[1]], dtype }];
        }

        const strides = (attrs["strides"] as number[]) ?? [1, 1];
        const dilations = (attrs["dilations"] as number[]) ?? [1, 1];
        const pads = (attrs["pads"] as number[]) ?? [0, 0, 0, 0];
        const outPads = (attrs["output_padding"] as number[]) ?? [0, 0];

        // Formula: H_out = (H_in - 1) * stride - pad_top - pad_bottom + effective_kernel + out_padding
        const kEffH = dilations[0] * (kH - 1) + 1;
        const kEffW = dilations[1] * (kW - 1) + 1;

        const H_out = (H - 1) * strides[0] - (pads[0] ?? 0) - (pads[2] ?? 0) + kEffH + outPads[0];
        const W_out = (W - 1) * strides[1] - (pads[1] ?? 0) - (pads[3] ?? 0) + kEffW + outPads[1];

        return [{ shape: [N, M, H_out, W_out], dtype }];
    },
};

export const LpPool: OpSchema = {
    opType: "LpPool",
    sinceVersion: 12,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        auto_pad: {
            name: "auto_pad",
            type: AttributeType.STRING,
            required: false,
            defaultValue: "NOTSET",
        },
        dilations: { name: "dilations", type: AttributeType.INTS, required: false },
        kernel_shape: { name: "kernel_shape", type: AttributeType.INTS, required: true },
        pads: { name: "pads", type: AttributeType.INTS, required: false },
        strides: { name: "strides", type: AttributeType.INTS, required: false },
        p: { name: "p", type: AttributeType.INT, required: false, defaultValue: 2 },
    },
    inferShape: (inputs, attrs) => {
        // Similar to average pool (You can implement inferPoolDim logic here)
        return [{ shape: [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED }];
    },
};

export const GlobalLpPool: OpSchema = {
    opType: "GlobalLpPool",
    sinceVersion: 2,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [{ name: "X", typeConstraint: T_ANY }],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: { p: { name: "p", type: AttributeType.INT, required: false, defaultValue: 2 } },
    inferShape: (inputs) => {
        const xShape = toStaticShape(inputs[0]?.shape);
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;
        return [{ shape: xShape.map((dim, i) => (i < 2 ? dim : 1)), dtype }];
    },
};

export const PoolingOps: OpSchema[] = [AveragePool, MaxPool];

export const NeuralNetOps: OpSchema[] = [
    Conv,
    ...PoolingOps,
    ...GlobalPoolingOps,
    Resize,
    ConvTranspose,
    LpPool,
    GlobalLpPool,
];
