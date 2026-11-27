import type { AttributeValue } from "../../../ir/OnnxTypes.js";
import { DataType, AttributeType } from "../../../ir/OnnxTypes.js";
import { toStaticShape, inferPoolDim, resolveAutoPads } from "../../../utils/index.js";
import type { OpSchema, TensorInfo } from "../../OpSchema.js";
import { OpCategory, T_ANY, T_INT } from "../../OpSchema.js";

// --- Helper for Shared Pooling Inference ---
function inferPoolingShape(
    inputs: TensorInfo[],
    attrs: Record<string, AttributeValue>,
    isMaxPool: boolean,
) {
    const xShape = toStaticShape(inputs[0]?.shape);
    const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

    // Pooling requires at least [Batch, Channel, Spatial_1]
    if (xShape.length < 3) return [{ shape: xShape.slice(), dtype }];

    const rank = xShape.length;
    const spatialRank = rank - 2;
    const [N, C] = xShape;

    const kernel = (
        "kernel_shape" in attrs ? (attrs["kernel_shape"] as number[]) : Array(spatialRank).fill(1)
    ).map(Number);
    const strides = (
        "strides" in attrs ? (attrs["strides"] as number[]) : Array(spatialRank).fill(1)
    ).map(Number);
    const ceilMode = Number(attrs["ceil_mode"] ?? 0);
    const autoPad = "auto_pad" in attrs ? (attrs["auto_pad"] as string) : "NOTSET";
    let pads = ("pads" in attrs ? (attrs["pads"] as number[]) : Array(spatialRank * 2).fill(0)).map(
        Number,
    );

    // MaxPool supports dilations natively. AveragePool does not (until Opset 19).
    const dilations = isMaxPool
        ? ("dilations" in attrs
              ? (attrs["dilations"] as number[])
              : Array(spatialRank).fill(1)
          ).map(Number)
        : Array(spatialRank).fill(1);

    // Calculate effective padding if auto_pad is used
    if (xShape.slice(2).every((d) => d >= 0)) {
        pads = resolveAutoPads(autoPad, xShape.slice(2), kernel, strides, dilations, pads);
    } else if (autoPad === "VALID") {
        pads = Array(spatialRank * 2).fill(0);
    } else if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
        pads = Array(spatialRank * 2).fill(0);
        for (let i = 0; i < spatialRank; i++) {
            const dIn = xShape[i + 2] as number;
            if (dIn === -1) continue;

            const [head, tail] = resolveAutoPads(
                autoPad,
                [dIn],
                [kernel[i]],
                [strides[i]],
                [dilations[i]],
            );
            pads[i] = head;
            pads[i + spatialRank] = tail;
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

        const strides =
            "strides" in attrs ? (attrs["strides"] as number[]) : Array(spatialRank).fill(1);
        const dilations =
            "dilations" in attrs ? (attrs["dilations"] as number[]) : Array(spatialRank).fill(1);
        let pads = "pads" in attrs ? (attrs["pads"] as number[]) : Array(spatialRank * 2).fill(0);
        const autoPad = "auto_pad" in attrs ? (attrs["auto_pad"] as string) : "NOTSET";

        if (xShape.slice(2).every((d) => d >= 0) && wShape.slice(2).every((d) => d >= 0)) {
            pads = resolveAutoPads(
                autoPad,
                xShape.slice(2),
                wShape.slice(2),
                strides,
                dilations,
                pads,
            );
        } else if (autoPad === "VALID") {
            pads = Array(spatialRank * 2).fill(0);
        } else if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            pads = Array(spatialRank * 2).fill(0);
            for (let i = 0; i < spatialRank; i++) {
                const dIn = xShape[i + 2];
                const kDim = wShape[i + 2];
                if (dIn === -1 || kDim === -1) continue;

                const [head, tail] = resolveAutoPads(
                    autoPad,
                    [dIn],
                    [kDim],
                    [strides[i]],
                    [dilations[i]],
                );
                pads[i] = head;
                pads[i + spatialRank] = tail;
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
        cubic_coeff_a: {
            name: "cubic_coeff_a",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: -0.75,
        },
        exclude_outside: {
            name: "exclude_outside",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        extrapolation_value: {
            name: "extrapolation_value",
            type: AttributeType.FLOAT,
            required: false,
            defaultValue: 0.0,
        },
        antialias: {
            name: "antialias",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
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

        if (xShape.length < 3 || wShape.length < 3 || xShape.length !== wShape.length) {
            return [{ shape: [], dtype }];
        }

        const spatialRank = xShape.length - 2;
        const N = xShape[0];
        const M_over_group = wShape[1]; // W is [C, M/group, ...kernel]
        const group = "group" in attrs ? (attrs["group"] as number) : 1;
        const M = M_over_group >= 0 ? M_over_group * group : -1;

        // If output_shape is explicitly provided, use it directly
        const outShapeAttr = attrs["output_shape"] as number[] | undefined;
        if (outShapeAttr && outShapeAttr.length === spatialRank) {
            return [{ shape: [N, M, ...outShapeAttr], dtype }];
        }

        const strides =
            "strides" in attrs ? (attrs["strides"] as number[]) : Array(spatialRank).fill(1);
        const autoPad = "auto_pad" in attrs ? (attrs["auto_pad"] as string) : "NOTSET";

        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            const outSpatial = xShape.slice(2).map((d, i) => (d >= 0 ? d * (strides[i] ?? 1) : -1));
            return [{ shape: [N, M, ...outSpatial], dtype }];
        }

        const dilations =
            "dilations" in attrs ? (attrs["dilations"] as number[]) : Array(spatialRank).fill(1);
        const pads =
            autoPad === "VALID"
                ? Array(spatialRank * 2).fill(0)
                : "pads" in attrs
                  ? (attrs["pads"] as number[])
                  : Array(spatialRank * 2).fill(0);
        const outPads =
            "output_padding" in attrs
                ? (attrs["output_padding"] as number[])
                : Array(spatialRank).fill(0);

        const outSpatial = xShape.slice(2).map((dIn, i) => {
            const k = wShape[2 + i];
            if (dIn < 0 || k < 0) return -1;
            const kEff = (dilations[i] ?? 1) * (k - 1) + 1;
            return (
                (dIn - 1) * (strides[i] ?? 1) -
                (pads[i] ?? 0) -
                (pads[spatialRank + i] ?? 0) +
                kEff +
                (outPads[i] ?? 0)
            );
        });

        return [{ shape: [N, M, ...outSpatial], dtype }];
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
    inferShape: (inputs) => {
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

export const RotaryEmbedding: OpSchema = {
    opType: "RotaryEmbedding",
    sinceVersion: 23,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "X", typeConstraint: T_ANY },
        { name: "cos_cache", typeConstraint: T_ANY },
        { name: "sin_cache", typeConstraint: T_ANY },
        { name: "position_ids", typeConstraint: T_INT, optional: true },
    ],
    outputs: [{ name: "Y", typeConstraint: T_ANY }],
    attributes: {
        interleaved: {
            name: "interleaved",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        rotary_embedding_dim: {
            name: "rotary_embedding_dim",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
        num_heads: {
            name: "num_heads",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
    inferShape: (inputs) => [
        { shape: inputs[0]?.shape?.slice() ?? [], dtype: inputs[0]?.dtype ?? DataType.UNDEFINED },
    ],
};

export const Attention: OpSchema = {
    opType: "Attention",
    sinceVersion: 23,
    category: OpCategory.Spatial,
    broadcastable: false,
    hasState: false,
    inputs: [
        { name: "Q", typeConstraint: T_ANY },
        { name: "K", typeConstraint: T_ANY },
        { name: "V", typeConstraint: T_ANY },
        { name: "attn_mask", typeConstraint: T_ANY, optional: true },
        { name: "past_key", typeConstraint: T_ANY, optional: true },
        { name: "past_value", typeConstraint: T_ANY, optional: true },
        { name: "nonpad_kv_seqlen", typeConstraint: T_INT, optional: true },
    ],
    outputs: [
        { name: "Y", typeConstraint: T_ANY },
        { name: "present_key", typeConstraint: T_ANY, optional: true },
        { name: "present_value", typeConstraint: T_ANY, optional: true },
        { name: "qk_matmul_output", typeConstraint: T_ANY, optional: true },
    ],
    attributes: {
        is_causal: { name: "is_causal", type: AttributeType.INT, required: false, defaultValue: 0 },
        scale: { name: "scale", type: AttributeType.FLOAT, required: false },
        softcap: { name: "softcap", type: AttributeType.FLOAT, required: false, defaultValue: 0.0 },
        kv_num_heads: { name: "kv_num_heads", type: AttributeType.INT, required: false },
        q_num_heads: { name: "q_num_heads", type: AttributeType.INT, required: false },
        softmax_precision: { name: "softmax_precision", type: AttributeType.INT, required: false },
        qk_matmul_output_mode: {
            name: "qk_matmul_output_mode",
            type: AttributeType.INT,
            required: false,
            defaultValue: 0,
        },
    },
    inferShape: (inputs, attrs) => {
        const qShape = inputs[0]?.shape ?? [];
        const kShape = inputs[1]?.shape ?? [];
        const vShape = inputs[2]?.shape ?? [];
        const pastKShape = inputs[4]?.shape ?? [];
        const dtype = inputs[0]?.dtype ?? DataType.UNDEFINED;

        let yShape = qShape.slice();
        if (qShape.length === 4 && vShape.length === 4) {
            yShape = [qShape[0], qShape[1], qShape[2], vShape[3]];
        } else if (qShape.length === 3 && vShape.length === 3) {
            const qHeads = "q_num_heads" in attrs ? Number(attrs["q_num_heads"]) : 0;
            const kvHeads = "kv_num_heads" in attrs ? Number(attrs["kv_num_heads"]) : 0;
            const vHidden = typeof vShape[2] === "number" ? (vShape[2] as number) : -1;
            const outHidden =
                qHeads > 0 && kvHeads > 0 && vHidden > 0
                    ? Math.floor((vHidden / kvHeads) * qHeads)
                    : vShape[2];
            yShape = [qShape[0], qShape[1], outHidden];
        }

        const qHeadsAttr = "q_num_heads" in attrs ? Number(attrs["q_num_heads"]) : 0;
        const kvHeadsAttr = "kv_num_heads" in attrs ? Number(attrs["kv_num_heads"]) : 0;
        const kvHeads =
            kvHeadsAttr > 0 ? kvHeadsAttr : kShape.length === 4 ? (kShape[1] ?? -1) : -1;
        const kBatch = kShape[0] ?? -1;
        const curKvSeq =
            kShape.length === 4 ? (kShape[2] ?? -1) : kShape.length === 3 ? (kShape[1] ?? -1) : -1;
        const pastKvSeq = pastKShape.length === 4 ? (pastKShape[2] ?? 0) : 0;
        const totalKvSeq =
            typeof curKvSeq === "number" &&
            curKvSeq >= 0 &&
            typeof pastKvSeq === "number" &&
            pastKvSeq >= 0
                ? pastKvSeq + curKvSeq
                : -1;

        const kHeadDim =
            kShape.length === 4
                ? (kShape[3] ?? -1)
                : kShape.length === 3 &&
                    typeof kShape[2] === "number" &&
                    kShape[2] > 0 &&
                    kvHeadsAttr > 0
                  ? Math.floor((kShape[2] as number) / kvHeadsAttr)
                  : -1;
        const vHeadDim =
            vShape.length === 4
                ? (vShape[3] ?? -1)
                : vShape.length === 3 &&
                    typeof vShape[2] === "number" &&
                    vShape[2] > 0 &&
                    kvHeadsAttr > 0
                  ? Math.floor((vShape[2] as number) / kvHeadsAttr)
                  : -1;

        const presentKeyShape = [kBatch, kvHeads, totalKvSeq, kHeadDim];
        const presentValShape = [vShape[0] ?? kBatch, kvHeads, totalKvSeq, vHeadDim];

        const qkShape =
            qShape.length === 4
                ? [qShape[0], qShape[1], qShape[2], totalKvSeq]
                : [qShape[0] ?? -1, qHeadsAttr > 0 ? qHeadsAttr : -1, qShape[1] ?? -1, totalKvSeq];

        return [
            { shape: yShape, dtype },
            { shape: presentKeyShape, dtype: inputs[1]?.dtype ?? dtype },
            { shape: presentValShape, dtype: inputs[2]?.dtype ?? dtype },
            { shape: qkShape, dtype },
        ];
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
    RotaryEmbedding,
    Attention,
];
