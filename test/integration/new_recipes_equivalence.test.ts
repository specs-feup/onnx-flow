import { describe, it, expect, beforeAll, afterAll } from "vitest";
import * as ort from "onnxruntime-web";
import fs from "fs";
import os from "os";
import path from "path";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../src/Onnx/GraphBuilder.js";
import type { KnownShape } from "../../src/Onnx/OnnxTypes.js";
import { DataType, type AttributeMap, type TensorProto } from "../../src/Onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../src/Onnx/Schema/index.js";
import inferShapes from "../../src/Onnx/InferShapes.js";
import { convertFlowGraphToOnnxJson, prepareGraphForExport } from "../../src/flow2json.js";
import { json2onnx } from "../../src/json2onnx.js";
import {
    safeWriteJson,
    int64Vec,
    scalarInt64,
    floatVec,
    makeTensorProto,
} from "../../src/Onnx/Utils.js";
import { CanonicalizationPass } from "../../src/Onnx/transformation/canonicalization/CanonicalizationPass.js";
import { LoopLoweringPass } from "../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { HistoryManager } from "../../src/Onnx/transformation/tracking/HistoryManager.js";
import { IndexingMode } from "../../src/DecompositionOptions.js";
import {
    buildFeeds,
    compareOrtOutputs,
    ensureProtoExists,
    type FeedSpec,
} from "../compatibility_test.js";

interface SyntheticCase {
    name: string;
    opType: string;
    feeds: FeedSpec[];
    constInputs?: { index: number; proto: TensorProto }[];
    inputShapes?: KnownShape[];
    attrs?: AttributeMap;
    outputNames?: string[];
    outputSpecs?: { type: DataType; shape: KnownShape }[];
    tol?: number | Record<string, number>;
    rtol?: number;
    exact?: boolean;
    indexingModes?: IndexingMode[];
    /** Optional ORT graph optimization level override (defaults to ORT's default when omitted) */
    graphOptimizationLevel?: ort.InferenceSession.SessionOptions["graphOptimizationLevel"];
    /** Skip LoopLoweringPass and verify canonicalized graph directly in ORT (for very large unrolled canonicalizations) */
    skipLoopLowering?: boolean;
}

const CASES: SyntheticCase[] = [
    // Layer A
    {
        name: "Neg",
        opType: "Neg",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" }],
        tol: 1e-6,
    },
    {
        name: "Floor",
        opType: "Floor",
        feeds: [
            { name: "X", dtype: "float32", shape: [6], init: [-2.7, -2.0, -0.1, 0.0, 0.9, 3.4] },
        ],
        tol: 1e-6,
    },
    {
        name: "Reciprocal",
        opType: "Reciprocal",
        feeds: [
            {
                name: "X",
                dtype: "float32",
                shape: [6],
                init: [-4.0, -0.25, 0.125, 1.0, 3.0, 100.0],
            },
        ],
        tol: 1e-4,
    },
    {
        name: "Exp",
        opType: "Exp",
        feeds: [
            { name: "X", dtype: "float32", shape: [6], init: [-5.0, -1.0, 0.0, 0.5, 2.0, 8.0] },
        ],
        tol: 1e-5,
        rtol: 1e-5,
    },

    // Layer B
    {
        name: "PRelu",
        opType: "PRelu",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "S", dtype: "float32", shape: [3], init: [0.1, 0.25, 0.5] },
        ],
        tol: 1e-5,
    },
    {
        name: "HardSwish",
        opType: "HardSwish",
        feeds: [{ name: "X", dtype: "float32", shape: [6], init: [-4, -3, -1, 0, 2, 5] }],
        tol: 1e-5,
    },
    {
        name: "ThresholdedRelu",
        opType: "ThresholdedRelu",
        feeds: [{ name: "X", dtype: "float32", shape: [6], gen: "negmix" }],
        attrs: { alpha: 0.25 },
        tol: 1e-6,
    },
    {
        name: "Shrink",
        opType: "Shrink",
        feeds: [
            { name: "X", dtype: "float32", shape: [6], init: [-2.0, -0.4, 0.0, 0.4, 0.6, 2.0] },
        ],
        attrs: { bias: 0.1, lambd: 0.5 },
        tol: 1e-6,
    },
    {
        name: "Sum",
        opType: "Sum",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [2, 3] },
            { name: "C", dtype: "float32", shape: [2, 3] },
        ],
        tol: 1e-5,
    },
    {
        name: "Mean",
        opType: "Mean",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [2, 3] },
        ],
        tol: 1e-5,
    },
    {
        name: "CastLike",
        opType: "CastLike",
        feeds: [
            { name: "X", dtype: "int32", shape: [2, 3] },
            { name: "T", dtype: "float32", shape: [2, 3] },
        ],
        tol: 1e-6,
    },
    {
        name: "Gelu_none",
        opType: "Gelu",
        feeds: [{ name: "X", dtype: "float32", shape: [6], gen: "negmix" }],
        attrs: { approximate: "none" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [6] }],
        tol: 1e-3,
    },
    {
        name: "Gelu_tanh",
        opType: "Gelu",
        feeds: [{ name: "X", dtype: "float32", shape: [6], gen: "negmix" }],
        attrs: { approximate: "tanh" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [6] }],
        tol: 1e-3,
    },
    {
        name: "DynamicQuantizeLinear",
        opType: "DynamicQuantizeLinear",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        outputNames: ["Y", "Y_scale", "Y_zp"],
        tol: { Y: 1, Y_scale: 1e-5, Y_zp: 1 },
    },

    // Tier 1
    {
        name: "Flatten",
        opType: "Flatten",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
        attrs: { axis: 1 },
        tol: 1e-6,
    },
    {
        name: "Split",
        opType: "Split",
        feeds: [{ name: "X", dtype: "float32", shape: [4, 3] }],
        constInputs: [{ index: 1, proto: int64Vec([1, 3]) }],
        attrs: { axis: 0 },
        outputNames: ["Y0", "Y1"],
        tol: 1e-6,
    },
    {
        name: "Tile",
        opType: "Tile",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3] }],
        constInputs: [{ index: 1, proto: int64Vec([2, 2]) }],
        tol: 1e-6,
    },
    {
        name: "LogSoftmax",
        opType: "LogSoftmax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        attrs: { axis: -1 },
        tol: 1e-3,
    },
    {
        name: "Hardmax",
        opType: "Hardmax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [1, 5, 2, 3, -1, -2, 0, -5] }],
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        attrs: { axis: -1 },
        tol: 1e-6,
    },
    {
        name: "LpNormalization_L2",
        opType: "LpNormalization",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [3, 4, 0, 0, 1, 2, 2, 0] }],
        attrs: { p: 2, axis: -1 },
        tol: 1e-4,
    },
    {
        name: "GlobalAveragePool",
        opType: "GlobalAveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 3, 3] }],
        tol: 1e-4,
    },
    {
        name: "GlobalMaxPool",
        opType: "GlobalMaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 3, 3], gen: "negmix" }],
        tol: 1e-6,
    },
    {
        name: "GlobalLpPool_L2",
        opType: "GlobalLpPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 3, 3], gen: "negmix" }],
        attrs: { p: 2 },
        tol: 1e-4,
    },
    {
        name: "EyeLike",
        opType: "EyeLike",
        feeds: [{ name: "X", dtype: "float32", shape: [3, 4] }],
        attrs: { k: 1 },
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-6,
    },
    {
        name: "OneHot",
        opType: "OneHot",
        feeds: [{ name: "I", dtype: "int64", shape: [2, 2], init: [0n, 2n, -1n, 1n] }],
        constInputs: [
            { index: 1, proto: scalarInt64(4) },
            { index: 2, proto: makeTensorProto(DataType.FLOAT, [2], [0, 1]) },
        ],
        tol: 1e-6,
    },
    {
        name: "Trilu_upper",
        opType: "Trilu",
        feeds: [{ name: "X", dtype: "float32", shape: [3, 3] }],
        constInputs: [{ index: 1, proto: scalarInt64(0) }],
        attrs: { upper: 1 },
        tol: 1e-6,
    },
    {
        name: "MatMulInteger",
        opType: "MatMulInteger",
        feeds: [
            { name: "A", dtype: "uint8", shape: [2, 3], init: [1, 2, 3, 4, 5, 6] },
            { name: "B", dtype: "uint8", shape: [3, 2], init: [1, 0, 2, 1, 0, 3] },
        ],
        exact: true,
    },

    // Tier 2
    {
        name: "LpPool_L2",
        opType: "LpPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [2, 2], strides: [2, 2], pads: [0, 0, 0, 0], p: 2 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2] }],
        tol: 1e-4,
    },
    {
        name: "NLLLoss",
        opType: "NegativeLogLikelihoodLoss",
        feeds: [
            { name: "X", dtype: "float32", shape: [3, 4], gen: "negmix" },
            { name: "T", dtype: "int64", shape: [3], init: [0n, 2n, 3n] },
        ],
        attrs: { reduction: "mean" },
        tol: 1e-4,
    },
    {
        name: "SoftmaxCrossEntropyLoss",
        opType: "SoftmaxCrossEntropyLoss",
        feeds: [
            { name: "X", dtype: "float32", shape: [3, 4], gen: "negmix" },
            { name: "T", dtype: "int64", shape: [3], init: [1n, 0n, 3n] },
        ],
        attrs: { reduction: "mean" },
        outputNames: ["loss", "log_prob"],
        tol: 1e-3,
    },
    {
        name: "RNN",
        opType: "RNN",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 1, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [1, 2, 3], gen: "negmix" },
            { name: "R", dtype: "float32", shape: [1, 2, 2], gen: "negmix" },
        ],
        attrs: { hidden_size: 2 },
        outputNames: ["Y", "Y_h"],
        tol: 1e-4,
    },
    {
        name: "GRU",
        opType: "GRU",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 1, 2], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [1, 6, 2], gen: "negmix" },
            { name: "R", dtype: "float32", shape: [1, 6, 2], gen: "negmix" },
        ],
        attrs: { hidden_size: 2 },
        outputNames: ["Y", "Y_h"],
        tol: 1e-4,
    },
    {
        name: "LSTM",
        opType: "LSTM",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 1, 2], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [1, 8, 2], gen: "negmix" },
            { name: "R", dtype: "float32", shape: [1, 8, 2], gen: "negmix" },
        ],
        attrs: { hidden_size: 2 },
        outputNames: ["Y", "Y_h", "Y_c"],
        tol: 1e-4,
    },

    // Tier 3 (Loop Lowering — tested in both Pointer and Radix modes)
    {
        name: "CumSum_inc",
        opType: "CumSum",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        constInputs: [{ index: 1, proto: scalarInt64(1) }],
        attrs: { exclusive: 0, reverse: 0 },
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "CumSum_exc_rev",
        opType: "CumSum",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        constInputs: [{ index: 1, proto: scalarInt64(1) }],
        attrs: { exclusive: 1, reverse: 1 },
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ReverseSequence",
        opType: "ReverseSequence",
        feeds: [{ name: "X", dtype: "float32", shape: [4, 2], gen: "negmix" }],
        constInputs: [{ index: 1, proto: int64Vec([2, 4]) }],
        attrs: { time_axis: 0, batch_axis: 1 },
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "GatherElements",
        opType: "GatherElements",
        feeds: [
            { name: "D", dtype: "float32", shape: [2, 4] },
            { name: "I", dtype: "int64", shape: [2, 2], init: [0n, 3n, -1n, 1n] },
        ],
        attrs: { axis: 1 },
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "GatherND",
        opType: "GatherND",
        feeds: [
            { name: "D", dtype: "float32", shape: [2, 3, 2] },
            { name: "I", dtype: "int64", shape: [2, 2], init: [0n, 1n, 1n, -1n] },
        ],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPool",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [2, 2], strides: [2, 2], pads: [0, 0, 0, 0] },
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ScatterND",
        opType: "ScatterND",
        feeds: [
            { name: "D", dtype: "float32", shape: [4, 2] },
            { name: "I", dtype: "int64", shape: [2, 1], init: [1n, 3n] },
            { name: "U", dtype: "float32", shape: [2, 2] },
        ],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ConvTranspose",
        opType: "ConvTranspose",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 3, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 2, 2], gen: "negmix" },
        ],
        attrs: { kernel_shape: [2, 2], strides: [1, 1], pads: [0, 0, 0, 0] },
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Resize_nearest",
        opType: "Resize",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 1, 2, 2] }],
        constInputs: [
            { index: 1, proto: floatVec([]) },
            { index: 2, proto: floatVec([1, 1, 2, 2]) },
        ],
        attrs: { mode: "nearest" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 1, 4, 4] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Resize_linear",
        opType: "Resize",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 1, 2, 2] }],
        constInputs: [
            { index: 1, proto: floatVec([]) },
            { index: 2, proto: floatVec([1, 1, 2, 2]) },
        ],
        attrs: { mode: "linear" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 1, 4, 4] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ArgMax_first",
        opType: "ArgMax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [1, 5, 5, 2, -3, -1, -1, -4] }],
        attrs: { axis: -1, keepdims: 1, select_last_index: 0 },
        outputSpecs: [{ type: DataType.INT64, shape: [2, 1] }],
        exact: true,
    },
    {
        name: "ArgMax_last_nokeep",
        opType: "ArgMax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [1, 5, 5, 2, -3, -1, -1, -4] }],
        attrs: { axis: 1, keepdims: 0, select_last_index: 1 },
        outputSpecs: [{ type: DataType.INT64, shape: [2] }],
        exact: true,
    },
    {
        name: "ArgMin_first",
        opType: "ArgMin",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [4, 1, 1, 3, 2, -5, 0, -5] }],
        attrs: { axis: 1, keepdims: 1, select_last_index: 0 },
        outputSpecs: [{ type: DataType.INT64, shape: [2, 1] }],
        exact: true,
    },
    {
        name: "ArgMin_last_nokeep",
        opType: "ArgMin",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [4, 1, 1, 3, 2, -5, 0, -5] }],
        attrs: { axis: 0, keepdims: 0, select_last_index: 1 },
        outputSpecs: [{ type: DataType.INT64, shape: [4] }],
        exact: true,
    },

    {
        name: "TopK_largest",
        opType: "TopK",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 5], init: [3, 1, 5, 5, 2, -1, -4, 0, 2, 2] },
        ],
        constInputs: [{ index: 1, proto: int64Vec([3]) }],
        attrs: { axis: 1, largest: 1, sorted: 1 },
        outputNames: ["Values", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [2, 3] },
            { type: DataType.INT64, shape: [2, 3] },
        ],
        exact: true,
    },
    {
        name: "TopK_smallest",
        opType: "TopK",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 5], init: [3, 1, 5, 1, 2, -1, -4, 0, -4, 2] },
        ],
        constInputs: [{ index: 1, proto: int64Vec([2]) }],
        attrs: { axis: -1, largest: 0, sorted: 1 },
        outputNames: ["Values", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [2, 2] },
            { type: DataType.INT64, shape: [2, 2] },
        ],
        exact: true,
    },
    {
        name: "Einsum_matmul",
        opType: "Einsum",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [3, 4], gen: "negmix" },
        ],
        attrs: { equation: "ij,jk->ik" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 4] }],
        tol: 1e-4,
    },
    {
        name: "Einsum_batched_attn",
        opType: "Einsum",
        feeds: [
            { name: "Q", dtype: "float32", shape: [1, 2, 3, 4], gen: "negmix" },
            { name: "K", dtype: "float32", shape: [1, 2, 3, 4], gen: "negmix" },
        ],
        attrs: { equation: "bhid,bhjd->bhij" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 3, 3] }],
        tol: 1e-4,
    },
    {
        name: "Einsum_trace",
        opType: "Einsum",
        feeds: [{ name: "X", dtype: "float32", shape: [3, 3], gen: "negmix" }],
        attrs: { equation: "ii->" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [] }],
        tol: 1e-4,
    },
    {
        name: "Einsum_outer",
        opType: "Einsum",
        feeds: [
            { name: "A", dtype: "float32", shape: [3], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [4], gen: "negmix" },
        ],
        attrs: { equation: "i,j->ij" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [3, 4] }],
        tol: 1e-4,
    },
    {
        name: "Det_2x2",
        opType: "Det",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 2], init: [3, 8, 4, 6] }],
        outputSpecs: [{ type: DataType.FLOAT, shape: [] }],
        tol: 1e-4,
    },
    {
        name: "Det_3x3_batched",
        opType: "Det",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 3], gen: "negmix" }],
        outputSpecs: [{ type: DataType.FLOAT, shape: [2] }],
        tol: 1e-4,
    },
    {
        name: "Det_4x4",
        opType: "Det",
        feeds: [
            {
                name: "X",
                dtype: "float32",
                shape: [4, 4],
                init: [0, 2, 1, 3, 4, 1, 5, 2, 2, 3, 0, 1, 1, 0, 2, 4],
            },
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [] }],
        tol: 1e-3,
    },
    {
        name: "AveragePool_1D_pad_excl",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 5], gen: "negmix" }],
        attrs: { kernel_shape: [3], strides: [2], pads: [1, 1], count_include_pad: 0 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 3] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "AveragePool_2D_non_overlap",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [2, 2], strides: [2, 2], pads: [0, 0, 0, 0] },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "AveragePool_3D",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [2, 2, 2], strides: [2, 2, 2], pads: [0, 0, 0, 0, 0, 0] },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2, 2] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Conv_1D_grouped_bias",
        opType: "Conv",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 4, 6], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [4, 2, 3], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [4], gen: "negmix" },
        ],
        attrs: { group: 2, strides: [1], pads: [1, 1] },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 4, 6] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Conv_3D",
        opType: "Conv",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 3, 3, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 2, 2, 2], gen: "negmix" },
        ],
        attrs: { strides: [1, 1, 1], pads: [0, 0, 0, 0, 0, 0] },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2, 2] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },

    {
        name: "RNNLoop_bidir_seqlen_clip",
        opType: "RNN",
        feeds: [
            { name: "X", dtype: "float32", shape: [4, 2, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3], gen: "negmix" },
            { name: "R", dtype: "float32", shape: [2, 2, 2], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [2, 4], gen: "negmix" },
            { name: "seq_lens", dtype: "int32", shape: [2], init: [2, 4] },
        ],
        attrs: { hidden_size: 2, direction: "bidirectional", clip: 1.5 },
        outputNames: ["Y", "Y_h"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [4, 2, 2, 2] },
            { type: DataType.FLOAT, shape: [2, 2, 2] },
        ],
        tol: 1e-4,
    },
    {
        name: "GRULoop_reverse_seqlen",
        opType: "GRU",
        feeds: [
            { name: "X", dtype: "float32", shape: [3, 2, 2], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [1, 6, 2], gen: "negmix" },
            { name: "R", dtype: "float32", shape: [1, 6, 2], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [1, 12], gen: "negmix" },
            { name: "seq_lens", dtype: "int32", shape: [2], init: [1, 3] },
        ],
        attrs: { hidden_size: 2, direction: "reverse", linear_before_reset: 1 },
        outputNames: ["Y", "Y_h"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [3, 1, 2, 2] },
            { type: DataType.FLOAT, shape: [1, 2, 2] },
        ],
        tol: 1e-4,
    },
    {
        name: "LSTMLoop_seqlen_clip",
        opType: "LSTM",
        feeds: [
            { name: "X", dtype: "float32", shape: [3, 2, 2], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [1, 8, 2], gen: "negmix" },
            { name: "R", dtype: "float32", shape: [1, 8, 2], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [1, 16], gen: "negmix" },
            { name: "seq_lens", dtype: "int32", shape: [2], init: [2, 3] },
        ],
        attrs: { hidden_size: 2, clip: 2.0 },
        outputNames: ["Y", "Y_h", "Y_c"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [3, 1, 2, 2] },
            { type: DataType.FLOAT, shape: [1, 2, 2] },
            { type: DataType.FLOAT, shape: [1, 2, 2] },
        ],
        tol: 1e-4,
    },
    {
        name: "SliceLoop_dynamic_rev_step",
        opType: "Slice",
        feeds: [
            { name: "X", dtype: "float32", shape: [4, 5], gen: "negmix" },
            { name: "starts", dtype: "int64", shape: [2], init: [3n, 1n] },
            { name: "ends", dtype: "int64", shape: [2], init: [0n, 5n] },
            { name: "axes", dtype: "int64", shape: [2], init: [0n, 1n] },
            { name: "steps", dtype: "int64", shape: [2], init: [-1n, 2n] },
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [3, 2] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "PadLoop_wrap",
        opType: "Pad",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "pads", dtype: "int64", shape: [4], init: [1n, 2n, 2n, 1n] },
        ],
        attrs: { mode: "wrap" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [5, 6] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "PadLoop_reflect_dynamic",
        opType: "Pad",
        feeds: [
            { name: "X", dtype: "float32", shape: [3, 4], gen: "negmix" },
            { name: "pads", dtype: "int64", shape: [4], init: [2n, 3n, 2n, 3n] },
        ],
        attrs: { mode: "reflect" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [7, 10] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "PadLoop_constant_dynamic",
        opType: "Pad",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "pads", dtype: "int64", shape: [4], init: [1n, 0n, 1n, 2n] },
            { name: "val", dtype: "float32", shape: [], init: [-5.0] },
        ],
        attrs: { mode: "constant" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [4, 5] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "TileLoop_dynamic",
        opType: "Tile",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "repeats", dtype: "int64", shape: [2], init: [2n, 3n] },
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [4, 9] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "SplitDynamic_runtime_split",
        opType: "Split",
        feeds: [
            { name: "X", dtype: "float32", shape: [5, 3], gen: "negmix" },
            { name: "split", dtype: "int64", shape: [2], init: [2n, 3n] },
        ],
        attrs: { axis: 0 },
        outputNames: ["Y0", "Y1"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [2, 3] },
            { type: DataType.FLOAT, shape: [3, 3] },
        ],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "VariadicMin_3in",
        opType: "Min",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "C", dtype: "float32", shape: [2, 3], gen: "negmix" },
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 3] }],
        tol: 1e-6,
    },
    {
        name: "VariadicMax_3in",
        opType: "Max",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [2, 3], gen: "negmix" },
            { name: "C", dtype: "float32", shape: [2, 3], gen: "negmix" },
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 3] }],
        tol: 1e-6,
    },
    {
        name: "OneHotLoop_dynamic_depth",
        opType: "OneHot",
        feeds: [
            { name: "I", dtype: "int64", shape: [2, 2], init: [0n, 2n, -1n, 1n] },
            { name: "depth", dtype: "int64", shape: [], init: [4n] },
            { name: "vals", dtype: "float32", shape: [2], init: [0.0, 1.0] },
        ],
        attrs: { axis: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 4, 2] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "DynamicReduceMean_keepdims1",
        opType: "ReduceMean",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
        inputShapes: [[-1, 3, 4]],
        constInputs: [{ index: 1, proto: int64Vec([1]) }],
        attrs: { keepdims: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 1, 4] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "DynamicReduceMax_keepdims0",
        opType: "ReduceMax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
        inputShapes: [[-1, -1, 4]],
        constInputs: [{ index: 1, proto: int64Vec([0, 2]) }],
        attrs: { keepdims: 0 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ArgMaxLoop_dynamic_axis_first",
        opType: "ArgMax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [1, 5, 5, 2, -3, -1, -1, -4] }],
        inputShapes: [[2, -1]],
        attrs: { axis: 1, keepdims: 1, select_last_index: 0 },
        outputSpecs: [{ type: DataType.INT64, shape: [2, 1] }],
        exact: true,
    },
    {
        name: "ArgMinLoop_dynamic_axis_last_nokeep",
        opType: "ArgMin",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [4, 1, 1, 3, 2, -5, 0, -5] }],
        inputShapes: [[-1, 4]],
        attrs: { axis: 0, keepdims: 0, select_last_index: 1 },
        outputSpecs: [{ type: DataType.INT64, shape: [4] }],
        exact: true,
    },
    {
        name: "TopKLoop_dynamic_k_largest",
        opType: "TopK",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 5], init: [3, 1, 5, 5, 2, -1, -4, 0, 2, 2] },
            { name: "K", dtype: "int64", shape: [1], init: [3n] },
        ],
        inputShapes: [[2, -1], [1]],
        attrs: { axis: 1, largest: 1, sorted: 1 },
        outputNames: ["Values", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [2, -1] },
            { type: DataType.INT64, shape: [2, -1] },
        ],
        exact: true,
    },
    {
        name: "TopKLoop_dynamic_k_smallest_axis0",
        opType: "TopK",
        feeds: [
            { name: "X", dtype: "float32", shape: [4, 2], init: [3, 1, 5, -2, -1, 4, 0, -2] },
            { name: "K", dtype: "int64", shape: [1], init: [2n] },
        ],
        attrs: { axis: 0, largest: 0, sorted: 1 },
        outputNames: ["Values", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [-1, 2] },
            { type: DataType.INT64, shape: [-1, 2] },
        ],
        exact: true,
    },
    {
        name: "ResizeCubic_2d",
        opType: "Resize",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 1, 3, 3], gen: "negmix" }],
        constInputs: [
            { index: 1, proto: floatVec([]) },
            { index: 2, proto: floatVec([1, 1, 2, 2]) },
        ],
        attrs: { mode: "cubic", cubic_coeff_a: -0.75, exclude_outside: 0 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 1, 6, 6] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ResizeCubic_exclude_outside",
        opType: "Resize",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 1, 3, 3], gen: "negmix" }],
        constInputs: [
            { index: 1, proto: floatVec([]) },
            { index: 2, proto: floatVec([1, 1, 2, 2]) },
        ],
        attrs: { mode: "cubic", cubic_coeff_a: -0.75, exclude_outside: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 1, 6, 6] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPoolWithIndices_row_major_padded",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [2, 2], strides: [2, 2], pads: [1, 1, 1, 1], storage_order: 0 },
        outputNames: ["Y", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [1, 2, 3, 3] },
            { type: DataType.INT64, shape: [1, 2, 3, 3] },
        ],
        tol: { Y: 1e-6, Indices: 0 },
    },
    {
        name: "MaxPoolWithIndices_col_major",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [2, 2], strides: [2, 2], pads: [0, 0, 0, 0], storage_order: 1 },
        outputNames: ["Y", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [1, 2, 2, 2] },
            { type: DataType.INT64, shape: [1, 2, 2, 2] },
        ],
        tol: { Y: 1e-6, Indices: 0 },
    },
    {
        name: "DynamicBatchedMatMul_4d_bcast",
        opType: "MatMul",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 1, 3, 4], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [1, 3, 4, 2], gen: "negmix" },
        ],
        inputShapes: [
            [-1, 1, 3, 4],
            [1, -1, 4, 2],
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, 3, 2] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "DynamicBatchedMatMul_3d_2d",
        opType: "MatMul",
        feeds: [
            { name: "A", dtype: "float32", shape: [3, 2, 4], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [4, 3], gen: "negmix" },
        ],
        inputShapes: [
            [-1, -1, 4],
            [4, -1],
        ],
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, -1] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "HardmaxDynamic_dyn_axis",
        opType: "Hardmax",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 5], gen: "negmix" }],
        inputShapes: [[2, -1]],
        attrs: { axis: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, -1] }],
        tol: 1e-6,
    },
    {
        name: "TriluDynamic_lower_k1",
        opType: "Trilu",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" },
            { name: "k", dtype: "int64", shape: [], init: [1n] },
        ],
        inputShapes: [[-1, -1, -1], []],
        attrs: { upper: 0 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, -1] }],
        tol: 1e-6,
    },
    {
        name: "EyeLikeDynamic_k_neg1",
        opType: "EyeLike",
        feeds: [{ name: "X", dtype: "float32", shape: [3, 4], gen: "negmix" }],
        inputShapes: [[-1, -1]],
        attrs: { k: -1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1] }],
        tol: 1e-6,
    },
    {
        name: "DetDynamic_batch_3x3",
        opType: "Det",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 3], gen: "negmix" }],
        inputShapes: [[-1, 3, 3]],
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1] }],
        tol: 1e-4,
    },
    {
        name: "DetDynamic_batch_4x4",
        opType: "Det",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4, 4], gen: "negmix" }],
        inputShapes: [[-1, 4, 4]],
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1] }],
        tol: 1e-4,
        skipLoopLowering: true,
    },
    // Auto-pad / padding coverage across canonicalization & loop-lowering recipes
    {
        name: "Conv_2D_SAME_UPPER_asym",
        opType: "Conv",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 4, 5], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" },
        ],
        attrs: { strides: [2, 2], auto_pad: "SAME_UPPER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 3] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Conv_2D_SAME_LOWER_asym",
        opType: "Conv",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 1, 4, 5], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [1, 1, 3, 3], gen: "negmix" },
        ],
        attrs: { strides: [2, 2], auto_pad: "SAME_LOWER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 1, 2, 3] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Conv_1D_SAME_UPPER",
        opType: "Conv",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 5], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3], gen: "negmix" },
        ],
        attrs: { strides: [2], auto_pad: "SAME_UPPER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 3] }],
        tol: 1e-4,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPool_SAME_UPPER",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 5], gen: "negmix" }],
        attrs: { kernel_shape: [3, 3], strides: [2, 2], auto_pad: "SAME_UPPER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 3] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPool_SAME_LOWER",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 5], gen: "negmix" }],
        attrs: { kernel_shape: [3, 3], strides: [2, 2], auto_pad: "SAME_LOWER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 3] }],
        tol: 1e-6,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPoolWithIndices_SAME_LOWER",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [3, 3], strides: [2, 2], auto_pad: "SAME_LOWER" },
        outputNames: ["Y", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [1, 2, 2, 2] },
            { type: DataType.INT64, shape: [1, 2, 2, 2] },
        ],
        tol: { Y: 1e-6, Indices: 0 },
    },
    {
        name: "AveragePool_2D_VALID_ceil1_excl",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: {
            kernel_shape: [3, 3],
            strides: [2, 2],
            auto_pad: "VALID",
            ceil_mode: 1,
            count_include_pad: 0,
        },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "AveragePool_2D_SAME_UPPER_excl",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: {
            kernel_shape: [3, 3],
            strides: [2, 2],
            auto_pad: "SAME_UPPER",
            count_include_pad: 0,
        },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "AveragePool_1D_SAME_LOWER_excl",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4], gen: "negmix" }],
        attrs: {
            kernel_shape: [3],
            strides: [2],
            auto_pad: "SAME_LOWER",
            count_include_pad: 0,
        },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "AveragePool_1D_VALID_ceil1_excl",
        opType: "AveragePool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4], gen: "negmix" }],
        attrs: {
            kernel_shape: [3],
            strides: [2],
            auto_pad: "VALID",
            ceil_mode: 1,
            count_include_pad: 0,
        },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ConvTranspose_SAME_UPPER",
        opType: "ConvTranspose",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 3, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" },
        ],
        attrs: { strides: [2, 2], auto_pad: "SAME_UPPER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 6, 6] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ConvTranspose_SAME_LOWER",
        opType: "ConvTranspose",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 3, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" },
        ],
        attrs: { strides: [2, 2], auto_pad: "SAME_LOWER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 6, 6] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ConvTranspose_output_shape",
        opType: "ConvTranspose",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 2, 3, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" },
        ],
        attrs: { strides: [2, 2], output_shape: [6, 6] },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 6, 6] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "LpPool_SAME_UPPER",
        opType: "LpPool",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        attrs: { kernel_shape: [3, 3], strides: [2, 2], auto_pad: "SAME_UPPER", p: 2 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 2, 2] }],
        tol: 1e-4,
    },

    // --- Dynamic Shape (-1) Equivalence Tests for Batch 1-3 ---
    {
        name: "CumSum_dynamic_rev",
        opType: "CumSum",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
        inputShapes: [[-1, 3, -1]],
        constInputs: [{ index: 1, proto: scalarInt64(2) }],
        attrs: { exclusive: 0, reverse: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 3, -1] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ReverseSequence_dynamic",
        opType: "ReverseSequence",
        feeds: [
            { name: "X", dtype: "float32", shape: [3, 4, 2], gen: "negmix" },
            { name: "seq_lens", dtype: "int64", shape: [3], init: [2, 4, 1] },
        ],
        inputShapes: [[-1, -1, 2], [-1]],
        attrs: { batch_axis: 0, time_axis: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, 2] }],
        exact: true,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "GatherElements_dynamic",
        opType: "GatherElements",
        feeds: [
            { name: "data", dtype: "float32", shape: [2, 3, 2], gen: "negmix" },
            {
                name: "indices",
                dtype: "int64",
                shape: [2, 2, 2],
                init: [0, -1, 2, 1, 1, 0, -2, 2],
            },
        ],
        inputShapes: [
            [-1, 3, -1],
            [-1, -1, -1],
        ],
        attrs: { axis: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, -1] }],
        exact: true,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "GatherND_dynamic",
        opType: "GatherND",
        feeds: [
            { name: "data", dtype: "float32", shape: [2, 3, 4], gen: "negmix" },
            { name: "indices", dtype: "int64", shape: [2, 2, 1], init: [0, 2, 1, -1] },
        ],
        inputShapes: [
            [-1, 3, -1],
            [-1, -1, 1],
        ],
        attrs: { batch_dims: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, -1] }],
        exact: true,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "ScatterND_dynamic_add",
        opType: "ScatterND",
        feeds: [
            { name: "data", dtype: "float32", shape: [2, 3, 4], gen: "negmix" },
            { name: "indices", dtype: "int64", shape: [2, 2], init: [0, 1, 1, 2] },
            { name: "updates", dtype: "float32", shape: [2, 4], gen: "negmix" },
        ],
        inputShapes: [
            [-1, 3, -1],
            [-1, 2],
            [-1, -1],
        ],
        attrs: { reduction: "add" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 3, -1] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPool_dynamic_SAME_UPPER",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 2, 5, 5], gen: "negmix" }],
        inputShapes: [[-1, 2, -1, -1]],
        attrs: { kernel_shape: [3, 3], strides: [2, 2], auto_pad: "SAME_UPPER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 2, -1, -1] }],
        exact: true,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "MaxPool_WithIndices_dynamic",
        opType: "MaxPool",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 2, 4, 4], gen: "negmix" }],
        inputShapes: [[-1, 2, -1, -1]],
        attrs: { kernel_shape: [2, 2], strides: [2, 2], storage_order: 1 },
        outputNames: ["Y", "Indices"],
        outputSpecs: [
            { type: DataType.FLOAT, shape: [-1, 2, -1, -1] },
            { type: DataType.INT64, shape: [-1, 2, -1, -1] },
        ],
        exact: true,
    },
    {
        name: "ConvTranspose_dynamic_bias",
        opType: "ConvTranspose",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" },
            { name: "W", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [2], gen: "negmix" },
        ],
        inputShapes: [[-1, 2, -1, -1], [2, 2, 3, 3], [2]],
        attrs: { strides: [2, 2], auto_pad: "SAME_UPPER" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 2, -1, -1] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Resize_linear_dynamic_scales",
        opType: "Resize",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" }],
        inputShapes: [[-1, 2, -1, -1]],
        constInputs: [
            { index: 1, proto: floatVec([]) },
            { index: 2, proto: floatVec([1, 1, 2, 2]) },
        ],
        attrs: { mode: "linear", coordinate_transformation_mode: "align_corners" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 2, -1, -1] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Resize_cubic_dynamic_scales",
        opType: "Resize",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 2, 3, 3], gen: "negmix" }],
        inputShapes: [[-1, 2, -1, -1]],
        constInputs: [
            { index: 1, proto: floatVec([]) },
            { index: 2, proto: floatVec([1, 1, 2, 2]) },
        ],
        attrs: { mode: "cubic", exclude_outside: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, 2, -1, -1] }],
        tol: 1e-5,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
    },
    {
        name: "Einsum_dynamic_batch_contract",
        opType: "Einsum",
        feeds: [
            { name: "A", dtype: "float32", shape: [2, 3, 4], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [2, 4, 5], gen: "negmix" },
        ],
        inputShapes: [
            [-1, -1, -1],
            [-1, -1, -1],
        ],
        attrs: { equation: "bij,bjk->bik" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1, -1] }],
        tol: 1e-5,
    },
    {
        name: "Einsum_dynamic_trace_mul",
        opType: "Einsum",
        feeds: [
            { name: "A", dtype: "float32", shape: [3, 3, 2], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [2, 4], gen: "negmix" },
        ],
        inputShapes: [
            [-1, -1, -1],
            [-1, -1],
        ],
        attrs: { equation: "iij,jk->ik" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [-1, -1] }],
        tol: 1e-5,
    },
];

const DTYPE_MAP: Record<FeedSpec["dtype"], DataType> = {
    float32: DataType.FLOAT,
    int32: DataType.INT32,
    int64: DataType.INT64,
    uint8: DataType.UINT8,
    int8: DataType.INT8,
    bool: DataType.BOOL,
};

function buildSyntheticGraph(c: SyntheticCase): OnnxGraph.Class {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const builder = new GraphBuilder(graph, "synth");

    const totalInputs = Math.max(
        c.feeds.length,
        ...(c.constInputs?.map((ci) => ci.index + 1) ?? [0]),
    );

    const constByIdx = new Map(c.constInputs?.map((ci) => [ci.index, ci.proto]) ?? []);
    const opInputs = [];
    let feedCursor = 0;

    for (let i = 0; i < totalInputs; i++) {
        const constProto = constByIdx.get(i);
        if (constProto) {
            opInputs.push(builder.createConstant(`c_${i}`, constProto));
        } else {
            const feedIdx = feedCursor++;
            const f = c.feeds[feedIdx];
            const graphShape = c.inputShapes?.[feedIdx] ?? f.shape;
            opInputs.push(
                graph
                    .addNode(f.name)
                    .init(new TensorNode.Builder(DTYPE_MAP[f.dtype], graphShape, "input"))
                    .as(TensorNode),
            );
        }
    }

    const outNames = c.outputNames ?? ["Y"];
    builder.createOpWithExact("target_op", c.opType, opInputs, outNames, c.attrs ?? {});
    inferShapes(graph);

    const outNodes = graph.getNodeById("target_op")!.outgoers.targets.toArray();
    outNodes.forEach((out, idx) => {
        if (out.is(TensorNode)) {
            const tn = out.as(TensorNode);
            tn.setType("output");
            const spec = c.outputSpecs?.[idx];
            if (spec) {
                tn.setLiteralType(spec.type);
                tn.setShape(spec.shape);
            }
        }
    });
    return graph;
}

async function exportGraph(
    graph: OnnxGraph.Class,
    onnxPath: string,
    minOpset?: number,
): Promise<void> {
    prepareGraphForExport(graph);
    const json = convertFlowGraphToOnnxJson(graph) as {
        irVersion?: number;
        opsetImport?: { domain?: string; version?: number }[];
    };
    if (minOpset !== undefined && json.opsetImport?.[0]) {
        json.opsetImport[0].version = Math.max(json.opsetImport[0].version ?? 13, minOpset);
        if (minOpset >= 20) {
            json.irVersion = Math.max(json.irVersion ?? 7, 9);
        }
    }
    const tmpJson = `${onnxPath}.json`;
    safeWriteJson(tmpJson, json);
    await json2onnx(tmpJson, onnxPath);
}

async function runOrtChecked(
    stage: string,
    modelPath: string,
    feeds: Record<string, ort.Tensor>,
    sessionOptions?: ort.InferenceSession.SessionOptions,
): Promise<Record<string, ort.Tensor>> {
    try {
        const session = sessionOptions
            ? await ort.InferenceSession.create(modelPath, sessionOptions)
            : await ort.InferenceSession.create(modelPath);
        return await session.run(feeds);
    } catch (e) {
        const prevDebug = ort.env.debug;
        const prevLogLevel = ort.env.logLevel;
        ort.env.debug = true;
        ort.env.logLevel = "verbose";
        console.error(
            `\n[ORT Diagnostic] Re-running ${stage} (${modelPath}) with verbose logging...`,
        );
        const jsonPath = `${modelPath}.json`;
        if (fs.existsSync(jsonPath)) {
            const m = JSON.parse(fs.readFileSync(jsonPath, "utf8"));
            console.error(`[Model Summary ${stage}] opset=`, m.opsetImport, "ir=", m.irVersion);
            console.error(`[Model Summary ${stage}] outputs=`, m.graph?.output);
            console.error(
                `[Model Summary ${stage}] nodes=`,
                (m.graph?.node ?? []).map(
                    (n: { opType: string; input: string[]; output: string[] }) =>
                        `${n.opType}(${n.input.join(",")})->(${n.output.join(",")})`,
                ),
            );
            const lastLoop = [...(m.graph?.node ?? [])]
                .reverse()
                .find((n: { opType: string }) => n.opType === "Loop");
            const bodyGraph = lastLoop?.attribute?.find(
                (a: { name: string }) => a.name === "body",
            )?.g;
            if (bodyGraph !== undefined) {
                console.error(`[Loop Body ${stage}] inputs=`, JSON.stringify(bodyGraph.input));
                console.error(`[Loop Body ${stage}] outputs=`, JSON.stringify(bodyGraph.output));
                console.error(
                    `[Loop Body ${stage}] nodes=`,
                    (bodyGraph.node ?? []).map(
                        (n: { opType: string; input: string[]; output: string[] }) =>
                            `${n.opType}(${n.input.join(",")})->(${n.output.join(",")})`,
                    ),
                );
            }
        }
        try {
            const sess = await ort.InferenceSession.create(modelPath, {
                executionProviders: ["wasm"],
                logSeverityLevel: 0,
                logVerbosityLevel: 4,
            });
            await sess.run(feeds);
        } catch (verboseErr) {
            console.error(`[ORT Diagnostic] ${stage} error:`, verboseErr);
        } finally {
            ort.env.debug = prevDebug!;
            ort.env.logLevel = prevLogLevel!;
        }
        throw new Error(`[${stage}] ORT failed: ${e instanceof Error ? e.message : String(e)}`);
    }
}

describe("Integration: Programmatic ORT Equivalence for New Recipes", () => {
    let tmpDir: string;

    beforeAll(() => {
        initializeSchemaRegistry();
        ensureProtoExists();
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-new-recipes-"));
    });

    afterAll(() => {
        if (tmpDir && fs.existsSync(tmpDir)) {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it.each(CASES)(
        "matches ONNX Runtime outputs for $name",
        async (tc) => {
            const modes = tc.indexingModes ?? [IndexingMode.Pointer];
            const feeds = buildFeeds(tc.feeds, tc.name);

            // 1. Export & run original reference graph
            const origGraph = buildSyntheticGraph(tc);
            const origPath = path.join(tmpDir, `${tc.name}_orig.onnx`);
            await exportGraph(origGraph, origPath);
            const origOuts = await runOrtChecked(`${tc.name}:orig`, origPath, feeds);

            for (const mode of modes) {
                // 2. Lower graph (Canonicalization -> InferShapes -> LoopLowering)
                const loweredGraph = buildSyntheticGraph(tc);
                const history = new HistoryManager(loweredGraph);
                new CanonicalizationPass().run(loweredGraph, history);
                inferShapes(loweredGraph);

                if (tc.skipLoopLowering === undefined || !tc.skipLoopLowering) {
                    new LoopLoweringPass({ coalesce: true, fuse: true, indexingMode: mode }).run(
                        loweredGraph,
                        history,
                    );
                    inferShapes(loweredGraph);
                }

                const loweredPath = path.join(tmpDir, `${tc.name}_lowered_${mode}.onnx`);
                await exportGraph(loweredGraph, loweredPath);
                const recOuts = await runOrtChecked(
                    `${tc.name}:lowered(${mode})`,
                    loweredPath,
                    feeds,
                );

                // 3. Compare EVERY output tensor (handles multi-output ops)
                // Guard against silent no-op (match() returning false)
                const topLevelOps = loweredGraph
                    .getOperationNodes()
                    .toArray()
                    .map((o) => o.type);
                expect(topLevelOps, `${tc.name} (${mode}) was not lowered!`).not.toContain(
                    tc.opType,
                );

                const cmp = compareOrtOutputs(origOuts, recOuts, {
                    tol: tc.tol ?? 1e-4,
                    rtol: tc.rtol ?? 0,
                    exact: tc.exact ?? false,
                });
                expect(cmp.passed, `${tc.name} (${mode}): ${cmp.message ?? ""}`).toBe(true);
            }
        },
        100000,
    );
});
