import { describe, it, expect, beforeAll, afterAll } from "vitest";
import * as ort from "onnxruntime-web";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../src/Onnx/TensorNode.js";
import OperationNode from "../../src/Onnx/OperationNode.js";
import OnnxEdge from "../../src/Onnx/OnnxEdge.js";
import { GraphBuilder } from "../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../src/Onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../src/Onnx/Schema/index.js";
import inferShapes from "../../src/Onnx/InferShapes.js";
import { int64Vec, scalarInt64, floatVec, makeTensorProto } from "../../src/Onnx/Utils.js";
import { CanonicalizationPass } from "../../src/Onnx/transformation/canonicalization/CanonicalizationPass.js";
import { LoopLoweringPass } from "../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { HistoryManager } from "../../src/Onnx/transformation/tracking/HistoryManager.js";
import { IndexingMode } from "../../src/DecompositionOptions.js";
import {
    buildFeeds,
    compareOrtOutputs,
    ensureProtoExists,
    exportGraphToOnnx,
    runOrtChecked,
} from "../helpers/ortHarness.js";
import { buildSyntheticGraph, type SyntheticCase } from "../helpers/modelFactory.js";

const TEST_TIMEOUT = 150000; //ms

interface OpEquivalenceCase extends SyntheticCase {
    indexingModes?: IndexingMode[];
    /** Optional ORT graph optimization level override */
    graphOptimizationLevel?: ort.InferenceSession.SessionOptions["graphOptimizationLevel"];
    /** Minimum ONNX opset version for original & lowered export (e.g., 21 for blocked QDQ) */
    minOpset?: number;
    /** Optional opset override only for original reference export (e.g., 10 for legacy Scatter) */
    origOpset?: number;
    /** Optional custom graph builder (e.g., Scan with body subgraph) */
    customBuild?: () => OnnxGraph.Class;
    /** Optional precomputed reference outputs when ORT lacks a reference kernel (e.g. LpNormalization p=3) */
    expectedOutputs?: Record<string, ort.Tensor>;
}

function buildScanGraph(): OnnxGraph.Class {
    const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const bodyBuilder = new GraphBuilder(body, "scan_body");
    const sIn = body
        .addNode("s_in")
        .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
        .as(TensorNode);
    const xIn = body
        .addNode("x_in")
        .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
        .as(TensorNode);
    const [sum] = bodyBuilder.createOp("Add", [sIn, xIn], {}, [
        { type: DataType.FLOAT, shape: [2] },
    ]);
    const [sOut] = bodyBuilder.createOp("Identity", [sum], {}, [
        { type: DataType.FLOAT, shape: [2] },
    ]);
    const [yOut] = bodyBuilder.createOp("Identity", [sum], {}, [
        { type: DataType.FLOAT, shape: [2] },
    ]);
    if (sOut.is(TensorNode)) sOut.as(TensorNode).setType("output");
    if (yOut.is(TensorNode)) yOut.as(TensorNode).setType("output");

    const outer = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const initState = outer
        .addNode("init_s")
        .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
        .as(TensorNode);
    const scanIn = outer
        .addNode("scan_x")
        .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
        .as(TensorNode);

    const scanOp = outer
        .addNode("scan_0")
        .init(
            new OperationNode.Builder("Scan", [initState, scanIn], { num_scan_inputs: 1 }, [body]),
        )
        .as(OperationNode);
    outer.addEdge(initState, scanOp).init(new OnnxEdge.Builder(DataType.FLOAT, [2]));
    outer.addEdge(scanIn, scanOp).init(new OnnxEdge.Builder(DataType.FLOAT, [3, 2]));

    const outS = outer
        .addNode("final_s")
        .init(new TensorNode.Builder(DataType.FLOAT, [2], "output"))
        .as(TensorNode);
    const outY = outer
        .addNode("scan_y")
        .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "output"))
        .as(TensorNode);
    outer.addEdge(scanOp, outS).init(new OnnxEdge.Builder(DataType.FLOAT, [2]));
    outer.addEdge(scanOp, outY).init(new OnnxEdge.Builder(DataType.FLOAT, [3, 2]));
    return outer;
}

const CASES: OpEquivalenceCase[] = [
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

    // ===== Previously Missing Recipe Coverage =====
    {
        name: "BatchNormalization",
        opType: "BatchNormalization",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 3, 2, 2], gen: "negmix" },
            { name: "scale", dtype: "float32", shape: [3], init: [1.0, 0.5, 2.0] },
            { name: "B", dtype: "float32", shape: [3], init: [0.1, -0.2, 0.3] },
            { name: "mean", dtype: "float32", shape: [3], init: [0.0, 0.5, -0.5] },
            { name: "var", dtype: "float32", shape: [3], init: [1.0, 2.0, 0.5] },
        ],
        attrs: { epsilon: 1e-5 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 3, 2, 2] }],
        tol: 1e-4,
    },
    {
        name: "LayerNormalization",
        opType: "LayerNormalization",
        feeds: [
            { name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" },
            { name: "scale", dtype: "float32", shape: [4], init: [1.0, 1.5, 0.5, 2.0] },
            { name: "B", dtype: "float32", shape: [4], init: [0.0, 0.1, -0.1, 0.2] },
        ],
        attrs: { axis: -1, epsilon: 1e-5 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 4] }],
        tol: 1e-4,
    },
    {
        name: "InstanceNormalization",
        opType: "InstanceNormalization",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 3, 2, 2], gen: "negmix" },
            { name: "scale", dtype: "float32", shape: [3], init: [1.0, 0.5, 2.0] },
            { name: "B", dtype: "float32", shape: [3], init: [0.1, -0.2, 0.3] },
        ],
        attrs: { epsilon: 1e-5 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 3, 2, 2] }],
        tol: 1e-4,
    },
    {
        name: "GroupNormalization",
        opType: "GroupNormalization",
        feeds: [
            { name: "X", dtype: "float32", shape: [1, 4, 2, 2], gen: "negmix" },
            { name: "scale", dtype: "float32", shape: [4], init: [1.0, 0.5, 2.0, 1.2] },
            { name: "bias", dtype: "float32", shape: [4], init: [0.1, -0.1, 0.2, 0.0] },
        ],
        attrs: { num_groups: 2, epsilon: 1e-5 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 4, 2, 2] }],
        minOpset: 21,
        tol: 1e-4,
    },
    {
        name: "Ceil",
        opType: "Ceil",
        feeds: [
            { name: "X", dtype: "float32", shape: [6], init: [-2.7, -2.0, -0.1, 0.0, 0.9, 3.4] },
        ],
        tol: 1e-6,
    },
    {
        name: "SqrtPC",
        opType: "Sqrt",
        feeds: [
            { name: "X", dtype: "float32", shape: [6], init: [0.0, 0.25, 1.0, 2.0, 9.0, 100.0] },
        ],
        tol: 1e-3,
    },
    {
        name: "SinPC",
        opType: "Sin",
        feeds: [{ name: "X", dtype: "float32", shape: [4], init: [-0.8, -0.2, 0.3, 0.9] }],
        tol: 1e-4,
    },
    {
        name: "CosPC",
        opType: "Cos",
        feeds: [{ name: "X", dtype: "float32", shape: [4], init: [-0.8, -0.2, 0.3, 0.9] }],
        tol: 1e-4,
    },
    {
        name: "LogPC",
        opType: "Log",
        feeds: [{ name: "X", dtype: "float32", shape: [4], init: [0.4, 0.75, 1.0, 2.5] }],
        tol: 1e-3,
    },
    {
        name: "Erf",
        opType: "Erf",
        feeds: [{ name: "X", dtype: "float32", shape: [4], init: [-0.5, -0.2, 0.2, 0.5] }],
        tol: 5e-3,
    },
    {
        name: "IsNaN",
        opType: "IsNaN",
        feeds: [
            { name: "X", dtype: "float32", shape: [4], init: [0.0, Number.NaN, -1.5, Number.NaN] },
        ],
        outputSpecs: [{ type: DataType.BOOL, shape: [4] }],
        exact: true,
    },
    {
        name: "IsInf",
        opType: "IsInf",
        feeds: [
            {
                name: "X",
                dtype: "float32",
                shape: [4],
                init: [0.0, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, 3.5],
            },
        ],
        outputSpecs: [{ type: DataType.BOOL, shape: [4] }],
        exact: true,
    },
    {
        name: "RoundAdd",
        opType: "Round",
        feeds: [
            { name: "X", dtype: "float32", shape: [6], init: [-2.7, -1.5, -0.4, 0.4, 1.5, 2.7] },
        ],
        tol: 1e-6,
    },
    {
        name: "QuantizeLinear_uint8",
        opType: "QuantizeLinear",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" }],
        constInputs: [
            { index: 1, proto: makeTensorProto(DataType.FLOAT, [], [0.1]) },
            { index: 2, proto: makeTensorProto(DataType.UINT8, [], [128]) },
        ],
        outputSpecs: [{ type: DataType.UINT8, shape: [2, 3] }],
        tol: 1,
    },
    {
        name: "QuantizeLinearExtended_int16",
        opType: "QuantizeLinear",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        constInputs: [
            { index: 1, proto: makeTensorProto(DataType.FLOAT, [], [0.01]) },
            { index: 2, proto: makeTensorProto(DataType.INT16, [], [0]) },
        ],
        outputSpecs: [{ type: DataType.INT16, shape: [2, 4] }],
        minOpset: 21,
        tol: 1,
    },
    {
        name: "QuantizeLinearBlocked_int8",
        opType: "QuantizeLinear",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 6], gen: "negmix" }],
        constInputs: [
            {
                index: 1,
                proto: makeTensorProto(DataType.FLOAT, [2, 3], [0.1, 0.2, 0.5, 0.1, 0.2, 0.5]),
            },
            { index: 2, proto: makeTensorProto(DataType.INT8, [2, 3], [0, 1, -1, 0, 1, -1]) },
        ],
        attrs: { axis: 1, block_size: 2 },
        outputSpecs: [{ type: DataType.INT8, shape: [2, 6] }],
        minOpset: 21,
        tol: 1,
    },
    {
        name: "DequantizeLinearBlocked_int8",
        opType: "DequantizeLinear",
        feeds: [
            {
                name: "X",
                dtype: "int8",
                shape: [2, 6],
                init: [-10, 5, 20, -5, 0, 15, 1, 2, 3, 4, 5, 6],
            },
        ],
        constInputs: [
            {
                index: 1,
                proto: makeTensorProto(DataType.FLOAT, [2, 3], [0.1, 0.2, 0.5, 0.1, 0.2, 0.5]),
            },
            { index: 2, proto: makeTensorProto(DataType.INT8, [2, 3], [0, 1, -1, 0, 1, -1]) },
        ],
        attrs: { axis: 1, block_size: 2 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 6] }],
        minOpset: 21,
        tol: 1e-5,
    },
    {
        name: "ConvInteger",
        opType: "ConvInteger",
        feeds: [
            {
                name: "X",
                dtype: "uint8",
                shape: [1, 1, 4, 4],
                init: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            { name: "W", dtype: "uint8", shape: [1, 1, 2, 2], init: [1, 0, 0, 2] },
        ],
        attrs: { kernel_shape: [2, 2], strides: [1, 1], pads: [0, 0, 0, 0] },
        outputSpecs: [{ type: DataType.INT32, shape: [1, 1, 3, 3] }],
        exact: true,
    },
    {
        name: "QLinearMatMul",
        opType: "QLinearMatMul",
        feeds: [
            { name: "a", dtype: "uint8", shape: [2, 3], init: [10, 20, 30, 40, 50, 60] },
            { name: "b", dtype: "uint8", shape: [3, 2], init: [1, 2, 3, 4, 5, 6] },
        ],
        constInputs: [
            { index: 1, proto: makeTensorProto(DataType.FLOAT, [], [0.1]) },
            { index: 2, proto: makeTensorProto(DataType.UINT8, [], [0]) },
            { index: 4, proto: makeTensorProto(DataType.FLOAT, [], [0.2]) },
            { index: 5, proto: makeTensorProto(DataType.UINT8, [], [0]) },
            { index: 6, proto: makeTensorProto(DataType.FLOAT, [], [0.5]) },
            { index: 7, proto: makeTensorProto(DataType.UINT8, [], [0]) },
        ],
        outputSpecs: [{ type: DataType.UINT8, shape: [2, 2] }],
        tol: 1,
    },
    {
        name: "QLinearConv",
        opType: "QLinearConv",
        feeds: [
            {
                name: "x",
                dtype: "uint8",
                shape: [1, 1, 4, 4],
                init: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            },
            { name: "w", dtype: "uint8", shape: [1, 1, 2, 2], init: [1, 2, 3, 4] },
        ],
        constInputs: [
            { index: 1, proto: makeTensorProto(DataType.FLOAT, [], [0.1]) },
            { index: 2, proto: makeTensorProto(DataType.UINT8, [], [0]) },
            { index: 4, proto: makeTensorProto(DataType.FLOAT, [], [0.1]) },
            { index: 5, proto: makeTensorProto(DataType.UINT8, [], [0]) },
            { index: 6, proto: makeTensorProto(DataType.FLOAT, [], [0.2]) },
            { index: 7, proto: makeTensorProto(DataType.UINT8, [], [0]) },
        ],
        attrs: { kernel_shape: [2, 2], strides: [1, 1], pads: [0, 0, 0, 0] },
        outputSpecs: [{ type: DataType.UINT8, shape: [1, 1, 3, 3] }],
        tol: 1,
    },
    {
        name: "LpNormalizationGeneral_L3",
        opType: "LpNormalization",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], init: [1, -2, 3, -1, 2, 2, -1, 0] }],
        attrs: { p: 3, axis: -1 },
        // ORT CPU kernel only implements p=1 and p=2; verify lowered graph against exact L3 math
        expectedOutputs: {
            Y: new ort.Tensor(
                "float32",
                new Float32Array([
                    1 / Math.cbrt(37),
                    -2 / Math.cbrt(37),
                    3 / Math.cbrt(37),
                    -1 / Math.cbrt(37),
                    2 / Math.cbrt(17),
                    2 / Math.cbrt(17),
                    -1 / Math.cbrt(17),
                    0,
                ]),
                [2, 4],
            ),
        },
        tol: 1e-3,
    },
    {
        name: "Scan_stateful_sum",
        opType: "Scan",
        feeds: [
            { name: "init_s", dtype: "float32", shape: [2], init: [1.0, 2.0] },
            {
                name: "scan_x",
                dtype: "float32",
                shape: [3, 2],
                init: [0.5, 1.0, 2.0, 3.0, -1.0, 0.5],
            },
        ],
        outputNames: ["final_s", "scan_y"],
        customBuild: buildScanGraph,
        tol: 1e-5,
    },
    {
        name: "Scatter_legacy_to_ScatterElements",
        opType: "Scatter",
        feeds: [
            { name: "D", dtype: "float32", shape: [2, 4] },
            { name: "I", dtype: "int64", shape: [2, 2], init: [0n, 2n, 1n, 3n] },
            { name: "U", dtype: "float32", shape: [2, 2] },
        ],
        attrs: { axis: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 4] }],
        origOpset: 10,
        tol: 1e-6,
    },
    {
        name: "Swish_opset24",
        opType: "Swish",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        attrs: { alpha: 1.25 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 4] }],
        minOpset: 24,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-4,
    },
    {
        name: "RMSNormalization_opset23",
        opType: "RMSNormalization",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
        constInputs: [
            {
                index: 1,
                proto: makeTensorProto(DataType.FLOAT, [4], [1.0, 1.5, 0.75, 1.25]),
            },
        ],
        attrs: { axis: -1, epsilon: 1e-5 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 4] }],
        minOpset: 23,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-4,
    },
    {
        name: "SpaceToDepth_dcr",
        opType: "SpaceToDepth",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "range" }],
        attrs: { blocksize: 2 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 8, 2, 2] }],
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-6,
    },
    {
        name: "RotaryEmbedding_4d_split",
        opType: "RotaryEmbedding",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
        constInputs: [
            {
                index: 1,
                proto: makeTensorProto(
                    DataType.FLOAT,
                    [8, 2],
                    Array.from({ length: 16 }, (_, i) => Math.cos(i * 0.1)),
                ),
            },
            {
                index: 2,
                proto: makeTensorProto(
                    DataType.FLOAT,
                    [8, 2],
                    Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.1)),
                ),
            },
            {
                index: 3,
                proto: makeTensorProto(DataType.INT64, [1, 4], [0, 2, 1, 3]),
            },
        ],
        attrs: { interleaved: 0 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 2, 4, 4] }],
        minOpset: 23,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-4,
    },
    {
        name: "RotaryEmbedding_3d_interleaved_partial",
        opType: "RotaryEmbedding",
        feeds: [{ name: "X", dtype: "float32", shape: [1, 3, 12], gen: "negmix" }],
        constInputs: [
            {
                index: 1,
                proto: makeTensorProto(
                    DataType.FLOAT,
                    [1, 3, 2],
                    Array.from({ length: 6 }, (_, i) => Math.cos(i * 0.2)),
                ),
            },
            {
                index: 2,
                proto: makeTensorProto(
                    DataType.FLOAT,
                    [1, 3, 2],
                    Array.from({ length: 6 }, (_, i) => Math.sin(i * 0.2)),
                ),
            },
        ],
        attrs: { interleaved: 1, num_heads: 2, rotary_embedding_dim: 4 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 3, 12] }],
        minOpset: 23,
        tol: 1e-4,
    },
    {
        name: "Attention_3d_gqa_causal_softcap",
        opType: "Attention",
        feeds: [
            { name: "Q", dtype: "float32", shape: [1, 3, 8], gen: "negmix" },
            { name: "K", dtype: "float32", shape: [1, 3, 4], gen: "negmix" },
            { name: "V", dtype: "float32", shape: [1, 3, 4], gen: "negmix" },
        ],
        attrs: { q_num_heads: 4, kv_num_heads: 2, is_causal: 1, softcap: 5.0 },
        outputNames: ["Y"],
        outputSpecs: [{ type: DataType.FLOAT, shape: [1, 3, 8] }],
        minOpset: 23,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-4,
    },
    {
        name: "Attention_4d_kvcache_masked",
        opType: "Attention",
        feeds: [
            { name: "Q", dtype: "float32", shape: [1, 2, 2, 4], gen: "negmix" },
            { name: "K", dtype: "float32", shape: [1, 2, 2, 4], gen: "negmix" },
            { name: "V", dtype: "float32", shape: [1, 2, 2, 4], gen: "negmix" },
            { name: "past_k", dtype: "float32", shape: [1, 2, 3, 4], gen: "negmix" },
            { name: "past_v", dtype: "float32", shape: [1, 2, 3, 4], gen: "negmix" },
        ],
        outputNames: ["Y", "present_k", "present_v"],
        customBuild: () => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "attn_kv");
            const q = g
                .addNode("Q")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 2, 4], "input"))
                .as(TensorNode);
            const k = g
                .addNode("K")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 2, 4], "input"))
                .as(TensorNode);
            const v = g
                .addNode("V")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 2, 4], "input"))
                .as(TensorNode);
            const mask = b.createConstant(
                "attn_mask",
                makeTensorProto(DataType.FLOAT, [1, 1, 2, 5], new Array<number>(10).fill(0)),
            );
            const pk = g
                .addNode("past_k")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 3, 4], "input"))
                .as(TensorNode);
            const pv = g
                .addNode("past_v")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 3, 4], "input"))
                .as(TensorNode);
            const attnOuts = b.createOpWithExact(
                "attn_op",
                "Attention",
                [q, k, v, mask, pk, pv],
                ["Y", "present_k", "present_v"],
                { q_num_heads: 2, kv_num_heads: 2, is_causal: 1 },
            );
            for (const o of attnOuts) {
                if (o.is(TensorNode)) o.as(TensorNode).setType("output");
            }
            inferShapes(g);
            return g;
        },
        minOpset: 23,
        tol: 1e-4,
    },
    {
        name: "TensorScatter_circular_cache",
        opType: "TensorScatter",
        feeds: [
            { name: "cache", dtype: "float32", shape: [2, 1, 4, 2], gen: "zeros" },
            { name: "upd", dtype: "float32", shape: [2, 1, 2, 2], gen: "range" },
        ],
        constInputs: [
            {
                index: 2,
                proto: int64Vec([1, 3]),
            },
        ],
        attrs: { axis: -2, mode: "circular" },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 1, 4, 2] }],
        minOpset: 24,
        indexingModes: [IndexingMode.Pointer, IndexingMode.Radix],
        tol: 1e-5,
    },
    {
        name: "ReduceSum_noop_empty_axes",
        opType: "ReduceSum",
        feeds: [{ name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" }],
        attrs: { noop_with_empty_axes: 1, keepdims: 1 },
        outputSpecs: [{ type: DataType.FLOAT, shape: [2, 3] }],
        tol: 1e-6,
    },
    {
        name: "QDQ_int4_output_dtype",
        opType: "QuantizeLinear",
        feeds: [
            {
                name: "X",
                dtype: "float32",
                shape: [2, 4],
                init: [-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0],
            },
        ],
        customBuild: () => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "qdq4");
            const x = g
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                .as(TensorNode);
            const s = b.createConstant("q_s", makeTensorProto(DataType.FLOAT, [], [0.5]));
            const [q] = b.createOp("QuantizeLinear", [x, s], { output_dtype: DataType.INT4 }, [
                { type: DataType.INT4, shape: [2, 4] },
            ]);
            const [dq] = b.createOp("DequantizeLinear", [q, s], {}, [
                { type: DataType.FLOAT, shape: [2, 4] },
            ]);
            if (dq.is(TensorNode)) dq.as(TensorNode).setType("output");
            return g;
        },
        minOpset: 21,
        tol: 1e-5,
    },
];

describe("Integration: Programmatic ORT Equivalence for Operation Recipes", () => {
    let tmpDir: string;

    beforeAll(() => {
        initializeSchemaRegistry();
        ensureProtoExists();
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-op-equiv-"));
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

            let origOuts: Record<string, ort.Tensor>;
            if (tc.expectedOutputs !== undefined) {
                origOuts = tc.expectedOutputs;
            } else {
                const origGraph = tc.customBuild ? tc.customBuild() : buildSyntheticGraph(tc);
                const origPath = path.join(tmpDir, `${tc.name}_orig.onnx`);
                await exportGraphToOnnx(origGraph, origPath, tc.minOpset, tc.origOpset);
                origOuts = await runOrtChecked(`${tc.name}:orig`, origPath, feeds);
            }

            for (const mode of modes) {
                const loweredGraph = tc.customBuild ? tc.customBuild() : buildSyntheticGraph(tc);
                const history = new HistoryManager(loweredGraph);
                new CanonicalizationPass().run(loweredGraph, history);
                inferShapes(loweredGraph);

                if (tc.skipLoopLowering !== true) {
                    new LoopLoweringPass({ coalesce: true, fuse: true, indexingMode: mode }).run(
                        loweredGraph,
                        history,
                    );
                    inferShapes(loweredGraph);
                }

                const loweredPath = path.join(tmpDir, `${tc.name}_lowered_${mode}.onnx`);
                await exportGraphToOnnx(loweredGraph, loweredPath, tc.minOpset);
                const recOuts = await runOrtChecked(
                    `${tc.name}:lowered(${mode})`,
                    loweredPath,
                    feeds,
                );

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
        TEST_TIMEOUT,
    );
});
