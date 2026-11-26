import { describe, it, expect, beforeAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Graph from "@specs-feup/flow/graph/Graph";
import * as ort from "onnxruntime-web";
import OnnxGraph from "../../../src/onnx/OnnxGraph.js";
import TensorNode from "../../../src/onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/onnx/GraphBuilder.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { Recipes } from "../../../src/onnx/transformation/recipes.js";
import { CanonicalizationPass } from "../../../src/onnx/transformation/canonicalization/CanonicalizationPass.js";
import { HistoryManager } from "../../../src/onnx/transformation/tracking/HistoryManager.js";
import { convertFlowGraphToOnnxJson } from "../../../src/io/OnnxGraphExporter.js";
import { json2onnx } from "../../../src/io/JsonToOnnx.js";
import { DataType, type AttributeMap } from "../../../src/onnx/OnnxTypes.js";
import { makeTensorProto, topologicalSortOperationNodes } from "../../../src/onnx/utils/index.js";
import { ensureProtoExists } from "../../setup.js";

function getOps(graph: OnnxGraph.Class): string[] {
    return topologicalSortOperationNodes(graph).map((n) => n.type);
}

async function runOrt(
    graph: OnnxGraph.Class,
    feeds: Record<string, ort.Tensor>,
    tmpDir: string,
    name: string,
    exportOpts?: { targetOpset?: number; irVersion?: number },
): Promise<Record<string, ort.Tensor>> {
    const json = convertFlowGraphToOnnxJson(graph, name, undefined, exportOpts);
    const jsonPath = path.join(tmpDir, `${name}.json`);
    const onnxPath = path.join(tmpDir, `${name}.onnx`);
    fs.writeFileSync(jsonPath, JSON.stringify(json));
    await json2onnx(jsonPath, onnxPath, exportOpts);
    const session = await ort.InferenceSession.create(onnxPath);
    return await session.run(feeds);
}

function expectCloseArrays(actual: Float32Array, expected: Float32Array, tol = 1e-4): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < actual.length; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(tol);
    }
}

describe("Opset 21-28 Canonicalization Recipes & Equivalence", () => {
    initializeSchemaRegistry();

    beforeAll(() => {
        ensureProtoExists();
    });

    function buildSingleOpGraph(
        opType: string,
        inputs: {
            name: string;
            dtype: DataType;
            shape: number[];
            constantValues?: number[];
        }[],
        attrs: AttributeMap = {},
    ): {
        graph: OnnxGraph.Class;
        outNodes: TensorNode.Class[];
    } {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const resolvedInputs = inputs.map((inp) => {
            if (inp.constantValues !== undefined) {
                return builder.createConstant(
                    inp.name,
                    makeTensorProto(inp.dtype, inp.shape, inp.constantValues),
                );
            }
            return graph
                .addNode(inp.name)
                .init(new TensorNode.Builder(inp.dtype, inp.shape, "input"))
                .as(TensorNode);
        });

        const outs = builder.createOp(opType, resolvedInputs, attrs);
        const outNodes: TensorNode.Class[] = [];
        outs.forEach((outNode) => {
            if (outNode.is(TensorNode)) {
                outNode.setType("output");
                outNodes.push(outNode);
            }
        });
        return { graph, outNodes };
    }

    it("canonicalizes Swish and matches ORT numerical output", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-swish-"));
        try {
            const xData = new Float32Array([-2.0, -0.5, 0.0, 1.0, 2.5, -1.2]);
            const orig = buildSingleOpGraph(
                "Swish",
                [{ name: "X", dtype: DataType.FLOAT, shape: [2, 3] }],
                { alpha: 1.5 },
            );
            const canon = buildSingleOpGraph(
                "Swish",
                [{ name: "X", dtype: DataType.FLOAT, shape: [2, 3] }],
                { alpha: 1.5 },
            );
            const hm = new HistoryManager(canon.graph);
            new CanonicalizationPass([new Recipes.canonicalization.Swish()]).run(canon.graph, hm);
            expect(getOps(canon.graph)).toEqual(["Mul", "Sigmoid", "Mul"]);

            const feed = { X: new ort.Tensor("float32", xData, [2, 3]) };
            const origRes = await runOrt(orig.graph, feed, tmpDir, "swish_orig");
            const canonRes = await runOrt(canon.graph, feed, tmpDir, "swish_canon");
            expectCloseArrays(
                canonRes[canon.outNodes[0].id].data as Float32Array,
                origRes[orig.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("canonicalizes RMSNormalization and matches ORT numerical output", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-rmsnorm-"));
        try {
            const orig = buildSingleOpGraph(
                "RMSNormalization",
                [
                    { name: "X", dtype: DataType.FLOAT, shape: [2, 4] },
                    {
                        name: "scale",
                        dtype: DataType.FLOAT,
                        shape: [4],
                        constantValues: [1.0, 2.0, 0.5, 1.5],
                    },
                ],
                { axis: -1, epsilon: 1e-5 },
            );
            const canon = buildSingleOpGraph(
                "RMSNormalization",
                [
                    { name: "X", dtype: DataType.FLOAT, shape: [2, 4] },
                    {
                        name: "scale",
                        dtype: DataType.FLOAT,
                        shape: [4],
                        constantValues: [1.0, 2.0, 0.5, 1.5],
                    },
                ],
                { axis: -1, epsilon: 1e-5 },
            );

            new CanonicalizationPass([new Recipes.canonicalization.RMSNormalization()]).run(
                canon.graph,
                new HistoryManager(canon.graph),
            );
            expect(getOps(canon.graph)).not.toContain("RMSNormalization");
            expect(getOps(canon.graph)).toContain("ReduceMean");

            const feed = {
                X: new ort.Tensor(
                    "float32",
                    new Float32Array([1, -2, 3, -4, 0.5, 1.5, -2.5, 3.5]),
                    [2, 4],
                ),
            };
            const origRes = await runOrt(orig.graph, feed, tmpDir, "rms_orig");
            const canonRes = await runOrt(canon.graph, feed, tmpDir, "rms_canon");
            expectCloseArrays(
                canonRes[canon.outNodes[0].id].data as Float32Array,
                origRes[orig.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("canonicalizes SpaceToDepth and matches ORT output", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-s2d-"));
        try {
            const orig = buildSingleOpGraph(
                "SpaceToDepth",
                [{ name: "X", dtype: DataType.FLOAT, shape: [1, 2, 4, 4] }],
                { blocksize: 2 },
            );
            const canon = buildSingleOpGraph(
                "SpaceToDepth",
                [{ name: "X", dtype: DataType.FLOAT, shape: [1, 2, 4, 4] }],
                { blocksize: 2 },
            );
            new CanonicalizationPass([new Recipes.canonicalization.SpaceToDepth()]).run(
                canon.graph,
                new HistoryManager(canon.graph),
            );
            expect(getOps(canon.graph)).toEqual(["Reshape", "Transpose", "Reshape"]);

            const data = Float32Array.from({ length: 32 }, (_, i) => i + 1);
            const feed = { X: new ort.Tensor("float32", data, [1, 2, 4, 4]) };
            const origRes = await runOrt(orig.graph, feed, tmpDir, "s2d_orig");
            const canonRes = await runOrt(canon.graph, feed, tmpDir, "s2d_canon");
            expectCloseArrays(
                canonRes[canon.outNodes[0].id].data as Float32Array,
                origRes[orig.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("canonicalizes RotaryEmbedding (4D split-half & 3D interleaved) and matches ORT output", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-rope-"));
        try {
            // Case A: 4D split-half with position_ids
            const cosVals = Array.from({ length: 16 }, (_, i) => Math.cos(i * 0.1));
            const sinVals = Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.1));
            const posVals = [0, 2, 1, 3];

            const make4D = () =>
                buildSingleOpGraph(
                    "RotaryEmbedding",
                    [
                        { name: "X", dtype: DataType.FLOAT, shape: [1, 2, 4, 4] },
                        {
                            name: "cos",
                            dtype: DataType.FLOAT,
                            shape: [8, 2],
                            constantValues: cosVals,
                        },
                        {
                            name: "sin",
                            dtype: DataType.FLOAT,
                            shape: [8, 2],
                            constantValues: sinVals,
                        },
                        {
                            name: "pos",
                            dtype: DataType.INT64,
                            shape: [1, 4],
                            constantValues: posVals,
                        },
                    ],
                    { interleaved: 0 },
                );

            const orig4D = make4D();
            const canon4D = make4D();
            new CanonicalizationPass([new Recipes.canonicalization.RotaryEmbedding()]).run(
                canon4D.graph,
                new HistoryManager(canon4D.graph),
            );
            expect(getOps(canon4D.graph)).not.toContain("RotaryEmbedding");

            const x4Data = Float32Array.from({ length: 32 }, (_, i) => (i - 15) * 0.25);
            const feed4D = { X: new ort.Tensor("float32", x4Data, [1, 2, 4, 4]) };
            const origRes4 = await runOrt(orig4D.graph, feed4D, tmpDir, "rope4_orig");
            const canonRes4 = await runOrt(canon4D.graph, feed4D, tmpDir, "rope4_canon");
            expectCloseArrays(
                canonRes4[canon4D.outNodes[0].id].data as Float32Array,
                origRes4[orig4D.outNodes[0].id].data as Float32Array,
            );

            // Case B: 3D interleaved with partial rotary_embedding_dim
            const cosB = Array.from({ length: 6 }, (_, i) => Math.cos(i * 0.2));
            const sinB = Array.from({ length: 6 }, (_, i) => Math.sin(i * 0.2));
            const make3D = () =>
                buildSingleOpGraph(
                    "RotaryEmbedding",
                    [
                        { name: "X", dtype: DataType.FLOAT, shape: [1, 3, 12] },
                        {
                            name: "cos",
                            dtype: DataType.FLOAT,
                            shape: [1, 3, 2],
                            constantValues: cosB,
                        },
                        {
                            name: "sin",
                            dtype: DataType.FLOAT,
                            shape: [1, 3, 2],
                            constantValues: sinB,
                        },
                    ],
                    { interleaved: 1, num_heads: 2, rotary_embedding_dim: 4 },
                );

            const orig3D = make3D();
            const canon3D = make3D();
            new CanonicalizationPass([new Recipes.canonicalization.RotaryEmbedding()]).run(
                canon3D.graph,
                new HistoryManager(canon3D.graph),
            );
            expect(getOps(canon3D.graph)).not.toContain("RotaryEmbedding");

            const x3Data = Float32Array.from({ length: 36 }, (_, i) => (i - 10) * 0.15);
            const feed3D = { X: new ort.Tensor("float32", x3Data, [1, 3, 12]) };
            const origRes3 = await runOrt(orig3D.graph, feed3D, tmpDir, "rope3_orig");
            const canonRes3 = await runOrt(canon3D.graph, feed3D, tmpDir, "rope3_canon");
            expectCloseArrays(
                canonRes3[canon3D.outNodes[0].id].data as Float32Array,
                origRes3[orig3D.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("canonicalizes Attention (3D GQA causal + softcap) and matches ORT output", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-attn-"));
        try {
            const makeAttn = () =>
                buildSingleOpGraph(
                    "Attention",
                    [
                        { name: "Q", dtype: DataType.FLOAT, shape: [1, 3, 8] }, // q_num_heads=4, d=2
                        { name: "K", dtype: DataType.FLOAT, shape: [1, 3, 4] }, // kv_num_heads=2, d=2
                        { name: "V", dtype: DataType.FLOAT, shape: [1, 3, 4] },
                    ],
                    { q_num_heads: 4, kv_num_heads: 2, is_causal: 1, softcap: 5.0 },
                );

            const orig = makeAttn();
            const canon = makeAttn();
            new CanonicalizationPass([new Recipes.canonicalization.Attention()]).run(
                canon.graph,
                new HistoryManager(canon.graph),
            );
            expect(getOps(canon.graph)).not.toContain("Attention");
            expect(getOps(canon.graph)).toContain("MatMul");
            expect(getOps(canon.graph)).toContain("Softmax");

            const qData = Float32Array.from({ length: 24 }, (_, i) => Math.sin(i * 0.3));
            const kData = Float32Array.from({ length: 12 }, (_, i) => Math.cos(i * 0.4));
            const vData = Float32Array.from({ length: 12 }, (_, i) => (i - 5) * 0.2);
            const feed = {
                Q: new ort.Tensor("float32", qData, [1, 3, 8]),
                K: new ort.Tensor("float32", kData, [1, 3, 4]),
                V: new ort.Tensor("float32", vData, [1, 3, 4]),
            };

            const origRes = await runOrt(orig.graph, feed, tmpDir, "attn_orig");
            const canonRes = await runOrt(canon.graph, feed, tmpDir, "attn_canon");
            expectCloseArrays(
                canonRes[canon.outNodes[0].id].data as Float32Array,
                origRes[orig.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("canonicalizes TensorScatter (circular KV-cache update) and matches ORT output", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-tscatter-"));
        try {
            const makeTs = () =>
                buildSingleOpGraph(
                    "TensorScatter",
                    [
                        { name: "cache", dtype: DataType.FLOAT, shape: [2, 1, 4, 2] },
                        { name: "upd", dtype: DataType.FLOAT, shape: [2, 1, 2, 2] },
                        { name: "widx", dtype: DataType.INT64, shape: [2], constantValues: [1, 3] },
                    ],
                    { axis: -2, mode: "circular" },
                );

            const orig = makeTs();
            const canon = makeTs();
            new CanonicalizationPass([new Recipes.canonicalization.TensorScatter()]).run(
                canon.graph,
                new HistoryManager(canon.graph),
            );
            expect(getOps(canon.graph)).not.toContain("TensorScatter");
            expect(getOps(canon.graph)).toContain("ScatterElements");

            const cacheData = new Float32Array(16).fill(0);
            const updData = Float32Array.from({ length: 8 }, (_, i) => i + 10);
            const feed = {
                cache: new ort.Tensor("float32", cacheData, [2, 1, 4, 2]),
                upd: new ort.Tensor("float32", updData, [2, 1, 2, 2]),
            };

            const origRes = await runOrt(orig.graph, feed, tmpDir, "ts_orig");
            const canonRes = await runOrt(canon.graph, feed, tmpDir, "ts_canon");
            expectCloseArrays(
                canonRes[canon.outNodes[0].id].data as Float32Array,
                origRes[orig.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("handles GroupNormalization stash_type, INT4 QuantizeLinear, and Reduce* noop_with_empty_axes=1", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-ext28-"));
        try {
            // 1. GroupNormalization Opset 21 FP16 stash_type emits Cast around reduction
            const gn = buildSingleOpGraph(
                "GroupNormalization",
                [
                    { name: "X", dtype: DataType.FLOAT16, shape: [1, 4, 2, 2] },
                    {
                        name: "S",
                        dtype: DataType.FLOAT16,
                        shape: [4],
                        constantValues: [1, 1, 1, 1],
                    },
                    {
                        name: "B",
                        dtype: DataType.FLOAT16,
                        shape: [4],
                        constantValues: [0, 0, 0, 0],
                    },
                ],
                { num_groups: 2, epsilon: 1e-5, stash_type: DataType.FLOAT },
            );
            new CanonicalizationPass([new Recipes.canonicalization.GroupNormalization()]).run(
                gn.graph,
                new HistoryManager(gn.graph),
            );
            expect(getOps(gn.graph)).toContain("Cast");
            expect(getOps(gn.graph)).not.toContain("GroupNormalization");

            // 2. QuantizeLinear INT4 with output_dtype (no zero_point)
            const ql = buildSingleOpGraph(
                "QuantizeLinear",
                [
                    { name: "X", dtype: DataType.FLOAT, shape: [2, 4] },
                    { name: "S", dtype: DataType.FLOAT, shape: [], constantValues: [0.5] },
                ],
                { output_dtype: DataType.INT4 },
            );
            new CanonicalizationPass([
                new Recipes.canonicalization.QuantizeLinear(),
                new Recipes.canonicalization.QuantizeLinearExtended(),
            ]).run(ql.graph, new HistoryManager(ql.graph));
            expect(getOps(ql.graph)).toEqual(["Div", "Round", "Clip", "Cast"]);

            // 3. ReduceSum with noop_with_empty_axes=1 -> Identity & ORT match
            const redOrig = buildSingleOpGraph(
                "ReduceSum",
                [{ name: "X", dtype: DataType.FLOAT, shape: [2, 3] }],
                { noop_with_empty_axes: 1, keepdims: 1 },
            );
            expect(redOrig.outNodes[0].shape).toEqual([2, 3]);

            const redCanon = buildSingleOpGraph(
                "ReduceSum",
                [{ name: "X", dtype: DataType.FLOAT, shape: [2, 3] }],
                { noop_with_empty_axes: 1, keepdims: 1 },
            );
            new CanonicalizationPass([new Recipes.canonicalization.ReductionNoop("ReduceSum")]).run(
                redCanon.graph,
                new HistoryManager(redCanon.graph),
            );
            expect(getOps(redCanon.graph)).toEqual(["Identity"]);

            const feed = {
                X: new ort.Tensor("float32", new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]),
            };
            const origRes = await runOrt(redOrig.graph, feed, tmpDir, "red_noop_orig");
            const canonRes = await runOrt(redCanon.graph, feed, tmpDir, "red_noop_canon");
            expectCloseArrays(
                canonRes[redCanon.outNodes[0].id].data as Float32Array,
                origRes[redOrig.outNodes[0].id].data as Float32Array,
            );
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });
});
