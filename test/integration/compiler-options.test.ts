import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../src/Onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../src/Onnx/Schema/index.js";
import OnnxGraphTransformer from "../../src/Onnx/transformation/index.js";
import { IndexingMode, type DecompositionOptions } from "../../src/DecompositionOptions.js";
import { int64Vec } from "../../src/Onnx/Utils.js";
import validateGraph from "../../src/Onnx/validation/ValidateGraph.js";
import {
    buildFeeds,
    compareOrtOutputs,
    ensureProtoExists,
    exportGraphToOnnx,
    runOrtChecked,
    type FeedSpec,
} from "../helpers/ortHarness.js";

interface PipelineCase {
    label: string;
    feeds: FeedSpec[];
    build: () => OnnxGraph.Class;
    tol?: number;
}

function makeSimpleAddChain(): OnnxGraph.Class {
    const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const b = new GraphBuilder(g, "p");
    const x = g
        .addNode("X")
        .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"))
        .as(TensorNode);
    const y = g
        .addNode("Y")
        .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"))
        .as(TensorNode);
    const [r] = b.createOp("Relu", [x], {}, [{ type: DataType.FLOAT, shape: [2, 5] }]);
    const [out] = b.createOp("Add", [r, y], {}, [{ type: DataType.FLOAT, shape: [2, 5] }]);
    if (out.is(TensorNode)) out.as(TensorNode).setType("output");
    return g;
}

function makeMatMulAddRelu(): OnnxGraph.Class {
    const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const b = new GraphBuilder(g, "mm");
    const a = g
        .addNode("A")
        .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
        .as(TensorNode);
    const w = g
        .addNode("W")
        .init(new TensorNode.Builder(DataType.FLOAT, [3, 4], "input"))
        .as(TensorNode);
    const bias = g
        .addNode("B")
        .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
        .as(TensorNode);
    const [mm] = b.createOp("MatMul", [a, w], {}, [{ type: DataType.FLOAT, shape: [2, 4] }]);
    const [sum] = b.createOp("Add", [mm, bias], {}, [{ type: DataType.FLOAT, shape: [2, 4] }]);
    const [out] = b.createOp("Relu", [sum], {}, [{ type: DataType.FLOAT, shape: [2, 4] }]);
    if (out.is(TensorNode)) out.as(TensorNode).setType("output");
    return g;
}

function makeBatchedMatMul(): OnnxGraph.Class {
    const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const b = new GraphBuilder(g, "bmm");
    const a = g
        .addNode("A")
        .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
        .as(TensorNode);
    const w = g
        .addNode("W")
        .init(new TensorNode.Builder(DataType.FLOAT, [1, 4, 2], "input"))
        .as(TensorNode);
    const [out] = b.createOp("MatMul", [a, w], {}, [{ type: DataType.FLOAT, shape: [2, 3, 2] }]);
    if (out.is(TensorNode)) out.as(TensorNode).setType("output");
    return g;
}

function makeTransposeAdd(): OnnxGraph.Class {
    const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const b = new GraphBuilder(g, "tp");
    const x = g
        .addNode("X")
        .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
        .as(TensorNode);
    const y = g
        .addNode("Y")
        .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
        .as(TensorNode);
    const [t] = b.createOp("Transpose", [x], { perm: [1, 0] }, [
        { type: DataType.FLOAT, shape: [3, 2] },
    ]);
    const [out] = b.createOp("Add", [t, y], {}, [{ type: DataType.FLOAT, shape: [3, 2] }]);
    if (out.is(TensorNode)) out.as(TensorNode).setType("output");
    return g;
}

function makeConvRelu(): OnnxGraph.Class {
    const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const b = new GraphBuilder(g, "cv");
    const x = g
        .addNode("X")
        .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 4, 4], "input"))
        .as(TensorNode);
    const w = g
        .addNode("W")
        .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
        .as(TensorNode);
    const [c] = b.createOp("Conv", [x, w], { strides: [1, 1], pads: [0, 0, 0, 0] }, [
        { type: DataType.FLOAT, shape: [1, 1, 2, 2] },
    ]);
    const [out] = b.createOp("Relu", [c], {}, [{ type: DataType.FLOAT, shape: [1, 1, 2, 2] }]);
    if (out.is(TensorNode)) out.as(TensorNode).setType("output");
    return g;
}

describe("Integration: Compiler Options & Pipeline Combinations (ORT Equivalence)", () => {
    let tmpDir: string;

    beforeAll(() => {
        initializeSchemaRegistry();
        ensureProtoExists();
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-opts-"));
    });

    afterAll(() => {
        if (tmpDir && fs.existsSync(tmpDir)) {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    async function assertTransformMatchesOrt(
        tag: string,
        tc: PipelineCase,
        opts: Partial<DecompositionOptions>,
        inspectTransformed?: (g: OnnxGraph.Class) => void,
    ) {
        const feeds = buildFeeds(tc.feeds, tag);
        const origGraph = tc.build();
        const origPath = path.join(tmpDir, `${tag}_orig.onnx`);
        await exportGraphToOnnx(origGraph, origPath);
        const origOuts = await runOrtChecked(`${tag}:orig`, origPath, feeds);

        const loweredGraph = tc.build();
        new OnnxGraphTransformer(opts).apply(loweredGraph, false);
        expect(() => validateGraph(loweredGraph)).not.toThrow();
        if (inspectTransformed) inspectTransformed(loweredGraph);

        const recPath = path.join(tmpDir, `${tag}_rec.onnx`);
        await exportGraphToOnnx(loweredGraph, recPath);
        const recOuts = await runOrtChecked(`${tag}:rec`, recPath, feeds);

        const cmp = compareOrtOutputs(origOuts, recOuts, { tol: tc.tol ?? 1e-4 });
        expect(cmp.passed, `${tag}: ${cmp.message ?? ""}`).toBe(true);
    }

    describe("Loop Tiling (tileSize > 1)", () => {
        it.each([IndexingMode.Pointer, IndexingMode.Radix])(
            "matches ORT for static exact & non-divisible tiled loops in %s mode",
            async (indexingMode) => {
                // Non-divisible: 2x5 = 10 elems, tileSize = 4 -> tiles (4, 4, 2)
                await assertTransformMatchesOrt(
                    `tile_tail_${indexingMode}`,
                    {
                        label: "tail",
                        feeds: [
                            { name: "X", dtype: "float32", shape: [2, 5], gen: "negmix" },
                            { name: "Y", dtype: "float32", shape: [2, 5], gen: "negmix" },
                        ],
                        build: makeSimpleAddChain,
                    },
                    {
                        canonicalize: true,
                        loopLowering: true,
                        fuse: true,
                        optimize: true,
                        indexingMode,
                        tileSize: 4,
                    },
                    (g) => {
                        const outerLoops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(outerLoops.length).toBe(1);
                        const innerLoops = outerLoops[0].regions[0]
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(innerLoops.length).toBe(1);
                    },
                );

                // Exact multiple: 2x4 = 8 elems, tileSize = 4 -> tiles (4, 4)
                await assertTransformMatchesOrt(
                    `tile_exact_${indexingMode}`,
                    {
                        label: "exact",
                        feeds: [
                            { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
                            { name: "W", dtype: "float32", shape: [3, 4], gen: "negmix" },
                            { name: "B", dtype: "float32", shape: [2, 4], gen: "negmix" },
                        ],
                        build: makeMatMulAddRelu,
                    },
                    {
                        canonicalize: true,
                        loopLowering: true,
                        fuse: true,
                        coalesce: true,
                        optimize: true,
                        indexingMode,
                        tileSize: 4,
                    },
                );
            },
        );

        it("matches ORT for dynamic-shaped tiled loops", async () => {
            await assertTransformMatchesOrt(
                "tile_dynamic",
                {
                    label: "dyn_tile",
                    feeds: [
                        { name: "X", dtype: "float32", shape: [3, 4], gen: "negmix" },
                        { name: "Y", dtype: "float32", shape: [3, 4], gen: "negmix" },
                    ],
                    build: () => {
                        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                        const b = new GraphBuilder(g, "dt");
                        const x = g
                            .addNode("X")
                            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                            .as(TensorNode);
                        const y = g
                            .addNode("Y")
                            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                            .as(TensorNode);
                        const [out] = b.createOp("Add", [x, y], {}, [
                            { type: DataType.FLOAT, shape: [-1, 4] },
                        ]);
                        if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                        return g;
                    },
                },
                {
                    canonicalize: true,
                    loopLowering: true,
                    optimize: true,
                    indexingMode: IndexingMode.Pointer,
                    tileSize: 5,
                },
            );
        });
    });

    describe("Optimization Passes (optimize: true)", () => {
        it("runs DCE and ShapeOptimization end-to-end and matches ORT", async () => {
            await assertTransformMatchesOrt(
                "optimize_true_full",
                {
                    label: "opt_full",
                    feeds: [
                        { name: "A", dtype: "float32", shape: [3, 1], gen: "negmix" },
                        { name: "B", dtype: "float32", shape: [1, 4], gen: "negmix" },
                        { name: "C", dtype: "float32", shape: [1, 1], gen: "negmix" },
                    ],
                    build: () => {
                        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                        const b = new GraphBuilder(g, "opt");
                        const a = g
                            .addNode("A")
                            .init(new TensorNode.Builder(DataType.FLOAT, [3, 1], "input"))
                            .as(TensorNode);
                        const bIn = g
                            .addNode("B")
                            .init(new TensorNode.Builder(DataType.FLOAT, [1, 4], "input"))
                            .as(TensorNode);
                        const cIn = g
                            .addNode("C")
                            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1], "input"))
                            .as(TensorNode);

                        // Dead branch that DCE should clean up
                        b.createOp("Relu", [a], {}, [{ type: DataType.FLOAT, shape: [3, 1] }]);

                        // Outer product MatMul [3,1] x [1,4] -> ShapeOptimization rewrites to Mul
                        const [mm] = b.createOp("MatMul", [a, bIn], {}, [
                            { type: DataType.FLOAT, shape: [3, 4] },
                        ]);
                        // 1x1 Add -> ShapeOptimization scalarizes
                        const [c1x1] = b.createOp("Add", [cIn, cIn], {}, [
                            { type: DataType.FLOAT, shape: [1, 1] },
                        ]);
                        // Static Shape + Cast -> ShapeOptimization folds Shape
                        const [shp] = b.createOp("Shape", [mm], {}, [
                            { type: DataType.INT64, shape: [2] },
                        ]);
                        const idx0 = b.createConstant("idx0", int64Vec([0]));
                        const [dim0] = b.createOp("Gather", [shp, idx0], { axis: 0 }, [
                            { type: DataType.INT64, shape: [1] },
                        ]);
                        const [dimF] = b.createOp("Cast", [dim0], { to: DataType.FLOAT }, [
                            { type: DataType.FLOAT, shape: [1] },
                        ]);

                        const [sum1] = b.createOp("Add", [mm, c1x1], {}, [
                            { type: DataType.FLOAT, shape: [3, 4] },
                        ]);
                        const [out] = b.createOp("Add", [sum1, dimF], {}, [
                            { type: DataType.FLOAT, shape: [3, 4] },
                        ]);
                        if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                        return g;
                    },
                },
                { canonicalize: true, loopLowering: true, fuse: true, optimize: true },
                (g) => {
                    const topOps = g
                        .getOperationNodes()
                        .toArray()
                        .map((o) => o.type);
                    expect(topOps).not.toContain("Shape");
                    expect(topOps).not.toContain("MatMul");
                },
            );
        });
    });

    describe("Coalescing (coalesce: false vs true) & Fusion (fuse: false vs true)", () => {
        it.each([IndexingMode.Pointer, IndexingMode.Radix])(
            "matches ORT with coalesce: false (2D & batched MatMul) in %s mode",
            async (indexingMode) => {
                await assertTransformMatchesOrt(
                    `no_coalesce_2d_${indexingMode}`,
                    {
                        label: "mm_2d",
                        feeds: [
                            { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
                            { name: "W", dtype: "float32", shape: [3, 4], gen: "negmix" },
                            { name: "B", dtype: "float32", shape: [2, 4], gen: "negmix" },
                        ],
                        build: makeMatMulAddRelu,
                    },
                    { coalesce: false, fuse: true, indexingMode },
                );

                await assertTransformMatchesOrt(
                    `no_coalesce_3d_${indexingMode}`,
                    {
                        label: "mm_3d",
                        feeds: [
                            { name: "A", dtype: "float32", shape: [2, 3, 4], gen: "negmix" },
                            { name: "W", dtype: "float32", shape: [1, 4, 2], gen: "negmix" },
                        ],
                        build: makeBatchedMatMul,
                    },
                    { coalesce: false, fuse: false, indexingMode },
                );
            },
        );

        it("matches ORT with fuse: false across multi-op chains", async () => {
            await assertTransformMatchesOrt(
                "no_fuse_mm_add_relu",
                {
                    label: "unfused_mm",
                    feeds: [
                        { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        { name: "W", dtype: "float32", shape: [3, 4], gen: "negmix" },
                        { name: "B", dtype: "float32", shape: [2, 4], gen: "negmix" },
                    ],
                    build: makeMatMulAddRelu,
                },
                { fuse: false, coalesce: true, optimize: true },
                (g) => {
                    const loops = g
                        .getOperationNodes()
                        .toArray()
                        .filter((o) => o.type === "Loop");
                    expect(loops.length).toBeGreaterThanOrEqual(3);
                },
            );

            await assertTransformMatchesOrt(
                "no_fuse_tp_add",
                {
                    label: "unfused_tp",
                    feeds: [
                        { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        { name: "Y", dtype: "float32", shape: [3, 2], gen: "negmix" },
                    ],
                    build: makeTransposeAdd,
                },
                { fuse: false, optimize: true },
                (g) => {
                    const loops = g
                        .getOperationNodes()
                        .toArray()
                        .filter((o) => o.type === "Loop");
                    expect(loops.length).toBe(2);
                },
            );
        });
    });

    describe("Pass Toggles (canonicalize: false, loopLowering: false, indexingMode: radix, decomposeForCgra)", () => {
        it("matches ORT with canonicalize: false (direct LoopLowering on Conv+Relu)", async () => {
            await assertTransformMatchesOrt(
                "no_canon_conv_relu",
                {
                    label: "no_canon",
                    feeds: [
                        { name: "X", dtype: "float32", shape: [1, 1, 4, 4], gen: "negmix" },
                        { name: "W", dtype: "float32", shape: [1, 1, 3, 3], gen: "negmix" },
                    ],
                    build: makeConvRelu,
                },
                { canonicalize: false, loopLowering: true, fuse: true, optimize: true },
                (g) => {
                    const ops = g
                        .getOperationNodes()
                        .toArray()
                        .map((o) => o.type);
                    expect(ops).toContain("Loop");
                    expect(ops).not.toContain("Conv");
                },
            );
        });

        it("matches ORT with loopLowering: false (canonicalization-only, no Loop nodes)", async () => {
            await assertTransformMatchesOrt(
                "no_loop_lowering",
                {
                    label: "canon_only",
                    feeds: [
                        { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        { name: "B", dtype: "float32", shape: [3, 4], gen: "negmix" },
                        { name: "C", dtype: "float32", shape: [2, 4], gen: "negmix" },
                    ],
                    build: () => {
                        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                        const b = new GraphBuilder(g, "co");
                        const a = g
                            .addNode("A")
                            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                            .as(TensorNode);
                        const w = g
                            .addNode("B")
                            .init(new TensorNode.Builder(DataType.FLOAT, [3, 4], "input"))
                            .as(TensorNode);
                        const c = g
                            .addNode("C")
                            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                            .as(TensorNode);
                        const [gemm] = b.createOp("Gemm", [a, w, c], { alpha: 1, beta: 1 }, [
                            { type: DataType.FLOAT, shape: [2, 4] },
                        ]);
                        const [out] = b.createOp("Softmax", [gemm], { axis: -1 }, [
                            { type: DataType.FLOAT, shape: [2, 4] },
                        ]);
                        if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                        return g;
                    },
                },
                { canonicalize: true, loopLowering: false, optimize: true },
                (g) => {
                    const ops = g
                        .getOperationNodes()
                        .toArray()
                        .map((o) => o.type);
                    expect(ops).not.toContain("Loop");
                    expect(ops).not.toContain("Gemm");
                    expect(ops).not.toContain("Softmax");
                },
            );
        });

        it("matches ORT in full IndexingMode.Radix pipeline for Conv+Relu and Transpose+Add", async () => {
            await assertTransformMatchesOrt(
                "radix_conv_relu",
                {
                    label: "radix_conv",
                    feeds: [
                        { name: "X", dtype: "float32", shape: [1, 1, 4, 4], gen: "negmix" },
                        { name: "W", dtype: "float32", shape: [1, 1, 3, 3], gen: "negmix" },
                    ],
                    build: makeConvRelu,
                },
                {
                    canonicalize: true,
                    loopLowering: true,
                    fuse: true,
                    optimize: true,
                    indexingMode: IndexingMode.Radix,
                },
            );

            await assertTransformMatchesOrt(
                "radix_tp_add",
                {
                    label: "radix_tp",
                    feeds: [
                        { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        { name: "Y", dtype: "float32", shape: [3, 2], gen: "negmix" },
                    ],
                    build: makeTransposeAdd,
                },
                {
                    canonicalize: true,
                    loopLowering: true,
                    fuse: true,
                    optimize: true,
                    indexingMode: IndexingMode.Radix,
                },
            );
        });

        it("matches ORT with decomposeForCgra: true on 2D MatMul chain", async () => {
            await assertTransformMatchesOrt(
                "decompose_for_cgra",
                {
                    label: "cgra_mm",
                    feeds: [
                        { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        { name: "W", dtype: "float32", shape: [3, 2], gen: "negmix" },
                        { name: "B", dtype: "float32", shape: [2, 2], gen: "negmix" },
                    ],
                    build: () => {
                        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                        const b = new GraphBuilder(g, "cgra");
                        const a = g
                            .addNode("A")
                            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                            .as(TensorNode);
                        const w = g
                            .addNode("W")
                            .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
                            .as(TensorNode);
                        const bias = g
                            .addNode("B")
                            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
                            .as(TensorNode);
                        const [aId] = b.createOp("Identity", [a], {}, [
                            { type: DataType.FLOAT, shape: [2, 3] },
                        ]);
                        const [wId] = b.createOp("Identity", [w], {}, [
                            { type: DataType.FLOAT, shape: [3, 2] },
                        ]);
                        const [mm] = b.createOp("MatMul", [aId, wId], {}, [
                            { type: DataType.FLOAT, shape: [2, 2] },
                        ]);
                        const [out] = b.createOp("Add", [mm, bias], {}, [
                            { type: DataType.FLOAT, shape: [2, 2] },
                        ]);
                        if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                        return g;
                    },
                },
                { decomposeForCgra: true },
                (g) => {
                    const ops = g
                        .getOperationNodes()
                        .toArray()
                        .map((o) => o.type);
                    expect(ops).not.toContain("MatMul");
                    expect(ops).toContain("ReduceSum");
                },
            );
        });
    });
});
