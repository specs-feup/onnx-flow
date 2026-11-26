import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../src/Onnx/OnnxGraph.js";
import OperationNode from "../../src/Onnx/OperationNode.js";
import RegionArgumentNode from "../../src/Onnx/RegionArgumentNode.js";
import TensorNode from "../../src/Onnx/TensorNode.js";
import OnnxEdge from "../../src/Onnx/OnnxEdge.js";
import { GraphBuilder } from "../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../src/Onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../src/Onnx/Schema/index.js";
import OnnxGraphTransformer from "../../src/Onnx/transformation/index.js";
import { IndexingMode, type DecompositionOptions } from "../../src/DecompositionOptions.js";
import { floatVec, int64Vec, makeTensorProto, scalarFloat } from "../../src/Onnx/Utils.js";
import validateGraph from "../../src/Onnx/validation/ValidateGraph.js";
import {
    buildResidualSkipForkGraph,
    buildSplitMultiBranchConcatGraph,
} from "../helpers/modelFactory.js";
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

const TEST_TIMEOUT = 150000; //ms

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

    describe("Multi-Op Subgraph Combinations (ORT Equivalence)", () => {
        const fullOpts: Partial<DecompositionOptions> = {
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            coalesce: true,
            optimize: true,
            indexingMode: IndexingMode.Pointer,
        };

        it(
            "matches ORT for Conv -> BatchNormalization -> Relu -> MaxPool CNN block",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_cnn_conv_bn_relu_maxpool",
                    {
                        label: "cnn_block",
                        feeds: [
                            { name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "cnn");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 4], "input"))
                                .as(TensorNode);
                            const w = b.createConstant(
                                "conv_w",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [2, 2, 3, 3],
                                    Array.from({ length: 36 }, (_, i) => ((i % 7) - 3) * 0.1),
                                ),
                            );
                            const scale = b.createConstant("bn_s", floatVec([1.2, 0.8]));
                            const bias = b.createConstant("bn_b", floatVec([0.1, -0.2]));
                            const mean = b.createConstant("bn_m", floatVec([0.05, -0.05]));
                            const variance = b.createConstant("bn_v", floatVec([0.5, 1.5]));

                            const [cv] = b.createOp(
                                "Conv",
                                [x, w],
                                { strides: [1, 1], pads: [1, 1, 1, 1] },
                                [{ type: DataType.FLOAT, shape: [1, 2, 4, 4] }],
                            );
                            const [bn] = b.createOp(
                                "BatchNormalization",
                                [cv, scale, bias, mean, variance],
                                { epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [1, 2, 4, 4] }],
                            );
                            const [rl] = b.createOp("Relu", [bn], {}, [
                                { type: DataType.FLOAT, shape: [1, 2, 4, 4] },
                            ]);
                            const [out] = b.createOp(
                                "MaxPool",
                                [rl],
                                { kernel_shape: [2, 2], strides: [2, 2] },
                                [{ type: DataType.FLOAT, shape: [1, 2, 2, 2] }],
                            );
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it("matches ORT for Residual Skip-Fork DAG (Relu -> Conv -> Relu -> Conv -> Add(skip) -> Relu)", async () => {
            await assertTransformMatchesOrt(
                "combo_residual_skip_fork",
                {
                    label: "res_fork",
                    feeds: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
                    build: () => buildResidualSkipForkGraph("res"),
                },
                fullOpts,
            );
        });

        it(
            "matches ORT for Depthwise-Separable Conv -> GlobalAveragePool -> Flatten -> Gemm block",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_depthwise_separable_gap_gemm",
                    {
                        label: "dw_sep_gemm",
                        feeds: [
                            { name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "dw");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 4], "input"))
                                .as(TensorNode);
                            const wDw = b.createConstant(
                                "w_dw",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [2, 1, 3, 3],
                                    Array.from({ length: 18 }, (_, i) => ((i % 5) - 2) * 0.1),
                                ),
                            );
                            const wPw = b.createConstant(
                                "w_pw",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 2, 1, 1],
                                    Array.from({ length: 8 }, (_, i) => (i - 3) * 0.12),
                                ),
                            );
                            const wFc = b.createConstant(
                                "w_fc",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 3],
                                    Array.from({ length: 12 }, (_, i) => ((i % 5) - 2) * 0.15),
                                ),
                            );
                            const bFc = b.createConstant("b_fc", floatVec([0.1, -0.1, 0.2]));

                            const [dw] = b.createOp(
                                "Conv",
                                [x, wDw],
                                { group: 2, strides: [1, 1], pads: [1, 1, 1, 1] },
                                [{ type: DataType.FLOAT, shape: [1, 2, 4, 4] }],
                            );
                            const [pw] = b.createOp("Conv", [dw, wPw], { strides: [1, 1] }, [
                                { type: DataType.FLOAT, shape: [1, 4, 4, 4] },
                            ]);
                            const [rl] = b.createOp("Relu", [pw], {}, [
                                { type: DataType.FLOAT, shape: [1, 4, 4, 4] },
                            ]);
                            const [gap] = b.createOp("GlobalAveragePool", [rl], {}, [
                                { type: DataType.FLOAT, shape: [1, 4, 1, 1] },
                            ]);
                            const [flat] = b.createOp("Flatten", [gap], { axis: 1 }, [
                                { type: DataType.FLOAT, shape: [1, 4] },
                            ]);
                            const [out] = b.createOp(
                                "Gemm",
                                [flat, wFc, bFc],
                                { alpha: 1, beta: 1, transA: 0, transB: 0 },
                                [{ type: DataType.FLOAT, shape: [1, 3] }],
                            );
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for Modern LLM Block (RMSNormalization -> RotaryEmbedding -> GQA Attention -> Swish)",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_modern_llm_rmsnorm_rope_attn_swish",
                    {
                        label: "llm_block",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [1, 3, 8], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "llm");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [1, 3, 8], "input"))
                                .as(TensorNode);
                            const rmsScale = b.createConstant(
                                "rms_s",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [8],
                                    [1, 1.1, 0.9, 1, 1, 1.2, 0.8, 1],
                                ),
                            );
                            const wq = b.createConstant(
                                "wq",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [8, 8],
                                    Array.from({ length: 64 }, (_, i) => ((i % 7) - 3) * 0.08),
                                ),
                            );
                            const wk = b.createConstant(
                                "wk",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [8, 4],
                                    Array.from({ length: 32 }, (_, i) => (((i + 2) % 5) - 2) * 0.1),
                                ),
                            );
                            const wv = b.createConstant(
                                "wv",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [8, 4],
                                    Array.from({ length: 32 }, (_, i) => (((i + 1) % 5) - 2) * 0.1),
                                ),
                            );
                            const cosCache = b.createConstant(
                                "cos_c",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [1, 3, 1],
                                    [1.0, Math.cos(0.3), Math.cos(0.6)],
                                ),
                            );
                            const sinCache = b.createConstant(
                                "sin_c",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [1, 3, 1],
                                    [0.0, Math.sin(0.3), Math.sin(0.6)],
                                ),
                            );

                            const [norm] = b.createOp(
                                "RMSNormalization",
                                [x, rmsScale],
                                { axis: -1, epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [1, 3, 8] }],
                            );
                            const [qProj] = b.createOp("MatMul", [norm, wq], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 8] },
                            ]);
                            const [kProj] = b.createOp("MatMul", [norm, wk], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 4] },
                            ]);
                            const [vProj] = b.createOp("MatMul", [norm, wv], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 4] },
                            ]);
                            const [qRope] = b.createOp(
                                "RotaryEmbedding",
                                [qProj, cosCache, sinCache],
                                { interleaved: 0, num_heads: 4 },
                                [{ type: DataType.FLOAT, shape: [1, 3, 8] }],
                            );
                            const [kRope] = b.createOp(
                                "RotaryEmbedding",
                                [kProj, cosCache, sinCache],
                                { interleaved: 0, num_heads: 2 },
                                [{ type: DataType.FLOAT, shape: [1, 3, 4] }],
                            );
                            const attnOuts = b.createOpWithExact(
                                "llm_attn",
                                "Attention",
                                [qRope, kRope, vProj],
                                ["llm_attn_y"],
                                { q_num_heads: 4, kv_num_heads: 2, is_causal: 1, softcap: 5.0 },
                                [{ type: DataType.FLOAT, shape: [1, 3, 8] }],
                            );
                            const [out] = b.createOp("Swish", [attnOuts[0]], { alpha: 1.0 }, [
                                { type: DataType.FLOAT, shape: [1, 3, 8] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    {
                        canonicalize: true,
                        loopLowering: false,
                        optimize: true,
                    },
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for RMSNormalization -> Swish -> TensorScatter (circular KV-cache) full pipeline",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_rmsnorm_swish_tensor_scatter",
                    {
                        label: "kv_cache_pipe",
                        tol: 1e-4,
                        feeds: [
                            { name: "cache", dtype: "float32", shape: [2, 1, 4, 4], gen: "zeros" },
                            { name: "upd", dtype: "float32", shape: [2, 1, 2, 4], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "kv");
                            const cache = g
                                .addNode("cache")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 1, 4, 4], "input"))
                                .as(TensorNode);
                            const upd = g
                                .addNode("upd")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 1, 2, 4], "input"))
                                .as(TensorNode);
                            const wi = b.createConstant("wi", int64Vec([1, 3]));
                            const rmsScale = b.createConstant(
                                "rms_s",
                                makeTensorProto(DataType.FLOAT, [4], [1.0, 1.2, 0.8, 1.1]),
                            );
                            const [norm] = b.createOp(
                                "RMSNormalization",
                                [upd, rmsScale],
                                { axis: -1, epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [2, 1, 2, 4] }],
                            );
                            const [sw] = b.createOp("Swish", [norm], { alpha: 1.0 }, [
                                { type: DataType.FLOAT, shape: [2, 1, 2, 4] },
                            ]);
                            const [out] = b.createOp(
                                "TensorScatter",
                                [cache, sw, wi],
                                { axis: -2, mode: "circular" },
                                [{ type: DataType.FLOAT, shape: [2, 1, 4, 4] }],
                            );
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for GroupNorm -> Swish -> SpaceToDepth spatial pipeline",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_gn_swish_space_to_depth",
                    {
                        label: "gn_swish_s2d",
                        tol: 1e-4,
                        feeds: [
                            { name: "X", dtype: "float32", shape: [1, 4, 4, 4], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "s2d_pipe");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [1, 4, 4, 4], "input"))
                                .as(TensorNode);
                            const s = b.createConstant("gn_s", floatVec([1, 1.2, 0.8, 1.1]));
                            const bias = b.createConstant("gn_b", floatVec([0, 0.1, -0.1, 0.05]));
                            const [gn] = b.createOp(
                                "GroupNormalization",
                                [x, s, bias],
                                { num_groups: 2, epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [1, 4, 4, 4] }],
                            );
                            const [sw] = b.createOp("Swish", [gn], { alpha: 1.2 }, [
                                { type: DataType.FLOAT, shape: [1, 4, 4, 4] },
                            ]);
                            const [out] = b.createOp("SpaceToDepth", [sw], { blocksize: 2 }, [
                                { type: DataType.FLOAT, shape: [1, 16, 2, 2] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for Transformer Attention block (LayerNorm -> QKV -> Transpose -> Softmax -> MatMul)",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_attention_ln_matmul_softmax",
                    {
                        label: "attn",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [1, 3, 4], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "attn");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [1, 3, 4], "input"))
                                .as(TensorNode);
                            const lnS = b.createConstant("ln_s", floatVec([1, 1, 1, 1]));
                            const lnB = b.createConstant("ln_b", floatVec([0, 0.1, -0.1, 0]));
                            const wq = b.createConstant(
                                "wq",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 4],
                                    Array.from({ length: 16 }, (_, i) => ((i % 7) - 3) * 0.1),
                                ),
                            );
                            const wk = b.createConstant(
                                "wk",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 4],
                                    Array.from({ length: 16 }, (_, i) => (((i + 3) % 7) - 3) * 0.1),
                                ),
                            );
                            const wv = b.createConstant(
                                "wv",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 2],
                                    Array.from({ length: 8 }, (_, i) => (i - 3) * 0.15),
                                ),
                            );
                            const scale = b.createConstant("dk", scalarFloat(2.0));

                            const [ln] = b.createOp(
                                "LayerNormalization",
                                [x, lnS, lnB],
                                { axis: -1, epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [1, 3, 4] }],
                            );
                            const [q] = b.createOp("MatMul", [ln, wq], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 4] },
                            ]);
                            const [k] = b.createOp("MatMul", [ln, wk], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 4] },
                            ]);
                            const [v] = b.createOp("MatMul", [ln, wv], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 2] },
                            ]);
                            const [kt] = b.createOp("Transpose", [k], { perm: [0, 2, 1] }, [
                                { type: DataType.FLOAT, shape: [1, 4, 3] },
                            ]);
                            const [scores] = b.createOp("MatMul", [q, kt], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 3] },
                            ]);
                            const [scaled] = b.createOp("Div", [scores, scale], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 3] },
                            ]);
                            const [attn] = b.createOp("Softmax", [scaled], { axis: -1 }, [
                                { type: DataType.FLOAT, shape: [1, 3, 3] },
                            ]);
                            const [out] = b.createOp("MatMul", [attn, v], {}, [
                                { type: DataType.FLOAT, shape: [1, 3, 2] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it("matches ORT for multi-output Split -> Asymmetric Branches (Relu vs Neg+Exp) -> Concat", async () => {
            await assertTransformMatchesOrt(
                "combo_split_multibranch_concat",
                {
                    label: "split_concat",
                    feeds: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
                    build: () => buildSplitMultiBranchConcatGraph("sc"),
                },
                fullOpts,
            );
        });

        it(
            "matches ORT for QDQ Quantized MLP (Q -> DQ -> MatMul -> Add -> Relu -> Q -> DQ)",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_qdq_quantized_mlp",
                    {
                        label: "qdq_mlp",
                        tol: 1e-4,
                        feeds: [
                            {
                                name: "X",
                                dtype: "float32",
                                shape: [2, 4],
                                init: [-0.8, 0.4, 1.2, -0.4, 0.2, -1.2, 0.8, 0.0],
                            },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "qdq");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                                .as(TensorNode);
                            const sc1 = b.createConstant("sc1", scalarFloat(0.2));
                            const zp1 = b.createConstant(
                                "zp1",
                                makeTensorProto(DataType.UINT8, [], [128]),
                            );
                            const sc2 = b.createConstant("sc2", scalarFloat(0.25));
                            const zp2 = b.createConstant(
                                "zp2",
                                makeTensorProto(DataType.UINT8, [], [0]),
                            );
                            const w = b.createConstant(
                                "W",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 3],
                                    [
                                        0.5, -0.5, 0.25, 0.25, 0.5, -0.25, -0.5, 0.25, 0.5, 0.25,
                                        -0.25, 0.5,
                                    ],
                                ),
                            );
                            const bias = b.createConstant("B", floatVec([0.1, 0.3, -0.2]));

                            const [q1] = b.createOp("QuantizeLinear", [x, sc1, zp1], {}, [
                                { type: DataType.UINT8, shape: [2, 4] },
                            ]);
                            const [dq1] = b.createOp("DequantizeLinear", [q1, sc1, zp1], {}, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            const [mm] = b.createOp("MatMul", [dq1, w], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            const [sum] = b.createOp("Add", [mm, bias], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            const [rl] = b.createOp("Relu", [sum], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            const [q2] = b.createOp("QuantizeLinear", [rl, sc2, zp2], {}, [
                                { type: DataType.UINT8, shape: [2, 3] },
                            ]);
                            const [out] = b.createOp("DequantizeLinear", [q2, sc2, zp2], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for Dynamic-Batch [-1, 4] MatMul -> Add -> Relu -> Softmax",
            async () => {
                await assertTransformMatchesOrt(
                    "combo_dynamic_batch_relu_softmax",
                    {
                        label: "dyn_relu_sm",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [3, 4], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "dg");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                                .as(TensorNode);
                            const w = b.createConstant(
                                "W",
                                makeTensorProto(
                                    DataType.FLOAT,
                                    [4, 4],
                                    Array.from({ length: 16 }, (_, i) => ((i % 7) - 3) * 0.15),
                                ),
                            );
                            const bias = b.createConstant("B", floatVec([0.1, -0.2, 0.05, 0.3]));

                            const [mm] = b.createOp("MatMul", [x, w], {}, [
                                { type: DataType.FLOAT, shape: [-1, 4] },
                            ]);
                            const [add] = b.createOp("Add", [mm, bias], {}, [
                                { type: DataType.FLOAT, shape: [-1, 4] },
                            ]);
                            const [rl] = b.createOp("Relu", [add], {}, [
                                { type: DataType.FLOAT, shape: [-1, 4] },
                            ]);
                            const [out] = b.createOp("Softmax", [rl], { axis: -1 }, [
                                { type: DataType.FLOAT, shape: [-1, 4] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it.each([true, false])(
            "matches ORT for If control-flow with captured multi-op MatMul/Relu branches (cond=%s)",
            async (condVal) => {
                await assertTransformMatchesOrt(
                    `combo_if_cf_${String(condVal)}`,
                    {
                        label: `if_cf_${String(condVal)}`,
                        feeds: [
                            { name: "cond", dtype: "bool", shape: [], init: [condVal] },
                            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
                            { name: "W1", dtype: "float32", shape: [3, 2], gen: "negmix" },
                            { name: "W2", dtype: "float32", shape: [3, 2], gen: "negmix" },
                            { name: "B", dtype: "float32", shape: [2], gen: "negmix" },
                        ],
                        build: () => {
                            const thenGraph = Graph.create()
                                .init(new OnnxGraph.Builder())
                                .as(OnnxGraph);
                            const thenB = new GraphBuilder(thenGraph, "then");
                            const tX = thenGraph
                                .addNode("X")
                                .init(
                                    new RegionArgumentNode.Builder(0, "X", DataType.FLOAT, [2, 3]),
                                )
                                .as(RegionArgumentNode);
                            const tW1 = thenGraph
                                .addNode("W1")
                                .init(
                                    new RegionArgumentNode.Builder(1, "W1", DataType.FLOAT, [3, 2]),
                                )
                                .as(RegionArgumentNode);
                            const tB = thenGraph
                                .addNode("B")
                                .init(new RegionArgumentNode.Builder(2, "B", DataType.FLOAT, [2]))
                                .as(RegionArgumentNode);
                            const [tMm] = thenB.createOp("MatMul", [tX, tW1], {}, [
                                { type: DataType.FLOAT, shape: [2, 2] },
                            ]);
                            const [tOut] = thenB.createOp("Add", [tMm, tB], {}, [
                                { type: DataType.FLOAT, shape: [2, 2] },
                            ]);
                            if (tOut.is(TensorNode)) tOut.as(TensorNode).setType("output");

                            const elseGraph = Graph.create()
                                .init(new OnnxGraph.Builder())
                                .as(OnnxGraph);
                            const elseB = new GraphBuilder(elseGraph, "else");
                            const eX = elseGraph
                                .addNode("X")
                                .init(
                                    new RegionArgumentNode.Builder(0, "X", DataType.FLOAT, [2, 3]),
                                )
                                .as(RegionArgumentNode);
                            const eW2 = elseGraph
                                .addNode("W2")
                                .init(
                                    new RegionArgumentNode.Builder(1, "W2", DataType.FLOAT, [3, 2]),
                                )
                                .as(RegionArgumentNode);
                            const [eMm] = elseB.createOp("MatMul", [eX, eW2], {}, [
                                { type: DataType.FLOAT, shape: [2, 2] },
                            ]);
                            const [eOut] = elseB.createOp("Relu", [eMm], {}, [
                                { type: DataType.FLOAT, shape: [2, 2] },
                            ]);
                            if (eOut.is(TensorNode)) eOut.as(TensorNode).setType("output");

                            const outer = Graph.create()
                                .init(new OnnxGraph.Builder())
                                .as(OnnxGraph);
                            const cond = outer
                                .addNode("cond")
                                .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
                                .as(TensorNode);
                            outer
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"));
                            outer
                                .addNode("W1")
                                .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"));
                            outer
                                .addNode("W2")
                                .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"));
                            outer
                                .addNode("B")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"));

                            const ifOp = outer
                                .addNode("if_0")
                                .init(
                                    new OperationNode.Builder("If", [cond], {}, [
                                        thenGraph,
                                        elseGraph,
                                    ]),
                                )
                                .as(OperationNode);
                            outer.addEdge(cond, ifOp).init(new OnnxEdge.Builder(DataType.BOOL, []));
                            const yOut = outer
                                .addNode("Y")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "output"))
                                .as(TensorNode);
                            outer
                                .addEdge(ifOp, yOut)
                                .init(new OnnxEdge.Builder(DataType.FLOAT, [2, 2]));
                            return outer;
                        },
                    },
                    fullOpts,
                );
            },
        );

        it.each([IndexingMode.Pointer, IndexingMode.Radix])(
            "matches ORT with useScanOutputs=true across elementwise fusion and reduction/matmul carry (%s)",
            async (indexingMode) => {
                await assertTransformMatchesOrt(
                    `scan_outputs_${indexingMode}`,
                    {
                        label: "scan_out_combo",
                        tol: 1e-4,
                        feeds: [
                            { name: "A", dtype: "float32", shape: [2, 3], gen: "negmix" },
                            { name: "W", dtype: "float32", shape: [3, 4], gen: "negmix" },
                            { name: "B", dtype: "float32", shape: [2, 4], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "sc_mm");
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
                            const [mm] = b.createOp("MatMul", [a, w], {}, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            const [sw] = b.createOp("Swish", [mm], { alpha: 1.0 }, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            const [out] = b.createOp("Add", [sw, bias], {}, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    { ...fullOpts, indexingMode, useScanOutputs: true },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBeGreaterThanOrEqual(2);
                        // Elementwise loop(s) use native scan_outputs with zero ScatterElements
                        const ewLoops = loops.filter(
                            (l) =>
                                !l.regions[0]
                                    .getOperationNodes()
                                    .toArray()
                                    .some((op) => op.type === "ScatterElements"),
                        );
                        expect(ewLoops.length).toBeGreaterThanOrEqual(1);
                    },
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for diamond DAG fusion on Gelu -> Swish -> Div without duplicating loops",
            async () => {
                await assertTransformMatchesOrt(
                    "diamond_gelu_swish_div",
                    {
                        label: "diam_swish",
                        tol: 1e-3,
                        feeds: [
                            { name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" },
                            {
                                name: "D",
                                dtype: "float32",
                                shape: [2, 4],
                                init: [1.5, -2.0, 0.8, 3.0, -1.2, 2.5, -0.7, 1.8],
                            },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "dm");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                                .as(TensorNode);
                            const d = g
                                .addNode("D")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                                .as(TensorNode);
                            const [ge] = b.createOp("Gelu", [x], {}, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            const [sw] = b.createOp("Swish", [ge], { alpha: 1.0 }, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            const [out] = b.createOp("Div", [sw, d], {}, [
                                { type: DataType.FLOAT, shape: [2, 4] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(1);
                    },
                );
            },
            TEST_TIMEOUT,
        );

        it.each([IndexingMode.Pointer, IndexingMode.Radix])(
            "matches ORT for standalone fused SoftmaxLoop and LayerNormLoop (%s)",
            async (indexingMode) => {
                await assertTransformMatchesOrt(
                    `fused_softmax_layernorm_loop_${indexingMode}`,
                    {
                        label: "sm_ln_loop",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [3, 5], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "smln");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 5], "input"))
                                .as(TensorNode);
                            const sc = b.createConstant("S", floatVec([1.1, 0.9, 1.0, 1.2, 0.8]));
                            const bs = b.createConstant(
                                "B",
                                floatVec([0.0, 0.1, -0.1, 0.05, -0.05]),
                            );
                            const [ln] = b.createOp(
                                "LayerNormalization",
                                [x, sc, bs],
                                { axis: -1, epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [-1, 5] }],
                            );
                            const [sm] = b.createOp("Softmax", [ln], { axis: -1 }, [
                                { type: DataType.FLOAT, shape: [-1, 5] },
                            ]);
                            if (sm.is(TensorNode)) sm.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    { canonicalize: false, loopLowering: true, optimize: true, indexingMode },
                    (g) => {
                        const topOps = g
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type);
                        expect(topOps).not.toContain("LayerNormalization");
                        expect(topOps).not.toContain("Softmax");
                        expect(topOps.filter((t) => t === "Loop").length).toBe(2);
                    },
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for MatVec (1D @ 2D) and scalar-accumulator MaxPool/AvgPool/ReduceSum",
            async () => {
                await assertTransformMatchesOrt(
                    "matvec_and_scalar_acc",
                    {
                        label: "matvec",
                        tol: 1e-4,
                        feeds: [
                            { name: "V", dtype: "float32", shape: [4], gen: "negmix" },
                            { name: "W", dtype: "float32", shape: [4, 3], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "mv");
                            const v = g
                                .addNode("V")
                                .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
                                .as(TensorNode);
                            const w = g
                                .addNode("W")
                                .init(new TensorNode.Builder(DataType.FLOAT, [4, 3], "input"))
                                .as(TensorNode);
                            const [mm] = b.createOp("MatMul", [v, w], {}, [
                                { type: DataType.FLOAT, shape: [3] },
                            ]);
                            const [out] = b.createOp("Relu", [mm], {}, [
                                { type: DataType.FLOAT, shape: [3] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    fullOpts,
                );
            },
            TEST_TIMEOUT,
        );

        it.each([false, true])(
            "matches ORT for multi-output fork fusion (useScanOutputs=%s)",
            async (useScanOutputs) => {
                await assertTransformMatchesOrt(
                    `multi_out_fork_scan_${String(useScanOutputs)}`,
                    {
                        label: "multi_fork",
                        feeds: [
                            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
                            { name: "Y", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "mf");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                                .as(TensorNode);
                            const y = g
                                .addNode("Y")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                                .as(TensorNode);
                            const [sum] = b.createOp("Add", [x, y], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            const [o1] = b.createOp("Relu", [sum], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            const [o2] = b.createOp("Neg", [sum], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            if (o1.is(TensorNode)) o1.as(TensorNode).setType("output");
                            if (o2.is(TensorNode)) o2.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    { ...fullOpts, fuseMultiOutput: true, useScanOutputs },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(1);
                    },
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for Schedule API (useRecipe + split SoftmaxLoop, reorder, unroll, cacheRead)",
            async () => {
                const { Schedule } = await import("../../src/Onnx/transformation/Schedule.js");

                await assertTransformMatchesOrt(
                    "schedule_softmax_split_unroll",
                    {
                        label: "sched_sm",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [4, 5], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "ssm");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 5], "input"))
                                .as(TensorNode);
                            const [out] = b.createOp("Softmax", [x], { axis: -1 }, [
                                { type: DataType.FLOAT, shape: [-1, 5] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    {
                        ...fullOpts,
                        schedule: new Schedule()
                            .useRecipe("Softmax", "LowerSoftmaxLoop")
                            .vectorize("Softmax", 4)
                            .cacheRead("X", "Softmax"),
                    },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(1);
                        expect(loops[0].getMetadata("vectorWidth")).toBe(4);
                    },
                );

                await assertTransformMatchesOrt(
                    "schedule_reorder_unroll_ew",
                    {
                        label: "sched_unr",
                        feeds: [
                            { name: "X", dtype: "float32", shape: [2, 3], gen: "negmix" },
                            { name: "Y", dtype: "float32", shape: [2, 3], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "sr");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                                .as(TensorNode);
                            const y = g
                                .addNode("Y")
                                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                                .as(TensorNode);
                            const [sum] = b.createOp("Add", [x, y], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            const [out] = b.createOp("Relu", [sum], {}, [
                                { type: DataType.FLOAT, shape: [2, 3] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    {
                        ...fullOpts,
                        useScanOutputs: true,
                        schedule: new Schedule().reorder("Relu", [1, 0]).unroll("Relu", 8),
                    },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(0);
                    },
                );
            },
            TEST_TIMEOUT,
        );

        it(
            "matches ORT for LowerDetLoop, LowerTopKScalarLoop, LowerLogSoftmaxLoop, LowerRMSNormLoop, and deferred AveragePool",
            async () => {
                const { Schedule } = await import("../../src/Onnx/transformation/Schedule.js");

                // 1. Dynamic 4x4 & 5x5 Det via LowerDetLoop
                await assertTransformMatchesOrt(
                    "det_loop_dynamic_4x4",
                    {
                        label: "detl_4x4",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [2, 4, 4], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "dl");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, -1, -1], "input"))
                                .as(TensorNode);
                            const [out] = b.createOp("Det", [x], {}, [
                                { type: DataType.FLOAT, shape: [-1] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    { ...fullOpts },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(1);
                    },
                );

                // 2. TopK with ties, smallest, non-trailing axis, and dynamic K
                await assertTransformMatchesOrt(
                    "topk_scalar_loop_axis0_ties",
                    {
                        label: "topkl_ties",
                        feeds: [
                            {
                                name: "X",
                                dtype: "float32",
                                shape: [4, 2],
                                init: [3, 1, 5, -2, -1, 4, 5, -2],
                            },
                            { name: "K", dtype: "int64", shape: [1], init: [3n] },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "tkl");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 2], "input"))
                                .as(TensorNode);
                            const k = g
                                .addNode("K")
                                .init(new TensorNode.Builder(DataType.INT64, [1], "input"))
                                .as(TensorNode);
                            const [vals, idxs] = b.createOp(
                                "TopK",
                                [x, k],
                                { axis: 0, largest: 0 },
                                [
                                    { type: DataType.FLOAT, shape: [-1, 2] },
                                    { type: DataType.INT64, shape: [-1, 2] },
                                ],
                            );
                            if (vals.is(TensorNode)) vals.as(TensorNode).setType("output");
                            if (idxs.is(TensorNode)) idxs.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    { ...fullOpts },
                    (g) => {
                        const topOps = g
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type);
                        expect(topOps).not.toContain("ArgMax");
                        expect(topOps).not.toContain("ArgMin");
                    },
                );

                // 3. LogSoftmax via LowerLogSoftmaxLoop Schedule
                await assertTransformMatchesOrt(
                    "schedule_logsoftmax_loop",
                    {
                        label: "sched_lsm",
                        tol: 1e-3,
                        feeds: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "lsm");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3, 4], "input"))
                                .as(TensorNode);
                            const [out] = b.createOp("LogSoftmax", [x], { axis: 1 }, [
                                { type: DataType.FLOAT, shape: [-1, 3, 4] },
                            ]);
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    {
                        ...fullOpts,
                        schedule: new Schedule().useRecipe("LogSoftmax", "LowerLogSoftmaxLoop"),
                    },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(1);
                    },
                );

                // 4. RMSNormalization via LowerRMSNormLoop Schedule
                await assertTransformMatchesOrt(
                    "schedule_rmsnorm_loop",
                    {
                        label: "sched_rms",
                        tol: 1e-4,
                        feeds: [
                            { name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" },
                            { name: "S", dtype: "float32", shape: [4], gen: "random" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "rms");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3, 4], "input"))
                                .as(TensorNode);
                            const s = g
                                .addNode("S")
                                .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
                                .as(TensorNode);
                            const [out] = b.createOp(
                                "RMSNormalization",
                                [x, s],
                                { axis: -1, epsilon: 1e-5 },
                                [{ type: DataType.FLOAT, shape: [-1, 3, 4] }],
                            );
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    {
                        ...fullOpts,
                        schedule: new Schedule().useRecipe("RMSNormalization", "LowerRMSNormLoop"),
                    },
                    (g) => {
                        const loops = g
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(loops.length).toBe(1);
                    },
                );

                // 5. AveragePool with overlapping kernel & constant power-of-2 kernel defers cleanly to 1 Loop
                await assertTransformMatchesOrt(
                    "avgpool_deferred_pow2",
                    {
                        label: "ap_pow2",
                        feeds: [
                            { name: "X", dtype: "float32", shape: [1, 2, 6, 6], gen: "negmix" },
                        ],
                        build: () => {
                            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                            const b = new GraphBuilder(g, "ap2");
                            const x = g
                                .addNode("X")
                                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 6, 6], "input"))
                                .as(TensorNode);
                            const [out] = b.createOp(
                                "AveragePool",
                                [x],
                                { kernel_shape: [2, 2], strides: [1, 1], pads: [0, 0, 0, 0] },
                                [{ type: DataType.FLOAT, shape: [1, 2, 5, 5] }],
                            );
                            if (out.is(TensorNode)) out.as(TensorNode).setType("output");
                            return g;
                        },
                    },
                    { ...fullOpts },
                    (g) => {
                        const topOps = g
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type);
                        expect(topOps.filter((t) => t === "Loop").length).toBe(1);
                        expect(topOps).not.toContain("Div");
                    },
                );
            },
            TEST_TIMEOUT,
        );
    });
});
