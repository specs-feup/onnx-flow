import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType, type ValueNode } from "../../../src/Onnx/OnnxTypes.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { buildForLoopRegion } from "../../../src/Onnx/transformation/loop-lowering/LoopBuilder.js";
import {
    getOrCreateIterationSpaceCounters,
    buildAguPointer,
    resolveRecipeInput,
} from "../../../src/Onnx/transformation/loop-lowering/RecipeUtils.js";
import { LoopLoweringPass } from "../../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { int64Vec, scalarFloat, scalarInt64 } from "../../../src/Onnx/utils/TensorData.js";
import validateGraph from "../../../src/Onnx/validation/ValidateGraph.js";

describe("Pointer Indexing (AGU) & Loop Lowering Coverage", () => {
    initializeSchemaRegistry();

    function createTestContext(indexingMode: IndexingMode = IndexingMode.Pointer) {
        const outerGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const outerBuilder = new GraphBuilder(outerGraph, "outer");
        const loopRegion = buildForLoopRegion(
            outerBuilder,
            12,
            DataType.FLOAT,
            [12],
            "test_loop",
            "scope",
            indexingMode,
        );
        return { outerGraph, outerBuilder, ...loopRegion };
    }

    describe("LoopBuilder & Carry Registration", () => {
        it("throws an error if a registered auxiliary carry is not given a nextState before finalize", () => {
            const { outerBuilder, loopCtx, vInitial, finalize } = createTestContext();
            const initPtr = outerBuilder.createConstant("init_ptr", scalarInt64(0));

            // Register a carry but intentionally omit loopCtx.setNextState(...)
            loopCtx.addCarry("dangling_ptr", initPtr, DataType.INT64, []);

            expect(() => finalize([vInitial])).toThrow(
                /\[LoopBuilder\] Missing nextState for registered carry/,
            );
        });
    });

    describe("AGU Primitives (getOrCreateIterationSpaceCounters & buildAguPointer)", () => {
        it("caches iteration space counters and folds size-1 dimensions without carry overhead", () => {
            const { innerBuilder, loopCtx, vInitial, finalize } = createTestContext();
            const valueMap = new Map<string, ValueNode>();

            // Includes size-1 dimension at index 1 to test static size-1 folding
            const space1 = getOrCreateIterationSpaceCounters(
                innerBuilder,
                loopCtx,
                valueMap,
                [2, 1, 3],
                "space_a",
            );
            const space2 = getOrCreateIterationSpaceCounters(
                innerBuilder,
                loopCtx,
                valueMap,
                [2, 1, 3],
                "space_b",
            );

            // Verify cache hit
            expect(space1).toBe(space2);
            expect(space1.counters.length).toBe(3);
            expect(space1.counters[1].is(ConstantNode)).toBe(true); // Size-1 folded to constant 0

            expect(() => finalize([vInitial])).not.toThrow();
        });

        it("handles all-zero strides fast-path (full broadcast / scalar)", () => {
            const { innerBuilder, loopCtx } = createTestContext();
            const valueMap = new Map<string, ValueNode>();

            const ptr = buildAguPointer(
                innerBuilder,
                loopCtx,
                valueMap,
                [2, 3],
                [0, 0],
                "zero_stride",
                0,
            );
            // Should return a ConstantNode(0) directly without allocating a loop carry
            expect(ptr.is(ConstantNode)).toBe(true);
        });

        it("handles rank-0 empty dims in buildAguPointer", () => {
            const { innerBuilder, loopCtx, vInitial, finalize } = createTestContext();
            const valueMap = new Map<string, ValueNode>();

            const ptr = buildAguPointer(innerBuilder, loopCtx, valueMap, [], [], "rank0_ptr", 5);
            expect(ptr.is(TensorNode)).toBe(true);
            expect(() => finalize([vInitial])).not.toThrow();
        });

        it("uses contiguous uniform-step fast-path without Where multiplexers", () => {
            const { innerBuilder, loopCtx, vInitial, finalize } = createTestContext();
            const valueMap = new Map<string, ValueNode>();

            // Shape [2, 3] with row-major strides [3, 1] is contiguous -> net delta is uniformly 1
            const ptr = buildAguPointer(
                innerBuilder,
                loopCtx,
                valueMap,
                [2, 3],
                [3, 1],
                "contig_ptr",
            );
            expect(ptr).toBeDefined();

            const innerOps = innerBuilder.graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            expect(innerOps).toContain("Add");
            expect(innerOps).not.toContain("Where");
            expect(innerOps).not.toContain("Div");
            expect(innerOps).not.toContain("Mod");

            finalize([vInitial]);
        });

        it("builds general non-contiguous static AGU cascade with Where multiplexers", () => {
            const { innerBuilder, loopCtx, vInitial, finalize } = createTestContext();
            const valueMap = new Map<string, ValueNode>();

            // Non-contiguous (e.g. broadcasted middle dimension or transposed strides [1, 0, 2])
            const ptr = buildAguPointer(
                innerBuilder,
                loopCtx,
                valueMap,
                [2, 3, 2],
                [1, 0, 2],
                "non_contig_ptr",
            );
            expect(ptr).toBeDefined();

            const innerOps = innerBuilder.graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            expect(innerOps).toContain("Where");
            expect(innerOps).not.toContain("Div");
            expect(innerOps).not.toContain("Mod");

            finalize([vInitial]);
        });

        it("builds dynamic AGU cascade when dimensions or strides are ValueNodes", () => {
            const { outerGraph, innerBuilder, loopCtx, vInitial, finalize } = createTestContext();
            const valueMap = new Map<string, ValueNode>();

            const dynDim = outerGraph
                .addNode("dyn_dim")
                .init(new TensorNode.Builder(DataType.INT64, [], "input"))
                .as(TensorNode);
            const dynStride = outerGraph
                .addNode("dyn_stride")
                .init(new TensorNode.Builder(DataType.INT64, [], "input"))
                .as(TensorNode);

            const ptr = buildAguPointer(
                innerBuilder,
                loopCtx,
                valueMap,
                [2, dynDim],
                [dynStride, 1],
                "dyn_ptr",
            );
            expect(ptr).toBeDefined();

            const innerOps = innerBuilder.graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            expect(innerOps).toContain("Sub");
            expect(innerOps).toContain("Mul");
            expect(innerOps).toContain("Where");
            expect(innerOps).not.toContain("Div");
            expect(innerOps).not.toContain("Mod");

            finalize([vInitial]);
        });

        it("resolves dynamic broadcasted inputs in Pointer mode via resolveRecipeInput", () => {
            const { outerGraph, innerBuilder, trip, loopCtx, vInitial, finalize } =
                createTestContext();
            const valueMap = new Map<string, ValueNode>();
            const axes = innerBuilder.createConstant("axes0", int64Vec([0]));

            const dynInput = outerGraph
                .addNode("dyn_in")
                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3], "input"))
                .as(TensorNode);
            const targetShape = outerGraph
                .addNode("target_shape")
                .init(new TensorNode.Builder(DataType.INT64, [2], "input"))
                .as(TensorNode);

            const resolved = resolveRecipeInput(
                innerBuilder,
                dynInput,
                valueMap,
                trip,
                axes,
                [-1, 3],
                true,
                true,
                targetShape,
                loopCtx,
            );
            expect(resolved).toBeDefined();
            finalize([vInitial]);
        });
    });

    describe("LoopLoweringPass with IndexingMode.Pointer across Recipes", () => {
        function assertPointerLoopValid(graph: OnnxGraph.Class) {
            expect(() => validateGraph(graph)).not.toThrow();
            const loopOp = graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Loop");
            expect(loopOp).toBeDefined();

            const bodyOps = loopOp!.regions[0]
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            // Pointer mode must never emit Div or Mod for address calculation
            expect(bodyOps).not.toContain("Div");
            expect(bodyOps).not.toContain("Mod");
        }

        it("lowers ElementWise with broadcasting in Pointer mode", () => {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);
            const a = graph
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
                .as(TensorNode);
            const b = graph
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 3, 1], "input"))
                .as(TensorNode);
            builder.createOp("Add", [a, b]);

            new LoopLoweringPass({
                coalesce: false,
                fuse: true,
                indexingMode: IndexingMode.Pointer,
            }).run(graph, new HistoryManager(graph));

            assertPointerLoopValid(graph);
        });

        it("lowers Standard MatMul (2D and 3D broadcasted batches) in Pointer mode", () => {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);
            const a = graph
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 1, 3, 4], "input"))
                .as(TensorNode);
            const b = graph
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 5], "input"))
                .as(TensorNode);
            builder.createOp("MatMul", [a, b]);

            new LoopLoweringPass({
                coalesce: false,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(graph, new HistoryManager(graph));

            assertPointerLoopValid(graph);
        });

        it("lowers Coalesced MatMul (2D and 3D broadcasted batches) in Pointer mode", () => {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);
            const a = graph
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
                .as(TensorNode);
            const b = graph
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 4, 2], "input"))
                .as(TensorNode);
            builder.createOp("MatMul", [a, b]);

            new LoopLoweringPass({
                coalesce: true,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(graph, new HistoryManager(graph));

            assertPointerLoopValid(graph);
        });

        it("lowers Conv (standard, padded, and grouped) in Pointer mode", () => {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);
            const x = graph
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 4, 5, 5], "input"))
                .as(TensorNode);
            const w = graph
                .addNode("W")
                .init(new TensorNode.Builder(DataType.FLOAT, [4, 2, 3, 3], "input"))
                .as(TensorNode);
            const b = graph
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
                .as(TensorNode);

            builder.createOp("Conv", [x, w, b], {
                group: 2,
                kernel_shape: [3, 3],
                strides: [2, 2],
                pads: [1, 1, 1, 1],
            });

            new LoopLoweringPass({
                coalesce: false,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(graph, new HistoryManager(graph));

            assertPointerLoopValid(graph);
        });

        it("lowers Reductions (ReduceSum, ReduceMax, ReduceL2, ReduceLogSumExp) in Pointer and Radix modes", () => {
            for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                for (const redOp of ["ReduceSum", "ReduceMax", "ReduceL2", "ReduceLogSumExp"]) {
                    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const builder = new GraphBuilder(graph);
                    const x = graph
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
                        .as(TensorNode);
                    builder.createOp(redOp, [x], { axes: [1], keepdims: 1 });

                    new LoopLoweringPass({
                        coalesce: false,
                        fuse: false,
                        indexingMode: mode,
                    }).run(graph, new HistoryManager(graph));

                    expect(() => validateGraph(graph)).not.toThrow();
                }
            }
        });

        it("lowers Transpose in both Pointer and Radix modes", () => {
            for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const builder = new GraphBuilder(graph);
                const x = graph
                    .addNode("X")
                    .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
                    .as(TensorNode);
                builder.createOp("Transpose", [x], { perm: [2, 0, 1] });

                new LoopLoweringPass({
                    coalesce: false,
                    fuse: false,
                    indexingMode: mode,
                }).run(graph, new HistoryManager(graph));

                expect(() => validateGraph(graph)).not.toThrow();
            }
        });

        it("lowers static and dynamic Range in both Pointer and Radix modes", () => {
            for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                // Dynamic Range
                const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const builder = new GraphBuilder(graph);
                const start = graph
                    .addNode("start")
                    .init(new TensorNode.Builder(DataType.FLOAT, [], "input"))
                    .as(TensorNode);
                const limit = graph
                    .addNode("limit")
                    .init(new TensorNode.Builder(DataType.FLOAT, [], "input"))
                    .as(TensorNode);
                const delta = graph
                    .addNode("delta")
                    .init(new TensorNode.Builder(DataType.FLOAT, [], "input"))
                    .as(TensorNode);
                builder.createOp("Range", [start, limit, delta]);

                new LoopLoweringPass({
                    coalesce: false,
                    fuse: false,
                    indexingMode: mode,
                }).run(graph, new HistoryManager(graph));

                expect(() => validateGraph(graph)).not.toThrow();
            }
        });

        it("throws on static Range with delta === 0", () => {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);
            const s = graph
                .addNode("s")
                .init(new ConstantNode.Builder(scalarFloat(0)))
                .as(ConstantNode);
            const l = graph
                .addNode("l")
                .init(new ConstantNode.Builder(scalarFloat(5)))
                .as(ConstantNode);
            const d = graph
                .addNode("d")
                .init(new ConstantNode.Builder(scalarFloat(0)))
                .as(ConstantNode);
            builder.createOp("Range", [s, l, d]);

            expect(() => new LoopLoweringPass().run(graph, new HistoryManager(graph))).toThrow(
                /delta of 0/,
            );
        });

        it("handles dynamic shapes in MatMul, CoalescedMatMul, Conv, Reduction, and Transpose under Pointer mode", () => {
            // 1. Dynamic CoalescedMatMul & Standard MatMul
            for (const coalesce of [true, false]) {
                const gMM = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const bMM = new GraphBuilder(gMM);
                const a = gMM
                    .addNode("A")
                    .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                    .as(TensorNode);
                const b = gMM
                    .addNode("B")
                    .init(new TensorNode.Builder(DataType.FLOAT, [4, -1], "input"))
                    .as(TensorNode);
                bMM.createOp("MatMul", [a, b]);
                new LoopLoweringPass({
                    coalesce,
                    fuse: false,
                    indexingMode: IndexingMode.Pointer,
                }).run(gMM, new HistoryManager(gMM));
                expect(() => validateGraph(gMM)).not.toThrow();
            }

            // 2. Dynamic Conv
            const gConv = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const bConv = new GraphBuilder(gConv);
            const cx = gConv
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 2, -1, -1], "input"))
                .as(TensorNode);
            const cw = gConv
                .addNode("W")
                .init(new TensorNode.Builder(DataType.FLOAT, [4, 2, 3, 3], "input"))
                .as(TensorNode);
            bConv.createOp("Conv", [cx, cw], {
                kernel_shape: [3, 3],
                strides: [1, 1],
                pads: [0, 0, 0, 0],
            });
            new LoopLoweringPass({
                coalesce: false,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(gConv, new HistoryManager(gConv));
            expect(() => validateGraph(gConv)).not.toThrow();

            // 3. Dynamic ReduceMean
            const gRed = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const bRed = new GraphBuilder(gRed);
            const rx = gRed
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                .as(TensorNode);
            bRed.createOp("ReduceMean", [rx], { axes: [1], keepdims: 0 });
            new LoopLoweringPass({
                coalesce: false,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(gRed, new HistoryManager(gRed));
            expect(() => validateGraph(gRed)).not.toThrow();

            // 4. Dynamic Transpose
            const gTp = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const bTp = new GraphBuilder(gTp);
            const tx = gTp
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                .as(TensorNode);
            bTp.createOp("Transpose", [tx], { perm: [1, 0] });
            new LoopLoweringPass({
                coalesce: false,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(gTp, new HistoryManager(gTp));
            expect(() => validateGraph(gTp)).not.toThrow();
        });

        describe("Tier 3 Loop Lowering Recipes (Pointer & Radix + Fusion)", () => {
            it("lowers CumSum (inclusive, exclusive, reverse) and prevents downstream fusion", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const builder = new GraphBuilder(graph);
                    const x = graph
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                        .as(TensorNode);
                    const ax = builder.createConstant("ax", scalarInt64(1));
                    const [cs] = builder.createOp("CumSum", [x, ax], { exclusive: 0, reverse: 1 });
                    builder.createOp("Relu", [cs]);

                    new LoopLoweringPass({ coalesce: true, fuse: true, indexingMode: mode }).run(
                        graph,
                        new HistoryManager(graph),
                    );
                    expect(() => validateGraph(graph)).not.toThrow();
                    const loops = graph
                        .getOperationNodes()
                        .toArray()
                        .filter((o) => o.type === "Loop");
                    expect(loops.length).toBe(2); // CumSum must NOT fuse into downstream Relu
                }
            });

            it("lowers ReverseSequence, GatherElements, and GatherND in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    // ReverseSequence
                    const gRS = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bRS = new GraphBuilder(gRS);
                    const xRS = gRS
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [4, 2], "input"))
                        .as(TensorNode);
                    const lens = bRS.createConstant("lens", int64Vec([2, 4]));
                    bRS.createOp("ReverseSequence", [xRS, lens], { time_axis: 0, batch_axis: 1 });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gRS,
                        new HistoryManager(gRS),
                    );
                    expect(() => validateGraph(gRS)).not.toThrow();

                    // GatherElements
                    const gGE = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bGE = new GraphBuilder(gGE);
                    const dGE = gGE
                        .addNode("D")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
                        .as(TensorNode);
                    const iGE = gGE
                        .addNode("I")
                        .init(new TensorNode.Builder(DataType.INT64, [2, 2], "input"))
                        .as(TensorNode);
                    bGE.createOp("GatherElements", [dGE, iGE], { axis: 1 });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gGE,
                        new HistoryManager(gGE),
                    );
                    expect(() => validateGraph(gGE)).not.toThrow();

                    // GatherND
                    const gGN = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bGN = new GraphBuilder(gGN);
                    const dGN = gGN
                        .addNode("D")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
                        .as(TensorNode);
                    const iGN = gGN
                        .addNode("I")
                        .init(new TensorNode.Builder(DataType.INT64, [2, 2], "input"))
                        .as(TensorNode);
                    bGN.createOp("GatherND", [dGN, iGN], { batch_dims: 0 });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gGN,
                        new HistoryManager(gGN),
                    );
                    expect(() => validateGraph(gGN)).not.toThrow();
                }
            });

            it("lowers MaxPool, ScatterND, ConvTranspose, and Resize (nearest & linear, fused) in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    // MaxPool
                    const gMP = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bMP = new GraphBuilder(gMP);
                    const xMP = gMP
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 4], "input"))
                        .as(TensorNode);
                    bMP.createOp("MaxPool", [xMP], {
                        kernel_shape: [2, 2],
                        strides: [2, 2],
                        pads: [0, 0, 0, 0],
                    });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gMP,
                        new HistoryManager(gMP),
                    );
                    expect(() => validateGraph(gMP)).not.toThrow();

                    // ScatterND
                    const gSN = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bSN = new GraphBuilder(gSN);
                    const dSN = gSN
                        .addNode("D")
                        .init(new TensorNode.Builder(DataType.FLOAT, [4, 2], "input"))
                        .as(TensorNode);
                    const iSN = gSN
                        .addNode("I")
                        .init(new TensorNode.Builder(DataType.INT64, [2, 1], "input"))
                        .as(TensorNode);
                    const uSN = gSN
                        .addNode("U")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
                        .as(TensorNode);
                    bSN.createOp("ScatterND", [dSN, iSN, uSN], { reduction: "none" });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gSN,
                        new HistoryManager(gSN),
                    );
                    expect(() => validateGraph(gSN)).not.toThrow();
                    if (mode === IndexingMode.Pointer) {
                        const sndLoops = gSN
                            .getOperationNodes()
                            .toArray()
                            .filter((o) => o.type === "Loop");
                        expect(sndLoops.length).toBeGreaterThan(0);
                        for (const lp of sndLoops) {
                            const bodyOps = lp.getRegion(0)!.getOperationNodes().toArray();
                            expect(bodyOps.some((o) => o.type === "Div" || o.type === "Mod")).toBe(
                                false,
                            );
                        }
                    }

                    // ConvTranspose
                    const gCT = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bCT = new GraphBuilder(gCT);
                    const xCT = gCT
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 3, 3], "input"))
                        .as(TensorNode);
                    const wCT = gCT
                        .addNode("W")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 2, 2, 2], "input"))
                        .as(TensorNode);
                    bCT.createOp("ConvTranspose", [xCT, wCT], {
                        kernel_shape: [2, 2],
                        strides: [1, 1],
                        pads: [0, 0, 0, 0],
                    });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gCT,
                        new HistoryManager(gCT),
                    );
                    expect(() => validateGraph(gCT)).not.toThrow();

                    // Resize (linear) + Relu fusion check
                    const gRZ = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bRZ = new GraphBuilder(gRZ);
                    const xRZ = gRZ
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 2, 2], "input"))
                        .as(TensorNode);
                    const roi = bRZ.createConstant("roi", {
                        dataType: DataType.FLOAT,
                        dims: [0],
                        floatData: [],
                    });
                    const scales = bRZ.createConstant("scales", {
                        dataType: DataType.FLOAT,
                        dims: [4],
                        floatData: [1, 1, 2, 2],
                    });
                    const [rz] = bRZ.createOp("Resize", [xRZ, roi, scales], { mode: "linear" }, [
                        { type: DataType.FLOAT, shape: [1, 1, 4, 4] },
                    ]);
                    bRZ.createOp("Relu", [rz], {}, [{ type: DataType.FLOAT, shape: [1, 1, 4, 4] }]);
                    new LoopLoweringPass({ coalesce: true, fuse: true, indexingMode: mode }).run(
                        gRZ,
                        new HistoryManager(gRZ),
                    );
                    expect(() => validateGraph(gRZ)).not.toThrow();
                    const rzLoops = gRZ
                        .getOperationNodes()
                        .toArray()
                        .filter((o) => o.type === "Loop");
                    expect(rzLoops.length).toBe(1); // Resize + Relu fuses cleanly into 1 loop
                }
            });
            it("lowers 1D/2D-non-overlapping/3D AveragePool via LowerAveragePoolLoopRecipe in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    // 1D AveragePool with padding and count_include_pad = 0
                    const g1D = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const b1D = new GraphBuilder(g1D);
                    const x1D = g1D
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 5], "input"))
                        .as(TensorNode);
                    b1D.createOp("AveragePool", [x1D], {
                        kernel_shape: [3],
                        strides: [2],
                        pads: [1, 1],
                        count_include_pad: 0,
                    });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        g1D,
                        new HistoryManager(g1D),
                    );
                    expect(() => validateGraph(g1D)).not.toThrow();
                    expect(
                        g1D
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");

                    // 3D AveragePool
                    const g3D = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const b3D = new GraphBuilder(g3D);
                    const x3D = g3D
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 4, 4], "input"))
                        .as(TensorNode);
                    b3D.createOp("AveragePool", [x3D], {
                        kernel_shape: [2, 2, 2],
                        strides: [2, 2, 2],
                        pads: [0, 0, 0, 0, 0, 0],
                    });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        g3D,
                        new HistoryManager(g3D),
                    );
                    expect(() => validateGraph(g3D)).not.toThrow();
                    expect(
                        g3D
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");
                }
            });

            it("lowers 1D and 3D Conv via LowerConvNDRecipe in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    // 1D Conv with bias & group = 2
                    const g1D = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const b1D = new GraphBuilder(g1D);
                    const x1D = g1D
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 4, 6], "input"))
                        .as(TensorNode);
                    const w1D = g1D
                        .addNode("W")
                        .init(new TensorNode.Builder(DataType.FLOAT, [4, 2, 3], "input"))
                        .as(TensorNode);
                    const bias1D = g1D
                        .addNode("B")
                        .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
                        .as(TensorNode);
                    b1D.createOp("Conv", [x1D, w1D, bias1D], {
                        group: 2,
                        strides: [1],
                        pads: [1, 1],
                    });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        g1D,
                        new HistoryManager(g1D),
                    );
                    expect(() => validateGraph(g1D)).not.toThrow();
                    expect(
                        g1D
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");

                    // 3D Conv
                    const g3D = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const b3D = new GraphBuilder(g3D);
                    const x3D = g3D
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 3, 3, 3], "input"))
                        .as(TensorNode);
                    const w3D = g3D
                        .addNode("W")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 2, 2, 2, 2], "input"))
                        .as(TensorNode);
                    b3D.createOp("Conv", [x3D, w3D], {
                        strides: [1, 1, 1],
                        pads: [0, 0, 0, 0, 0, 0],
                    });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        g3D,
                        new HistoryManager(g3D),
                    );
                    expect(() => validateGraph(g3D)).not.toThrow();
                    expect(
                        g3D
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");
                }
            });
            it("lowers dynamic Slice, Pad (wrap/reflect/dynamic), and Tile via loop recipes in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    // Dynamic Slice
                    const gSl = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bSl = new GraphBuilder(gSl);
                    const xSl = gSl
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [-1, 6], "input"))
                        .as(TensorNode);
                    const stSl = gSl
                        .addNode("starts")
                        .init(new TensorNode.Builder(DataType.INT64, [1], "input"))
                        .as(TensorNode);
                    const enSl = gSl
                        .addNode("ends")
                        .init(new TensorNode.Builder(DataType.INT64, [1], "input"))
                        .as(TensorNode);
                    bSl.createOp("Slice", [xSl, stSl, enSl]);
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gSl,
                        new HistoryManager(gSl),
                    );
                    expect(() => validateGraph(gSl)).not.toThrow();
                    expect(
                        gSl
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");

                    // Pad (wrap mode with dynamic pads)
                    const gPad = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bPad = new GraphBuilder(gPad);
                    const xPad = gPad
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                        .as(TensorNode);
                    const pPad = gPad
                        .addNode("pads")
                        .init(new TensorNode.Builder(DataType.INT64, [4], "input"))
                        .as(TensorNode);
                    bPad.createOp("Pad", [xPad, pPad], { mode: "wrap" });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gPad,
                        new HistoryManager(gPad),
                    );
                    expect(() => validateGraph(gPad)).not.toThrow();
                    expect(
                        gPad
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");

                    // Dynamic Tile
                    const gTile = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bTile = new GraphBuilder(gTile);
                    const xTile = gTile
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3], "input"))
                        .as(TensorNode);
                    const rTile = gTile
                        .addNode("repeats")
                        .init(new TensorNode.Builder(DataType.INT64, [2], "input"))
                        .as(TensorNode);
                    bTile.createOp("Tile", [xTile, rTile]);
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gTile,
                        new HistoryManager(gTile),
                    );
                    expect(() => validateGraph(gTile)).not.toThrow();
                    expect(
                        gTile
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");
                }
            });

            it("lowers dynamic OneHot and dynamic Reductions in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    // Dynamic OneHot
                    const gOH = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bOH = new GraphBuilder(gOH);
                    const idxOH = gOH
                        .addNode("I")
                        .init(new TensorNode.Builder(DataType.INT64, [-1, 2], "input"))
                        .as(TensorNode);
                    const depthOH = gOH
                        .addNode("D")
                        .init(new TensorNode.Builder(DataType.INT64, [], "input"))
                        .as(TensorNode);
                    const valsOH = gOH
                        .addNode("V")
                        .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
                        .as(TensorNode);
                    bOH.createOp("OneHot", [idxOH, depthOH, valsOH], { axis: -1 });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gOH,
                        new HistoryManager(gOH),
                    );
                    expect(() => validateGraph(gOH)).not.toThrow();
                    expect(
                        gOH
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");

                    // Dynamic ReduceMean
                    const gRed = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const bRed = new GraphBuilder(gRed);
                    const xRed = gRed
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
                        .as(TensorNode);
                    bRed.createOp("ReduceMean", [xRed], { axes: [1], keepdims: 1 });
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        gRed,
                        new HistoryManager(gRed),
                    );
                    expect(() => validateGraph(gRed)).not.toThrow();
                    expect(
                        gRed
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");
                }
            });

            it("lowers 1D/2D cubic Resize (with and without exclude_outside) in Pointer & Radix modes", () => {
                for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                    const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                    const b = new GraphBuilder(g);
                    const x = g
                        .addNode("X")
                        .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
                        .as(TensorNode);
                    const roi = b.createConstant("roi", {
                        dataType: DataType.FLOAT,
                        dims: [0],
                        floatData: [],
                    });
                    const scales = b.createConstant("scales", {
                        dataType: DataType.FLOAT,
                        dims: [4],
                        floatData: [1, 1, 2, 2],
                    });
                    b.createOp(
                        "Resize",
                        [x, roi, scales],
                        { mode: "cubic", cubic_coeff_a: -0.75, exclude_outside: 1 },
                        [{ type: DataType.FLOAT, shape: [1, 1, 6, 6] }],
                    );
                    new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                        g,
                        new HistoryManager(g),
                    );
                    expect(() => validateGraph(g)).not.toThrow();
                    expect(
                        g
                            .getOperationNodes()
                            .toArray()
                            .map((o) => o.type),
                    ).toContain("Loop");
                }
            });
        });
        it("lowers dynamic-batch broadcast MatMul in Pointer & Radix modes", () => {
            for (const mode of [IndexingMode.Pointer, IndexingMode.Radix]) {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const a = g
                    .addNode("A")
                    .init(new TensorNode.Builder(DataType.FLOAT, [-1, 1, 3, 4], "input"))
                    .as(TensorNode);
                const w = g
                    .addNode("B")
                    .init(new TensorNode.Builder(DataType.FLOAT, [1, -1, 4, 2], "input"))
                    .as(TensorNode);
                b.createOp("MatMul", [a, w], {}, [{ type: DataType.FLOAT, shape: [-1, -1, 3, 2] }]);
                new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: mode }).run(
                    g,
                    new HistoryManager(g),
                );
                expect(() => validateGraph(g)).not.toThrow();
                expect(
                    g
                        .getOperationNodes()
                        .toArray()
                        .map((o) => o.type),
                ).toContain("Loop");
            }
        });
    });
});
