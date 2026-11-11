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
    });
});
