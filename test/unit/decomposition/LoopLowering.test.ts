import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { LoopLoweringPass } from "../../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import {
    scalarFloat,
    int64Vec,
    floatVec,
    scalarInt64,
} from "../../../src/Onnx/utils/TensorData.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import validateGraph from "../../../src/Onnx/validation/ValidateGraph.js";

describe("Loop Lowering - Multi-Operation Passes", () => {
    function assertLoweredToLoop(graph: OnnxGraph.Class, originalOpType: string): void {
        expect(() => validateGraph(graph)).not.toThrow();
        const ops = graph.getOperationNodes().toArray();
        const opTypes = ops.map((o) => o.type);
        expect(opTypes).toContain("Loop");
        expect(opTypes).not.toContain(originalOpType);

        const loopOp = ops.find((o) => o.type === "Loop")!;
        expect(loopOp.regions.length).toBe(1);
        expect(loopOp.regions[0].getOperationNodes().toArray().length).toBeGreaterThan(0);
    }

    initializeSchemaRegistry();

    it("lowers MatMul (Standard) into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);

        builder.createOp("MatMul", [a, b]);
        new LoopLoweringPass({
            coalesce: false,
            fuse: false,
            indexingMode: IndexingMode.Radix,
        }).run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "MatMul");
    });

    it("lowers MatMul with coalesced scalar MAC into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);

        builder.createOp("MatMul", [a, b]);
        new LoopLoweringPass({ coalesce: true, fuse: false, indexingMode: IndexingMode.Radix }).run(
            graph,
            new HistoryManager(graph),
        );

        assertLoweredToLoop(graph, "MatMul");
    });

    it("lowers ElementWise (Add) into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);

        builder.createOp("Add", [a, b]);
        new LoopLoweringPass({
            coalesce: false,
            fuse: false,
            indexingMode: IndexingMode.Radix,
        }).run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "Add");
    });

    it("lowers Conv into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 4, 4], "input"))
            .as(TensorNode);
        const w = graph
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
            .as(TensorNode);

        builder.createOp("Conv", [x, w], {
            kernel_shape: [3, 3],
            strides: [1, 1],
            pads: [0, 0, 0, 0],
        });
        new LoopLoweringPass({
            coalesce: false,
            fuse: false,
            indexingMode: IndexingMode.Radix,
        }).run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "Conv");
    });

    it("lowers Transpose into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);

        builder.createOp("Transpose", [x], { perm: [1, 0] });
        new LoopLoweringPass().run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "Transpose");
    });

    it("lowers Reductions (ReduceMean) into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);

        builder.createOp("ReduceMean", [x], { axes: [1] });
        new LoopLoweringPass().run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "ReduceMean");
    });

    it("lowers static Range into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const start = graph
            .addNode("s")
            .init(new ConstantNode.Builder(scalarFloat(0)))
            .as(ConstantNode);
        const limit = graph
            .addNode("l")
            .init(new ConstantNode.Builder(scalarFloat(4)))
            .as(ConstantNode);
        const delta = graph
            .addNode("d")
            .init(new ConstantNode.Builder(scalarFloat(1)))
            .as(ConstantNode);

        builder.createOp("Range", [start, limit, delta]);
        new LoopLoweringPass().run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "Range");
    });

    it("lowers Conv with auto_pad overriding stale pads into a Loop", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 4, 4], "input"))
            .as(TensorNode);
        const w = graph
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
            .as(TensorNode);

        builder.createOp("Conv", [x, w], {
            kernel_shape: [3, 3],
            strides: [1, 1],
            auto_pad: "SAME_UPPER",
            pads: [0, 0, 0, 0],
        });
        new LoopLoweringPass({
            coalesce: false,
            fuse: false,
            indexingMode: IndexingMode.Radix,
        }).run(graph, new HistoryManager(graph));

        assertLoweredToLoop(graph, "Conv");
        const loopOp = graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Loop")!;
        const bodyOps = loopOp.regions[0]
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(bodyOps).toContain("Pad");
    });

    it("lowers MaxPool with SAME_UPPER and ConvTranspose with SAME_LOWER into Loops", () => {
        const gMp = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bMp = new GraphBuilder(gMp);
        const xMp = gMp
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 4, 4], "input"))
            .as(TensorNode);
        bMp.createOpWithExact("mp", "MaxPool", [xMp], ["Y"], {
            kernel_shape: [3, 3],
            strides: [2, 2],
            auto_pad: "SAME_UPPER",
        });
        new LoopLoweringPass().run(gMp, new HistoryManager(gMp));
        assertLoweredToLoop(gMp, "MaxPool");

        const gCt = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bCt = new GraphBuilder(gCt);
        const xCt = gCt
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
            .as(TensorNode);
        const wCt = gCt
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
            .as(TensorNode);
        bCt.createOp("ConvTranspose", [xCt, wCt], {
            strides: [2, 2],
            auto_pad: "SAME_LOWER",
        });
        new LoopLoweringPass().run(gCt, new HistoryManager(gCt));
        assertLoweredToLoop(gCt, "ConvTranspose");
    });

    it("matches and lowers SliceLoop, PadLoop, TileLoop, OneHotLoop, DynamicReduction, ResizeCubic, and DynamicBatchedMatMul", () => {
        // Negative match check
        const gNeg = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bNeg = new GraphBuilder(gNeg);
        const xNeg = gNeg
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        bNeg.createOp("Relu", [xNeg]);
        const reluOp = gNeg.getOperationNodes().toArray()[0];
        expect(new Recipes.loopLowering.SliceLoop().match(reluOp)).toBe(false);
        expect(new Recipes.loopLowering.PadLoop().match(reluOp)).toBe(false);
        expect(new Recipes.loopLowering.TileLoop().match(reluOp)).toBe(false);
        expect(new Recipes.loopLowering.OneHotLoop().match(reluOp)).toBe(false);
        expect(new Recipes.loopLowering.DynamicReduction().match(reluOp)).toBe(false);
        expect(new Recipes.loopLowering.ResizeCubic().match(reluOp)).toBe(false);
        expect(new Recipes.loopLowering.DynamicBatchedMatMul().match(reluOp)).toBe(false);

        // Positive lowering via LoopLoweringPass
        const cases: (() => { g: OnnxGraph.Class; op: string })[] = [
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const x = g
                    .addNode("X")
                    .init(new TensorNode.Builder(DataType.FLOAT, [4, 5], "input"))
                    .as(TensorNode);
                const s = g
                    .addNode("s")
                    .init(new TensorNode.Builder(DataType.INT64, [1], "input"))
                    .as(TensorNode);
                const e = g
                    .addNode("e")
                    .init(new TensorNode.Builder(DataType.INT64, [1], "input"))
                    .as(TensorNode);
                b.createOp("Slice", [x, s, e], {}, [{ type: DataType.FLOAT, shape: [-1, 5] }]);
                return { g, op: "Slice" };
            },
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const x = g
                    .addNode("X")
                    .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                    .as(TensorNode);
                const p = g
                    .addNode("p")
                    .init(new TensorNode.Builder(DataType.INT64, [4], "input"))
                    .as(TensorNode);
                b.createOp("Pad", [x, p], { mode: "reflect" }, [
                    { type: DataType.FLOAT, shape: [-1, -1] },
                ]);
                return { g, op: "Pad" };
            },
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const x = g
                    .addNode("X")
                    .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                    .as(TensorNode);
                const r = g
                    .addNode("r")
                    .init(new TensorNode.Builder(DataType.INT64, [2], "input"))
                    .as(TensorNode);
                b.createOp("Tile", [x, r], {}, [{ type: DataType.FLOAT, shape: [-1, -1] }]);
                return { g, op: "Tile" };
            },
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const idx = g
                    .addNode("I")
                    .init(new TensorNode.Builder(DataType.INT64, [2, 2], "input"))
                    .as(TensorNode);
                const depth = g
                    .addNode("D")
                    .init(new TensorNode.Builder(DataType.INT64, [], "input"))
                    .as(TensorNode);
                const vals = g
                    .addNode("V")
                    .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
                    .as(TensorNode);
                b.createOp("OneHot", [idx, depth, vals], { axis: 1 }, [
                    { type: DataType.FLOAT, shape: [2, -1, 2] },
                ]);
                return { g, op: "OneHot" };
            },
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const x = g
                    .addNode("X")
                    .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3, 4], "input"))
                    .as(TensorNode);
                const ax = b.createConstant("ax", int64Vec([1]));
                b.createOp("ReduceSum", [x, ax], { keepdims: 1 }, [
                    { type: DataType.FLOAT, shape: [-1, 1, 4] },
                ]);
                return { g, op: "ReduceSum" };
            },
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const x = g
                    .addNode("X")
                    .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 3, 3], "input"))
                    .as(TensorNode);
                const roi = b.createConstant("roi", floatVec([]));
                const scales = b.createConstant("sc", floatVec([1, 1, 2, 2]));
                b.createOp("Resize", [x, roi, scales], { mode: "cubic" }, [
                    { type: DataType.FLOAT, shape: [1, 1, 6, 6] },
                ]);
                return { g, op: "Resize" };
            },
            () => {
                const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
                const b = new GraphBuilder(g);
                const a = g
                    .addNode("A")
                    .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3, 4], "input"))
                    .as(TensorNode);
                const bIn = g
                    .addNode("B")
                    .init(new TensorNode.Builder(DataType.FLOAT, [4, -1], "input"))
                    .as(TensorNode);
                b.createOp("MatMul", [a, bIn], {}, [{ type: DataType.FLOAT, shape: [-1, 3, -1] }]);
                return { g, op: "MatMul" };
            },
        ];

        for (const make of cases) {
            const { g, op } = make();
            new LoopLoweringPass({
                coalesce: true,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).run(g, new HistoryManager(g));
            assertLoweredToLoop(g, op);
        }
        // Keep scalarInt64 import exercised for future cases
        expect(scalarInt64(1).int64Data).toEqual([1]);
    });
});
