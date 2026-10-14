import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { LoopLoweringPass } from "../../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { scalarFloat } from "../../../src/Onnx/utils/TensorData.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";

describe("Loop Lowering - Multi-Operation Passes", () => {
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
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

        expect(
            graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Loop");
    });
});
