import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { splitByAncestor } from "../../../src/Onnx/partitioning/Strategies.js";
import { partitionGraph } from "../../../src/Onnx/partitioning/Partition.js";
import { int64Vec } from "../../../src/Onnx/Utils.js";

describe("Graph Partitioning Strategies & Splitter", () => {
    function buildChainGraph() {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const [c1] = builder.createOp("Relu", [x]);
        const [c2] = builder.createOp("Abs", [c1]);
        graph
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "output"))
            .as(TensorNode);
        builder.createOpWithExact("final_id", "Identity", [c2], ["Y"]);

        return {
            graph,
            reluOp: graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Relu")!,
        };
    }

    it("splits node IDs into ancestor head and successor tail sets", () => {
        const { graph, reluOp } = buildChainGraph();
        const sets = splitByAncestor(graph, reluOp.id);

        expect(sets.head.size).toBeGreaterThan(0);
        expect(sets.tail.size).toBeGreaterThan(0);
        expect(sets.head.has(reluOp.id)).toBe(true);
        expect(sets.head.has("X")).toBe(true);
    });

    it("partitions an OnnxGraph into valid head and tail subgraphs", () => {
        const { graph, reluOp } = buildChainGraph();
        const sets = splitByAncestor(graph, reluOp.id);
        const { head, tail } = partitionGraph(graph, sets);

        expect(head).toBeDefined();
        expect(tail).toBeDefined();

        expect(head.hasNode("X")).toBe(true);
        expect(head.getInputTensorNodes().length).toBeGreaterThan(0);

        expect(tail.hasNode("Y")).toBe(true);
        expect(tail.getOutputTensorNodes().length).toBeGreaterThan(0);
    });

    it("handles constants and multi-input dependencies across partition boundary", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const c = builder.createConstant("bias", int64Vec([1, 1]));
        const [addOut] = builder.createOp("Add", [x, c]);
        const [subOut] = builder.createOp("Sub", [addOut, x]);

        const addOp = graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Add")!;
        const sets = splitByAncestor(graph, addOp.id);
        const { head, tail } = partitionGraph(graph, sets);

        expect(head.hasNode(c.id)).toBe(true);
        expect(tail.hasNode(subOut.id)).toBe(true);
        expect(tail.getNodes().length).toBeGreaterThan(0);
    });
});
