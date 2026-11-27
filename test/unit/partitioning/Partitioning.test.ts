import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import RegionArgumentNode from "../../../src/onnx/ir/RegionArgumentNode.js";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { splitByAncestor } from "../../../src/onnx/partitioning/Strategies.js";
import { partitionGraph } from "../../../src/onnx/partitioning/Partition.js";
import { PartitionedGraph } from "../../../src/onnx/partitioning/PartitionedGraph.js";
import { GraphPartitioner } from "../../../src/onnx/partitioning/GraphPartitioner.js";
import { AncestorSplitStrategy } from "../../../src/onnx/partitioning/PartitionStrategy.js";
import { int64Vec, makeTensorProto } from "../../../src/onnx/utils/index.js";

describe("Graph Partitioning Strategies, PartitionedGraph & GraphPartitioner", () => {
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
            absOp: graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Abs")!,
            idOp: graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Identity")!,
            c1,
            c2,
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

    it("partitions an OnnxGraph into a PartitionedGraph with valid head, tail, and connections", () => {
        const { graph, reluOp, c1 } = buildChainGraph();
        const sets = splitByAncestor(graph, reluOp.id);
        const partitioned = partitionGraph(graph, sets);

        expect(partitioned).toBeInstanceOf(PartitionedGraph);
        expect(partitioned.size).toBe(2);

        const { head, tail } = partitioned;
        expect(head.hasNode("X")).toBe(true);
        expect(head.getInputTensorNodes().length).toBeGreaterThan(0);

        expect(tail.hasNode("Y")).toBe(true);
        expect(tail.getOutputTensorNodes().length).toBeGreaterThan(0);

        expect(partitioned.connections).toEqual([
            { fromPartition: "head", toPartition: "tail", tensorId: c1.id },
        ]);
        expect(partitioned.getExecutionOrder().map((p) => p.id)).toEqual(["head", "tail"]);
        expect(partitioned.getBoundaryTensors("head", "tail")).toHaveLength(1);
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
        const partitioned = new GraphPartitioner().partition(
            graph,
            new AncestorSplitStrategy(addOp.id),
        );

        expect(partitioned.head.hasNode(c.id)).toBe(true);
        expect(partitioned.getPartition("head")!.inputs).toContain("X");
        expect(partitioned.tail.hasNode(subOut.id)).toBe(true);
        expect(partitioned.getPartition("tail")!.inputs).toContain("X");
        expect(partitioned.getPartition("tail")!.inputs).toContain(addOut.id);
        expect(partitioned.connections).toEqual([
            { fromPartition: "head", toPartition: "tail", tensorId: addOut.id },
        ]);
    });

    it("supports N-way (3-stage) partitioning with topological execution order", () => {
        const { graph, reluOp, absOp, idOp, c1, c2 } = buildChainGraph();
        const assignment = new Map<string, Set<string>>([
            ["stage_2", new Set([idOp.id, "Y"])],
            ["stage_0", new Set(["X", reluOp.id, c1.id])],
            ["stage_1", new Set([absOp.id, c2.id])],
        ]);

        const partitioned = new GraphPartitioner().partition(graph, assignment);

        expect(partitioned.size).toBe(3);
        expect(partitioned.getExecutionOrder().map((p) => p.id)).toEqual([
            "stage_0",
            "stage_1",
            "stage_2",
        ]);
        expect(partitioned.getBoundaryTensors("stage_0", "stage_1")).toEqual([
            { fromPartition: "stage_0", toPartition: "stage_1", tensorId: c1.id },
        ]);
        expect(partitioned.getBoundaryTensors("stage_1", "stage_2")).toEqual([
            { fromPartition: "stage_1", toPartition: "stage_2", tensorId: c2.id },
        ]);
        expect(partitioned.getGraph("stage_0")!.hasNode("X")).toBe(true);
        expect(partitioned.getGraph("stage_2")!.hasNode("Y")).toBe(true);

        initializeSchemaRegistry();
        expect(() => partitioned.inferAllShapes()).not.toThrow();
        expect(() => partitioned.validateAll()).not.toThrow();
    });

    it("detects cyclic partition dependencies in PartitionedGraph.getExecutionOrder", () => {
        const gA = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const gB = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const cyclic = new PartitionedGraph(
            [
                { id: "A", graph: gA, inputs: ["t_ba"], outputs: ["t_ab"] },
                { id: "B", graph: gB, inputs: ["t_ab"], outputs: ["t_ba"] },
            ],
            [
                { fromPartition: "A", toPartition: "B", tensorId: "t_ab" },
                { fromPartition: "B", toPartition: "A", tensorId: "t_ba" },
            ],
        );

        expect(() => cyclic.getExecutionOrder()).toThrow(/Cyclic dependency/i);
    });

    it("throws a clear error in AncestorSplitStrategy when splitNodeId is missing", () => {
        const { graph } = buildChainGraph();
        expect(() => new AncestorSplitStrategy("non_existent_node").split(graph)).toThrow(
            /non_existent_node/,
        );
    });

    it("preserves nested control-flow regions and RegionArgumentNodes when cloning/partitioning", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "loop_part");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const maxTrip = builder.createConstant("trip", makeTensorProto(DataType.INT64, [], [3]));
        const cond = builder.createConstant("cond", {
            dataType: DataType.BOOL,
            dims: [],
            int32Data: [1],
        });

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bodyBuilder = new GraphBuilder(body, "body");
        body.addNode("iter_in").init(new TensorNode.Builder(DataType.INT64, [], "input"));
        const condIn = body
            .addNode("cond_in")
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const vIn = body
            .addNode("v_in")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const extraArg = body
            .addNode("captured_arg")
            .init(new RegionArgumentNode.Builder(0, "captured", DataType.FLOAT, [2]))
            .as(RegionArgumentNode);
        const [vOut] = bodyBuilder.createOp("Add", [vIn, extraArg], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        const [condOut] = bodyBuilder.createOp("Identity", [condIn], {}, [
            { type: DataType.BOOL, shape: [] },
        ]);
        if (condOut.is(TensorNode)) condOut.as(TensorNode).setType("output");
        if (vOut.is(TensorNode)) vOut.as(TensorNode).setType("output");

        const [loopOut] = builder.createOp(
            "Loop",
            [maxTrip, cond, x],
            {},
            [{ type: DataType.FLOAT, shape: [2] }],
            [body],
        );
        const [finalOut] = builder.createOp("Abs", [loopOut], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        if (finalOut.is(TensorNode)) finalOut.as(TensorNode).setType("output");

        const loopOp = graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Loop")!;
        const partitioned = new GraphPartitioner().partition(
            graph,
            new AncestorSplitStrategy(loopOp.id),
        );

        const clonedLoop = partitioned.head
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Loop")!;
        expect(clonedLoop.regions).toHaveLength(1);
        const clonedRegion = clonedLoop.regions[0];
        expect(clonedRegion.getNodes().filterIs(RegionArgumentNode).length).toBe(1);
        expect(
            clonedRegion
                .getOperationNodes()
                .toArray()
                .some((o) => o.type === "Add"),
        ).toBe(true);
        expect(partitioned.tail.hasNode(finalOut.id)).toBe(true);
    });
});
