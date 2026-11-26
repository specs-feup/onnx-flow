import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/OnnxGraph.js";
import TensorNode from "../../../src/onnx/TensorNode.js";
import ConstantNode from "../../../src/onnx/ConstantNode.js";
import RegionArgumentNode from "../../../src/onnx/RegionArgumentNode.js";
import { GraphBuilder } from "../../../src/onnx/GraphBuilder.js";
import { DataType } from "../../../src/onnx/OnnxTypes.js";
import { scalarInt64 } from "../../../src/onnx/utils/TensorData.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { DeadCodeEliminationPass } from "../../../src/onnx/transformation/optimization/DeadCodeEliminationPass.js";

describe("Optimization Passes", () => {
    initializeSchemaRegistry();

    it("removes unconsumed intermediate nodes and constants while preserving graph inputs/outputs", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        graph.addNode("keep_in").init(new TensorNode.Builder(DataType.FLOAT, [2], "input"));
        graph.addNode("keep_out").init(new TensorNode.Builder(DataType.FLOAT, [2], "output"));

        graph.addNode("dead_const").init(new ConstantNode.Builder(scalarInt64(5)));
        graph
            .addNode("dead_tensor")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "intermediate"));

        const dce = new DeadCodeEliminationPass();
        const changed = dce.run(graph);

        expect(changed).toBe(true);
        expect(graph.hasNode("keep_in")).toBe(true);
        expect(graph.hasNode("keep_out")).toBe(true);
        expect(graph.hasNode("dead_const")).toBe(false);
        expect(graph.hasNode("dead_tensor")).toBe(false);
    });

    it("removes dead operation chains and returns false when graph is already clean", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "dce");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);

        // Live path: X -> Relu -> Y (output)
        const [liveOut] = builder.createOp("Relu", [x], {}, [{ type: DataType.FLOAT, shape: [2] }]);
        if (liveOut.is(TensorNode)) liveOut.as(TensorNode).setType("output");

        // Dead path: X -> Abs -> Neg (intermediate)
        const [dead1] = builder.createOp("Abs", [x], {}, [{ type: DataType.FLOAT, shape: [2] }]);
        const [dead2] = builder.createOp("Neg", [dead1], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);

        const dce = new DeadCodeEliminationPass();
        expect(dce.run(graph)).toBe(true);
        expect(graph.hasNode(liveOut.id)).toBe(true);
        expect(graph.hasNode(dead1.id)).toBe(false);
        expect(graph.hasNode(dead2.id)).toBe(false);

        // Second pass should be a no-op
        expect(dce.run(graph)).toBe(false);
    });

    it("preserves outer nodes captured by RegionArgumentNode inside Loop subgraphs", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "cap");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        // Intermediate node only referenced via RegionArgumentNode inside a Loop body
        const [captured] = builder.createOp("Relu", [x], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bb = new GraphBuilder(body, "b");
        const trip = body
            .addNode("iter")
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const cond = body
            .addNode("cond")
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const capArg = body
            .addNode("cap_arg")
            .init(new RegionArgumentNode.Builder(0, captured.id, DataType.FLOAT, [2]))
            .as(RegionArgumentNode);
        const [idOut] = bb.createOp("Identity", [capArg], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        const [condOut] = bb.createOp("Identity", [cond], {}, [{ type: DataType.BOOL, shape: [] }]);
        if (idOut.is(TensorNode)) idOut.as(TensorNode).setType("output");
        if (condOut.is(TensorNode)) condOut.as(TensorNode).setType("output");

        const loopTrip = builder.createConstant("trip", scalarInt64(1));
        const loopCond = builder.createConstant("lcond", {
            dataType: DataType.BOOL,
            dims: [],
            int32Data: [1],
        });
        const [loopOut] = builder.createOp(
            "Loop",
            [loopTrip, loopCond],
            {},
            [{ type: DataType.FLOAT, shape: [2] }],
            [body],
        );
        if (loopOut.is(TensorNode)) loopOut.as(TensorNode).setType("output");
        expect(trip.id).toBe("iter");

        new DeadCodeEliminationPass().run(graph);
        expect(graph.hasNode(captured.id)).toBe(true);
    });
});
