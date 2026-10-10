import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { scalarInt64 } from "../../../src/Onnx/utils/TensorData.js";
import { DeadCodeEliminationPass } from "../../../src/Onnx/transformation/optimization/DeadCodeEliminationPass.js";

describe("Optimization Passes", () => {
    it("removes unconsumed intermediate nodes and constants while preserving graph inputs/outputs", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        // Keep: input and output
        graph.addNode("keep_in").init(new TensorNode.Builder(DataType.FLOAT, [2], "input"));
        graph.addNode("keep_out").init(new TensorNode.Builder(DataType.FLOAT, [2], "output"));

        // Dead: orphaned constant and intermediate tensor
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
});
