import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import OnnxEdge from "../../../src/Onnx/OnnxEdge.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import OnnxDotFormatter from "../../../src/Onnx/dot/OnnxDotFormatter.js";
import CgraDotFormatter from "../../../src/Onnx/dot/CgraDotFormatter.js";
import { scalarFloat, floatVec } from "../../../src/Onnx/utils/TensorData.js";

describe("DOT Formatters (Onnx & CGRA)", () => {
    function createComplexGraph(): OnnxGraph.Class {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const inNode = graph
            .addNode("input_0")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 4], "input"))
            .as(TensorNode);
        const constNode = builder.createConstant("const_val", scalarFloat(1.5));
        const [addOut] = builder.createOp("Add", [inNode, constNode], { alpha: 1.0 });

        const innerGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const innerIn = innerGraph
            .addNode("inner_x")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 4], "input"))
            .as(TensorNode);
        const innerOut = innerGraph
            .addNode("inner_y")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 4], "output"))
            .as(TensorNode);
        const innerOp = innerGraph
            .addNode("inner_id")
            .init(new OperationNode.Builder("Identity", [innerIn]))
            .as(OperationNode);
        innerGraph.addEdge(innerIn, innerOp).init(new OnnxEdge.Builder(DataType.FLOAT, [1, 4]));
        innerGraph.addEdge(innerOp, innerOut).init(new OnnxEdge.Builder(DataType.FLOAT, [1, 4]));

        graph
            .addNode("loop_op")
            .init(new OperationNode.Builder("Loop", [addOut], {}, [innerGraph]))
            .as(OperationNode);

        return graph;
    }

    function create1DGraph(): OnnxGraph.Class {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const inNode = graph
            .addNode("input_0")
            .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
            .as(TensorNode);
        const constNode = builder.createConstant("const_val", floatVec([1.0, 2.0, 3.0, 4.0]));
        builder.createOp("Add", [inNode, constNode]);

        return graph;
    }

    it("formats graph to DOT using OnnxDotFormatter", () => {
        const graph = createComplexGraph();
        const formatter = new OnnxDotFormatter();
        const dot = graph.toString(formatter);

        expect(typeof dot).toBe("string");
        expect(dot).toContain("digraph");
        expect(dot).toContain("input_0");
        expect(dot).toContain("const_val");
        expect(dot).toContain("Add");
        expect(dot).toContain("subgraph cluster_");
    });

    it("formats graph to DOT using CgraDotFormatter", () => {
        const graph = create1DGraph();
        const formatter = new CgraDotFormatter();
        const dot = graph.toString(formatter);

        expect(typeof dot).toBe("string");
        expect(dot).toContain("digraph");
        expect(dot).toContain("input_0");
        expect(dot).toContain("const_val");
        expect(dot).toContain("Add");
    });
});
