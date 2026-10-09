import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import {
    prepareGraphForExport,
    convertFlowGraphToOnnxJson,
    generateUnifiedExplorerJson,
} from "../../../src/flow2json.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("flow2json & Explorer Serialization", () => {
    initializeSchemaRegistry();

    it("prepares graph by restoring missing edges between inputs and operations", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const inTensor = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const _op = graph
            .addNode("op_id")
            .init(new OperationNode.Builder("Identity", [inTensor]))
            .as(OperationNode);

        expect(inTensor.getOutgoers.length).toBe(0);
        prepareGraphForExport(graph);
        expect(inTensor.getOutgoers.length).toBe(1);
    });

    it("converts complex flow graph containing raw buffers, constants, and subgraphs to RawOnnxModel", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const inX = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 2], "input"))
            .as(TensorNode);
        const _outY = graph
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 2], "output"))
            .as(TensorNode);

        // Constant with rawData Buffer (INT64)
        const buf = Buffer.alloc(8);
        buf.writeBigInt64LE(100n, 0);
        const constNode = graph
            .addNode("C_raw")
            .init(
                new ConstantNode.Builder(
                    {
                        dataType: DataType.INT64,
                        dims: [1],
                        rawData: { type: "Buffer", data: Array.from(buf) },
                    },
                    true, // isInput = true
                ),
            )
            .as(ConstantNode);

        // Loop Subgraph
        const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        loopBody.addNode("iter").init(new TensorNode.Builder(DataType.INT64, [], "input"));
        loopBody.addNode("cond").init(new TensorNode.Builder(DataType.BOOL, [], "input"));
        loopBody.addNode("carry").init(new TensorNode.Builder(DataType.FLOAT, [1, 2], "input"));
        loopBody.addNode("cond_out").init(new TensorNode.Builder(DataType.BOOL, [], "output"));
        loopBody
            .addNode("carry_out")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 2], "output"));

        builder.createOp("Loop", [constNode, constNode, inX], {}, undefined, [loopBody]);
        builder.createOpWithExact("final_id", "Identity", [inX], ["Y"]);

        const rawModel = convertFlowGraphToOnnxJson(graph, "test_export");
        expect(rawModel.graph?.initializer?.length).toBeGreaterThan(0);
        expect(rawModel.graph?.input?.length).toBeGreaterThan(0);
        expect(rawModel.graph?.output?.length).toBe(1);

        const loopJsonNode = rawModel.graph?.node?.find((n) => n.opType === "Loop");
        expect(loopJsonNode).toBeDefined();
        const bodyAttr = loopJsonNode?.attribute?.find((a) => a.name === "body");
        expect(bodyAttr?.g).toBeDefined();
    });

    it("generates unified explorer JSON with cross-graph capture edges", () => {
        const rootGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(rootGraph);

        const rootTensor = rootGraph
            .addNode("outer_var")
            .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
            .as(TensorNode);

        const subGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const subBuilder = new GraphBuilder(subGraph);
        // Op referencing outer_var
        subBuilder.createOp("Relu", [rootTensor]);

        builder.createOp("Loop", [rootTensor], {}, undefined, [subGraph]);

        const unified = generateUnifiedExplorerJson(rootGraph, true);
        expect(unified.elements.nodes.length).toBeGreaterThan(0);

        const loopNode = unified.elements.nodes.find(
            (n) => n.data.onnxData?.kind === "OperationNode",
        );
        expect(loopNode).toBeDefined();
    });
});
