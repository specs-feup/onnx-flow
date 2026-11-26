import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/OnnxGraph.js";
import TensorNode from "../../../src/onnx/TensorNode.js";
import RegionArgumentNode from "../../../src/onnx/RegionArgumentNode.js";
import { GraphBuilder } from "../../../src/onnx/GraphBuilder.js";
import { DataType } from "../../../src/onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import inferShapes, { inferNodeShape, propagateToRegion } from "../../../src/onnx/InferShapes.js";
import { int64Vec } from "../../../src/onnx/utils/index.js";

describe("Shape Inference (InferShapes, inferNodeShape, propagateToRegion)", () => {
    initializeSchemaRegistry();

    it("infers elementwise broadcasting, MatMul, Conv, and Reshape shapes from Constants", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "shape");

        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 3], "input"))
            .as(TensorNode);
        const [added] = builder.createOp("Add", [a, b]);
        expect(added.shape).toEqual([2, 3]);

        const w = graph
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 5], "input"))
            .as(TensorNode);
        const [mm] = builder.createOp("MatMul", [added, w]);
        expect(mm.shape).toEqual([2, 5]);

        const targetShape = builder.createConstant("target_s", int64Vec([1, 10]));
        const [reshaped] = builder.createOp("Reshape", [mm, targetShape]);
        expect(reshaped.shape).toEqual([1, 10]);

        inferShapes(graph);
        expect(reshaped.shape).toEqual([1, 10]);
        expect(reshaped.literalType).toBe(DataType.FLOAT);
    });

    it("preserves dynamic -1 dimensions during inference", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "dyn");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3, 4], "input"))
            .as(TensorNode);
        const y = graph
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 3, 1], "input"))
            .as(TensorNode);
        const [out] = builder.createOp("Mul", [x, y]);
        const op = graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Mul")!;
        inferNodeShape(op, graph);

        expect(out.shape).toEqual([-1, 3, 4]);
    });

    it("propagates outer tensor shapes and dtypes into subgraph RegionArgumentNodes", () => {
        const outerGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const outerTensor = outerGraph
            .addNode("outer_weight")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 8], "input"))
            .as(TensorNode);

        const region = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const captured = region
            .addNode("cap_0")
            .init(new RegionArgumentNode.Builder(0, outerTensor.id, DataType.UNDEFINED, []))
            .as(RegionArgumentNode);

        propagateToRegion(outerGraph, region);

        expect(captured.shape).toEqual([4, 8]);
        expect(captured.literalType).toBe(DataType.FLOAT);
    });
});
