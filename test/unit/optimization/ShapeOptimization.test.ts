import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { ShapeOptimizationPass } from "../../../src/Onnx/transformation/optimization/ShapeOptimizationPass.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("ShapeOptimizationPass", () => {
    initializeSchemaRegistry();

    it("folds static Shape calls into Constant nodes", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const tensorWithStaticShape = graph
            .addNode("T")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 4, 5], "intermediate"))
            .as(TensorNode);

        builder.createOp("Shape", [tensorWithStaticShape]);

        const pass = new ShapeOptimizationPass();
        const changed = pass.run(graph);

        expect(changed).toBe(true);
        const opTypes = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(opTypes).not.toContain("Shape");
        expect(graph.getConstantNodes().length).toBeGreaterThan(0);
    });
});
