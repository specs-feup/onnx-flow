import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { CanonicalizationPass } from "../../../src/Onnx/transformation/canonicalization/CanonicalizationPass.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("CanonicalizationPass Batch Execution", () => {
    initializeSchemaRegistry();

    it("applies canonicalization passes across varied operations", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "canon");
        const history = new HistoryManager(graph);

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const y = graph
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const z = graph
            .addNode("Z")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);

        // Gemm -> MatMul + Add, Sub -> Add + Neg/Mul
        builder.createOp("Gemm", [x, y, z]);
        builder.createOp("Sub", [x, y]);

        const pass = new CanonicalizationPass();
        const mutated = pass.run(graph, history);

        expect(mutated).toBe(true);

        const remainingTypes = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(remainingTypes).not.toContain("Gemm");
        expect(remainingTypes).not.toContain("Sub");
        expect(remainingTypes).toContain("MatMul");
        expect(remainingTypes).toContain("Add");
        expect(history.getHistory().length).toBeGreaterThanOrEqual(2);
    });
});
