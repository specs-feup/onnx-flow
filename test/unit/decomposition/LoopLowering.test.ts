import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { LoopLoweringPass } from "../../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("Loop Lowering Pipeline", () => {
    initializeSchemaRegistry();

    it("lowers an element-wise operation into an ONNX Loop subgraph", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "loop_test");
        const history = new HistoryManager(graph);

        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);

        builder.createOp("Add", [a, b]);

        const pass = new LoopLoweringPass({ coalesce: true, fuse: true });
        const changed = pass.run(graph, history);

        expect(changed).toBe(true);

        const opTypes = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(opTypes).toContain("Loop");
        expect(history.getHistory().length).toBeGreaterThan(0);
    });
});
