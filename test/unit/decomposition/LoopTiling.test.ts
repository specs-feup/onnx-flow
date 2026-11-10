import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { LoopLoweringPass } from "../../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { LoopTilingPass } from "../../../src/Onnx/transformation/loop-lowering/LoopTilingPass.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import validateGraph from "../../../src/Onnx/validation/ValidateGraph.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("LoopTilingPass (Strip-Mining)", () => {
    initializeSchemaRegistry();

    it.each([IndexingMode.Pointer, IndexingMode.Radix])(
        "tiles a static Loop (exact & non-divisible tail) in %s mode and passes scope validation",
        (indexingMode) => {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);
            // 10 elements with tileSize 4 -> 3 outer tiles (4, 4, 2)
            const a = graph
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"))
                .as(TensorNode);
            const b = graph
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"))
                .as(TensorNode);
            builder.createOp("Add", [a, b]);

            const history = new HistoryManager(graph);
            new LoopLoweringPass({ coalesce: false, fuse: true, indexingMode }).run(graph, history);

            const tilingPass = new LoopTilingPass({ tileSize: 4 });
            const changed = tilingPass.run(graph, history);

            expect(changed).toBe(true);
            expect(() => validateGraph(graph)).not.toThrow();

            // Verify 2-level nested Loop structure
            const outerLoops = graph
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop");
            expect(outerLoops.length).toBe(1);

            const innerLoops = outerLoops[0].regions[0]
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop");
            expect(innerLoops.length).toBe(1);

            // Verify Undo restores the flat 1-level Loop
            history.undo();
            const restoredLoops = graph
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop");
            expect(restoredLoops.length).toBe(1);
            const nestedAfterUndo = restoredLoops[0].regions[0]
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop");
            expect(nestedAfterUndo.length).toBe(0);
        },
    );

    it("tiles a dynamic-shaped Loop cleanly", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
            .as(TensorNode);
        builder.createOp("Add", [a, b]);

        const history = new HistoryManager(graph);
        new LoopLoweringPass({
            coalesce: false,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).run(graph, history);

        const changed = new LoopTilingPass({ tileSize: 8 }).run(graph, history);
        expect(changed).toBe(true);
        expect(() => validateGraph(graph)).not.toThrow();
    });
});
