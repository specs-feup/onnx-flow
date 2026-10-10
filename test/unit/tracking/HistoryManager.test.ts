import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { TrackedGraphBuilder } from "../../../src/Onnx/transformation/tracking/TrackedGraphBuilder.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("HistoryManager & TrackedGraphBuilder", () => {
    initializeSchemaRegistry();

    it("records additions and undoes them cleanly to the initial graph state", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const initialInput = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);

        const history = new HistoryManager(graph);
        const builder = new TrackedGraphBuilder(graph, "opp_relu_1", "Apply Relu Transformation");

        // Perform operations through TrackedGraphBuilder
        const outputs = builder.createOp("Relu", [initialInput]);
        const patch = builder.commitPatch();
        history.pushPatch(patch);

        expect(graph.getNodes().length).toBe(3); // X, Relu op, Relu output
        expect(graph.hasNode(outputs[0].id)).toBe(true);

        // Undo
        const undone = history.undo();
        expect(undone?.opportunityId).toBe("opp_relu_1");
        expect(graph.hasNode(outputs[0].id)).toBe(false);
        expect(graph.getNodes().length).toBe(1); // Only original X remains

        // Redo
        const redone = history.redo();
        expect(redone?.opportunityId).toBe("opp_relu_1");
        expect(graph.hasNode(outputs[0].id)).toBe(true);
        expect(graph.getNodes().length).toBe(3);
    });

    it("tracks replaceAllUsesWith rewires and restores them on undo", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const inTensor = graph
            .addNode("in1")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);

        const history = new HistoryManager(graph);
        const builder = new TrackedGraphBuilder(
            graph,
            "opp_replace",
            "Replace input with new intermediate",
        );

        // Create an intermediate consumer op
        const [origInter] = builder.createOp("Identity", [inTensor]);
        const consumerOp = graph
            .addNode("consumer")
            .init(new OperationNode.Builder("Relu", [origInter]))
            .as(OperationNode);
        graph.addEdge(origInter, consumerOp);

        // Create alternative node and substitute it
        const [replacement] = builder.createOp("Relu", [inTensor]);
        builder.replaceAllUsesWith(origInter, replacement);

        const patch = builder.commitPatch();
        history.pushPatch(patch);

        // Consumer inputs should point to replacement
        expect(consumerOp.getInputs()?.[0].id).toBe(replacement.id);

        // Revert
        history.undo();
        expect(consumerOp.getInputs()?.[0].id).toBe(origInter.id);
    });
});
