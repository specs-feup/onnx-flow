import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import type { GraphPass } from "../../../src/Onnx/PassManager.js";
import { PassManager } from "../../../src/Onnx/PassManager.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { topologicalSortOperationNodes } from "../../../src/Onnx/Utils.js";

describe("PassManager & Topological Sort", () => {
    it("stops PassManager when a pass returns false (fixed-point reached)", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const history = new HistoryManager(graph);
        const pm = new PassManager();

        let executions = 0;
        const dummyPass: GraphPass = {
            name: "DummyNoOp",
            run: () => {
                executions++;
                return false; // No mutation
            },
        };

        pm.addPass(dummyPass);
        pm.run(graph, history, 5);

        expect(executions).toBe(1);
    });

    it("sorts operations in valid topological order according to intermediate tensors", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        // in -> Op1 -> mid -> Op2 -> out
        const inT = graph
            .addNode("in")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const midT = graph
            .addNode("mid")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "intermediate"))
            .as(TensorNode);
        const outT = graph
            .addNode("out")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "output"))
            .as(TensorNode);

        const op2 = graph
            .addNode("op2")
            .init(new OperationNode.Builder("Relu", [midT]))
            .as(OperationNode);
        const op1 = graph
            .addNode("op1")
            .init(new OperationNode.Builder("Abs", [inT]))
            .as(OperationNode);

        graph.addEdge(inT, op1);
        graph.addEdge(op1, midT);
        graph.addEdge(midT, op2);
        graph.addEdge(op2, outT);

        const sorted = topologicalSortOperationNodes(graph);
        expect(sorted.map((o) => o.id)).toEqual(["op1", "op2"]);
    });
});
