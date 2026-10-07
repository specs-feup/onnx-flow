import { describe, it, expect, vi } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import OnnxEdge from "../../../src/Onnx/OnnxEdge.js";
import validateGraph from "../../../src/Onnx/validation/ValidateGraph.js";
import validateScope from "../../../src/Onnx/validation/ValidateScope.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { scalarInt64 } from "../../../src/Onnx/utils/TensorData.js";
import type { NodeCollection } from "@specs-feup/flow/graph/NodeCollection";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";

describe("Graph Validation & Scope Verification", () => {
    it("passes validation on a structurally valid graph", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const inTensor = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const outTensor = graph
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "output"))
            .as(TensorNode);
        const op = graph
            .addNode("op_relu")
            .init(new OperationNode.Builder("Relu", [inTensor]))
            .as(OperationNode);

        graph.addEdge(inTensor, op).init(new OnnxEdge.Builder(DataType.FLOAT, [2, 2]));
        graph.addEdge(op, outTensor).init(new OnnxEdge.Builder(DataType.FLOAT, [2, 2]));

        expect(() => validateGraph(graph)).not.toThrow();
    });

    it("detects dangling edges when target nodes are absent from graph lookup", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const inTensor = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const outTensor = graph
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "output"))
            .as(TensorNode);

        graph.addEdge(inTensor, outTensor).init(new OnnxEdge.Builder(DataType.FLOAT, [2]));

        // Simulate Y being missing from graph.getNodes() while inTensor's edge still points to it
        vi.spyOn(graph, "getNodes").mockReturnValue({
            toArray: () => [inTensor],
            [Symbol.iterator]: function* () {
                yield inTensor;
            },
        } as unknown as NodeCollection<
            BaseNode.Class<BaseNode.Data, BaseNode.ScratchData>,
            BaseNode.Data,
            BaseNode.ScratchData
        >);

        expect(() => validateGraph(graph, { checkDanglingEdges: true, checkScope: false })).toThrow(
            /\[Integrity Error\] Node 'X' has outgoing edge to missing node 'Y'/,
        );
    });

    it("rejects direct cross-subgraph edges and enforces boundary captures", () => {
        const rootGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const rootInput = rootGraph
            .addNode("outer_A")
            .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
            .as(TensorNode);
        const intermediate = rootGraph
            .addNode("inner_target")
            .init(new TensorNode.Builder(DataType.FLOAT, [4], "intermediate"))
            .as(TensorNode);

        rootGraph.addEdge(rootInput, intermediate).init(new OnnxEdge.Builder(DataType.FLOAT, [4]));

        // Simulate inner_target residing in a different scope (hasNode returns false)
        vi.spyOn(rootGraph, "hasNode").mockImplementation((id: string) => id !== "inner_target");

        expect(() => validateScope(rootGraph)).toThrow(
            /\[Scope Error\] Node 'outer_A' in graph 'root' links to 'inner_target' which is NOT in the same graph/,
        );
    });

    it("warns on undefined tensor types and orphan constants", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

        graph
            .addNode("untyped_tensor")
            .init(new TensorNode.Builder(DataType.UNDEFINED, [2], "intermediate"))
            .as(TensorNode);

        graph
            .addNode("orphan_const")
            .init(new ConstantNode.Builder(scalarInt64(10)))
            .as(ConstantNode);

        validateGraph(graph, {
            checkDanglingEdges: false,
            checkScope: false,
            checkTypeConsistency: true,
            checkOrphans: true,
        });

        expect(warnSpy).toHaveBeenCalledWith(
            expect.stringContaining(
                "[Validation Warning] Tensor 'untyped_tensor' has UNDEFINED data type",
            ),
        );
        expect(warnSpy).toHaveBeenCalledWith(
            expect.stringContaining(
                "[Validation Warning] Constant 'orphan_const' is unused (orphan)",
            ),
        );

        warnSpy.mockRestore();
    });
});
