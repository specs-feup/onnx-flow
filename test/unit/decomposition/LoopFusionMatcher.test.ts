import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { LoopFusionMatcher } from "../../../src/Onnx/transformation/loop-lowering/LoopFusionMatcher.js";
import { scalarInt64 } from "../../../src/Onnx/Utils.js";

describe("LoopFusionMatcher", () => {
    initializeSchemaRegistry();

    function buildChain(shape: number[] = [2, 3]) {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "fuse");
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, shape, "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, shape, "input"))
            .as(TensorNode);
        const [addOut] = builder.createOp("Add", [a, b], {}, [{ type: DataType.FLOAT, shape }]);
        const [reluOut] = builder.createOp("Relu", [addOut], {}, [{ type: DataType.FLOAT, shape }]);
        if (reluOut.is(TensorNode)) reluOut.as(TensorNode).setType("output");
        return { graph, builder, a, b, addOut, reluOut };
    }

    it("fuses producer and elementwise consumer into a single opportunity when fuse=true", () => {
        const { graph } = buildChain();
        const matcher = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        });

        const opps = matcher.findOpportunities(graph);
        expect(opps.length).toBe(1);
        expect(opps[0].targetNodeId.split(",").length).toBe(2);
    });

    it("emits separate single-op opportunities when fuse=false", () => {
        const { graph } = buildChain();
        const matcher = new LoopFusionMatcher({
            coalesce: true,
            fuse: false,
            indexingMode: IndexingMode.Pointer,
        });

        const opps = matcher.findOpportunities(graph);
        expect(opps.length).toBe(2);
        expect(opps.every((o) => !o.targetNodeId.includes(","))).toBe(true);
    });

    it("stops fusion across multi-consumer forks and self-carry ops (CumSum)", () => {
        // 1. Fork: Add -> Relu and Add -> Neg (2 consumers of addOut -> cannot fuse)
        const { graph, builder, addOut } = buildChain();
        const [negOut] = builder.createOp("Neg", [addOut], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        if (negOut.is(TensorNode)) negOut.as(TensorNode).setType("output");

        const forkOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(graph);
        expect(forkOpps.length).toBe(3);

        // 2. Self-carry barrier: CumSum -> Relu cannot fuse
        const gCs = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bCs = new GraphBuilder(gCs, "cs");
        const x = gCs
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const ax = bCs.createConstant("ax", scalarInt64(1));
        const [csOut] = bCs.createOp("CumSum", [x, ax], { exclusive: 0, reverse: 0 }, [
            { type: DataType.FLOAT, shape: [2, 4] },
        ]);
        const [csRelu] = bCs.createOp("Relu", [csOut], {}, [
            { type: DataType.FLOAT, shape: [2, 4] },
        ]);
        if (csRelu.is(TensorNode)) csRelu.as(TensorNode).setType("output");

        const csOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(gCs);
        expect(csOpps.length).toBe(2);
    });

    it("selects CoalescedMatMul vs standard MatMul recipe based on coalesce flag", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "mm");
        const a = g
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const w = g
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
            .as(TensorNode);
        b.createOp("MatMul", [a, w], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);

        for (const coalesce of [true, false]) {
            const clone = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const cb = new GraphBuilder(clone, "mm");
            const ca = clone
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                .as(TensorNode);
            const cw = clone
                .addNode("W")
                .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
                .as(TensorNode);
            cb.createOp("MatMul", [ca, cw], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);

            const opps = new LoopFusionMatcher({
                coalesce,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).findOpportunities(clone);
            expect(opps.length).toBe(1);
            opps[0].apply(cb);
            const loopOp = clone
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Loop")!;
            const innerOps = loopOp.regions[0]
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            if (coalesce) {
                // Coalesced scalar MAC uses accumulator carries rather than vector ReduceSum
                expect(innerOps).not.toContain("ReduceSum");
            } else {
                expect(innerOps).toContain("ReduceSum");
            }
        }
        expect(g.getOperationNodes().length).toBe(1);
    });
});
