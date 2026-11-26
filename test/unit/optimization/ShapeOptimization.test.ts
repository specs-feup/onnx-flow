import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import OperationNode from "../../../src/onnx/ir/OperationNode.js";
import OnnxEdge from "../../../src/onnx/ir/OnnxEdge.js";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import { ShapeOptimizationPass } from "../../../src/onnx/transformation/optimization/ShapeOptimizationPass.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";

describe("ShapeOptimizationPass", () => {
    initializeSchemaRegistry();

    it("folds static Shape calls into Constant nodes and leaves dynamic shapes unchanged", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const staticTensor = graph
            .addNode("T_static")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 4, 5], "intermediate"))
            .as(TensorNode);
        builder.createOp("Shape", [staticTensor]);

        const dynamicTensor = graph
            .addNode("T_dynamic")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "intermediate"))
            .as(TensorNode);
        builder.createOp("Shape", [dynamicTensor]);

        const pass = new ShapeOptimizationPass();
        const changed = pass.run(graph);
        expect(changed).toBe(true);

        const remainingOps = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        // One Shape remains (dynamic), while the static one was folded into Constant
        expect(remainingOps.filter((t) => t === "Shape").length).toBe(1);
    });

    it("eliminates degenerate Concat operations with a single input", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        builder.createOp("Concat", [x], { axis: 0 });

        const pass = new ShapeOptimizationPass();
        const changed = pass.run(graph);
        expect(changed).toBe(true);

        const opTypes = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(opTypes).not.toContain("Concat");
        expect(opTypes).toContain("Identity");
    });

    it("optimizes 1x1 scalar math operations (Add, Sub, Mul, Div)", () => {
        for (const opType of ["Add", "Sub", "Mul", "Div"]) {
            const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder = new GraphBuilder(graph);

            const a = graph
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 1], "input"))
                .as(TensorNode);
            const b = graph
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 1], "input"))
                .as(TensorNode);
            builder.createOp(opType, [a, b]);

            const pass = new ShapeOptimizationPass();
            const changed = pass.run(graph);
            expect(changed).toBe(true);

            // Replaces root [1,1] op with Reshape / Gather / Squeeze pipeline
            const opTypes = graph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            expect(opTypes).toContain("Gather");
            expect(opTypes).toContain("Squeeze");
            expect(opTypes).toContain("Reshape");
        }
    });

    it("optimizes Outer Product MatMul into broadcasted Mul", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        // [M, 1] x [1, N]
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 1], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 5], "input"))
            .as(TensorNode);
        builder.createOp("MatMul", [a, b]);

        const pass = new ShapeOptimizationPass();
        const changed = pass.run(graph);
        expect(changed).toBe(true);

        const opTypes = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(opTypes).not.toContain("MatMul");
        expect(opTypes).toContain("Mul");
    });

    it("recurses into Loop subgraphs to fold static Shape nodes inside loop bodies", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bodyBuilder = new GraphBuilder(body, "body");
        const iterIn = body
            .addNode("iter")
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const condIn = body
            .addNode("cond_in")
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const staticInner = body
            .addNode("InnerT")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const [condOut] = bodyBuilder.createOp("Identity", [condIn], {}, [
            { type: DataType.BOOL, shape: [] },
        ]);
        if (condOut.is(TensorNode)) condOut.as(TensorNode).setType("output");
        const [shOut] = bodyBuilder.createOp("Shape", [staticInner], {}, [
            { type: DataType.INT64, shape: [2] },
        ]);
        if (shOut.is(TensorNode)) shOut.as(TensorNode).setType("output");

        const maxTrip = builder.createConstant("trip", {
            dataType: DataType.INT64,
            dims: [],
            int64Data: [2],
        });
        const cond = builder.createConstant("cond", {
            dataType: DataType.BOOL,
            dims: [],
            int32Data: [1],
        });
        const loopOp = graph
            .addNode("loop_test")
            .init(new OperationNode.Builder("Loop", [maxTrip, cond], {}, [body]))
            .as(OperationNode);
        const loopOut = graph
            .addNode("loop_out")
            .init(new TensorNode.Builder(DataType.INT64, [2, 2], "output"))
            .as(TensorNode);
        graph.addEdge(maxTrip, loopOp).init(new OnnxEdge.Builder(DataType.INT64, []));
        graph.addEdge(cond, loopOp).init(new OnnxEdge.Builder(DataType.BOOL, []));
        graph.addEdge(loopOp, loopOut).init(new OnnxEdge.Builder(DataType.INT64, [2, 2]));
        expect(iterIn.id).toBe("iter");

        const changed = new ShapeOptimizationPass().run(graph);
        expect(changed).toBe(true);
        const bodyOps = body
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(bodyOps).not.toContain("Shape");
    });
});
