import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import * as ort from "onnxruntime-web";
import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import ConstantNode from "../../../src/onnx/ir/ConstantNode.js";
import RegionArgumentNode from "../../../src/onnx/ir/RegionArgumentNode.js";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import {
    int64Vec,
    makeTensorProto,
    readTensorData,
    scalarFloat,
    scalarInt64,
} from "../../../src/onnx/utils/TensorData.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { DeadCodeEliminationPass } from "../../../src/onnx/transformation/optimization/DeadCodeEliminationPass.js";
import { ConstantFoldingPass } from "../../../src/onnx/transformation/optimization/ConstantFoldingPass.js";
import { AlgebraicSimplificationPass } from "../../../src/onnx/transformation/optimization/AlgebraicSimplificationPass.js";
import { CommonSubexpressionEliminationPass } from "../../../src/onnx/transformation/optimization/CommonSubexpressionEliminationPass.js";
import { ConvBnFoldingPass } from "../../../src/onnx/transformation/optimization/ConvBnFoldingPass.js";
import { buildFeeds, compareOrtOutputs, exportGraphToOnnx } from "../../helpers/OrtHarness.js";

describe("Optimization Passes", () => {
    initializeSchemaRegistry();

    it("removes unconsumed intermediate nodes and constants while preserving graph inputs/outputs", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        graph.addNode("keep_in").init(new TensorNode.Builder(DataType.FLOAT, [2], "input"));
        graph.addNode("keep_out").init(new TensorNode.Builder(DataType.FLOAT, [2], "output"));

        graph.addNode("dead_const").init(new ConstantNode.Builder(scalarInt64(5)));
        graph
            .addNode("dead_tensor")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "intermediate"));

        const dce = new DeadCodeEliminationPass();
        const changed = dce.run(graph);

        expect(changed).toBe(true);
        expect(graph.hasNode("keep_in")).toBe(true);
        expect(graph.hasNode("keep_out")).toBe(true);
        expect(graph.hasNode("dead_const")).toBe(false);
        expect(graph.hasNode("dead_tensor")).toBe(false);
    });

    it("removes dead operation chains and returns false when graph is already clean", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "dce");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);

        // Live path: X -> Relu -> Y (output)
        const [liveOut] = builder.createOp("Relu", [x], {}, [{ type: DataType.FLOAT, shape: [2] }]);
        if (liveOut.is(TensorNode)) liveOut.as(TensorNode).setType("output");

        // Dead path: X -> Abs -> Neg (intermediate)
        const [dead1] = builder.createOp("Abs", [x], {}, [{ type: DataType.FLOAT, shape: [2] }]);
        const [dead2] = builder.createOp("Neg", [dead1], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);

        const dce = new DeadCodeEliminationPass();
        expect(dce.run(graph)).toBe(true);
        expect(graph.hasNode(liveOut.id)).toBe(true);
        expect(graph.hasNode(dead1.id)).toBe(false);
        expect(graph.hasNode(dead2.id)).toBe(false);

        // Second pass should be a no-op
        expect(dce.run(graph)).toBe(false);
    });

    it("preserves outer nodes captured by RegionArgumentNode inside Loop subgraphs", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "cap");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        // Intermediate node only referenced via RegionArgumentNode inside a Loop body
        const [captured] = builder.createOp("Relu", [x], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);

        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bb = new GraphBuilder(body, "b");
        const trip = body
            .addNode("iter")
            .init(new TensorNode.Builder(DataType.INT64, [], "input"))
            .as(TensorNode);
        const cond = body
            .addNode("cond")
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);
        const capArg = body
            .addNode("cap_arg")
            .init(new RegionArgumentNode.Builder(0, captured.id, DataType.FLOAT, [2]))
            .as(RegionArgumentNode);
        const [idOut] = bb.createOp("Identity", [capArg], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        const [condOut] = bb.createOp("Identity", [cond], {}, [{ type: DataType.BOOL, shape: [] }]);
        if (idOut.is(TensorNode)) idOut.as(TensorNode).setType("output");
        if (condOut.is(TensorNode)) condOut.as(TensorNode).setType("output");

        const loopTrip = builder.createConstant("trip", scalarInt64(1));
        const loopCond = builder.createConstant("lcond", {
            dataType: DataType.BOOL,
            dims: [],
            int32Data: [1],
        });
        const [loopOut] = builder.createOp(
            "Loop",
            [loopTrip, loopCond],
            {},
            [{ type: DataType.FLOAT, shape: [2] }],
            [body],
        );
        if (loopOut.is(TensorNode)) loopOut.as(TensorNode).setType("output");
        expect(trip.id).toBe("iter");

        new DeadCodeEliminationPass().run(graph);
        expect(graph.hasNode(captured.id)).toBe(true);
    });

    it("ConstantFoldingPass folds Shape -> Gather -> Unsqueeze -> Concat -> ReduceProd chains", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "cf");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
            .as(TensorNode);
        const [sh] = builder.createOp("Shape", [x], {}, [{ type: DataType.INT64, shape: [3] }]);
        const idx1 = builder.createConstant("idx1", scalarInt64(1));
        const idx2 = builder.createConstant("idx2", scalarInt64(2));
        const ax0 = builder.createConstant("ax0", int64Vec([0]));
        const [d1] = builder.createOp("Gather", [sh, idx1], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);
        const [d2] = builder.createOp("Gather", [sh, idx2], { axis: 0 }, [
            { type: DataType.INT64, shape: [] },
        ]);
        const [prod] = builder.createOp("Mul", [d1, d2], {}, [{ type: DataType.INT64, shape: [] }]);
        const [prod1D] = builder.createOp("Unsqueeze", [prod, ax0], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);
        const two1D = builder.createConstant("two1d", int64Vec([2]));
        const [newShape] = builder.createOp("Concat", [two1D, prod1D], { axis: 0 }, [
            { type: DataType.INT64, shape: [2] },
        ]);
        const [y] = builder.createOp("Reshape", [x, newShape], {}, [
            { type: DataType.FLOAT, shape: [2, 12] },
        ]);
        if (y.is(TensorNode)) y.as(TensorNode).setType("output");

        const changed = new ConstantFoldingPass().run(graph);
        expect(changed).toBe(true);

        const remainingOps = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(remainingOps).toEqual(["Reshape"]);

        const reshapeOp = graph.getOperationNodes().first()!;
        const foldedShapeNode = reshapeOp.getInputs()![1];
        expect(foldedShapeNode.is(ConstantNode)).toBe(true);
        expect(readTensorData(foldedShapeNode)).toEqual([2, 12]);
    });

    it("AlgebraicSimplificationPass eliminates Add(x,0), Mul(x,1), no-op Cast, inverse Transpose, and chained Reshape", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "alg");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const zero = builder.createConstant("z", scalarFloat(0));
        const one = builder.createConstant("o", scalarFloat(1));
        const [a] = builder.createOp("Add", [x, zero], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        const [m] = builder.createOp("Mul", [a, one], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        const [c] = builder.createOp("Cast", [m], { to: DataType.FLOAT }, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        const [t1] = builder.createOp("Transpose", [c], { perm: [1, 0] }, [
            { type: DataType.FLOAT, shape: [3, 2] },
        ]);
        const [t2] = builder.createOp("Transpose", [t1], { perm: [1, 0] }, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        const s1 = builder.createConstant("s1", int64Vec([6]));
        const s2 = builder.createConstant("s2", int64Vec([3, 2]));
        const [r1] = builder.createOp("Reshape", [t2, s1], {}, [
            { type: DataType.FLOAT, shape: [6] },
        ]);
        const [r2] = builder.createOp("Reshape", [r1, s2], {}, [
            { type: DataType.FLOAT, shape: [3, 2] },
        ]);
        const [y] = builder.createOp("Relu", [r2], {}, [{ type: DataType.FLOAT, shape: [3, 2] }]);
        if (y.is(TensorNode)) y.as(TensorNode).setType("output");

        const changed = new AlgebraicSimplificationPass().run(graph);
        expect(changed).toBe(true);
        new DeadCodeEliminationPass().run(graph);

        const remainingOps = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(remainingOps.sort()).toEqual(["Relu", "Reshape"]);
    });

    it("CommonSubexpressionEliminationPass deduplicates identical pure ops (including commutative binary ops)", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "cse");

        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        // Commutative duplicates: Add(A, B) and Add(B, A)
        const [sum1] = builder.createOp("Add", [a, b], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        const [sum2] = builder.createOp("Add", [b, a], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        const [y] = builder.createOp("Mul", [sum1, sum2], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        if (y.is(TensorNode)) y.as(TensorNode).setType("output");

        const changed = new CommonSubexpressionEliminationPass().run(graph);
        expect(changed).toBe(true);

        const addOps = graph
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Add");
        expect(addOps).toHaveLength(1);
    });

    it("ConvBnFoldingPass folds BatchNormalization into Conv weights/bias and matches ORT numerically", async () => {
        const buildConvBn = (): OnnxGraph.Class => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "cbn");
            const x = g
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 4], "input"))
                .as(TensorNode);
            // Conv: C_out=3, C_in=2, K=3x3 -> 54 weights
            const wVals = Array.from({ length: 54 }, (_, i) => ((i % 7) - 3) * 0.15);
            const w = b.createConstant("W", makeTensorProto(DataType.FLOAT, [3, 2, 3, 3], wVals));
            const bias = b.createConstant(
                "B",
                makeTensorProto(DataType.FLOAT, [3], [0.1, -0.2, 0.3]),
            );
            const [convOut] = b.createOp("Conv", [x, w, bias], { pads: [1, 1, 1, 1] }, [
                { type: DataType.FLOAT, shape: [1, 3, 4, 4] },
            ]);

            const scale = b.createConstant(
                "sc",
                makeTensorProto(DataType.FLOAT, [3], [1.2, 0.8, 1.5]),
            );
            const bnB = b.createConstant(
                "bnb",
                makeTensorProto(DataType.FLOAT, [3], [-0.1, 0.25, 0.05]),
            );
            const mean = b.createConstant(
                "mn",
                makeTensorProto(DataType.FLOAT, [3], [0.05, -0.1, 0.2]),
            );
            const variance = b.createConstant(
                "vr",
                makeTensorProto(DataType.FLOAT, [3], [0.5, 1.2, 0.8]),
            );
            const [y] = b.createOp(
                "BatchNormalization",
                [convOut, scale, bnB, mean, variance],
                { epsilon: 1e-5 },
                [{ type: DataType.FLOAT, shape: [1, 3, 4, 4] }],
            );
            if (y.is(TensorNode)) y.as(TensorNode).setType("output");
            return g;
        };

        const origGraph = buildConvBn();
        const foldedGraph = buildConvBn();

        const changed = new ConvBnFoldingPass().run(foldedGraph);
        expect(changed).toBe(true);
        expect(
            foldedGraph
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toEqual(["Conv"]);

        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-convbn-"));
        try {
            const origPath = path.join(tmpDir, "orig.onnx");
            const foldedPath = path.join(tmpDir, "folded.onnx");
            await exportGraphToOnnx(origGraph, origPath);
            await exportGraphToOnnx(foldedGraph, foldedPath);

            const feeds = buildFeeds(
                [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
                "conv_bn_fold_seed",
            );
            const origOut = await (await ort.InferenceSession.create(origPath)).run(feeds);
            const foldedOut = await (await ort.InferenceSession.create(foldedPath)).run(feeds);

            const cmp = compareOrtOutputs(origOut, foldedOut, { tol: 1e-5 });
            expect(cmp.passed, cmp.message).toBe(true);
        } finally {
            if (fs.existsSync(tmpDir)) {
                fs.rmSync(tmpDir, { recursive: true, force: true });
            }
        }
    });
});
