import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it, expect, vi } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { HistoryManager } from "../../../src/onnx/transformation/tracking/HistoryManager.js";
import { CanonicalizationPass } from "../../../src/onnx/transformation/canonicalization/CanonicalizationPass.js";
import { InferShapesPass } from "../../../src/onnx/transformation/InferShapesPass.js";
import { OrchestratorPass } from "../../../src/onnx/transformation/OrchestratorPass.js";
import { TransformationRegistry } from "../../../src/onnx/transformation/TransformationRegistry.js";
import OnnxGraphTransformer from "../../../src/onnx/transformation/index.js";
import { Recipes } from "../../../src/onnx/transformation/RecipeCatalog.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import type { DecompositionRecipe } from "../../../src/onnx/transformation/Recipe.js";
import validateGraph from "../../../src/onnx/validation/ValidateGraph.js";
import { makeTensorProto } from "../../../src/onnx/utils/index.js";
import * as ort from "onnxruntime-web";
import { buildFeeds, compareOrtOutputs, exportGraphToOnnx } from "../../helpers/OrtHarness.js";

describe("Transformation Pipeline & Passes", () => {
    initializeSchemaRegistry();

    function createSimpleReluGraph(): OnnxGraph.Class {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        builder.createOp("Relu", [x]);
        return graph;
    }

    it("runs CanonicalizationPass and records mutations in HistoryManager", () => {
        const graph = createSimpleReluGraph();
        const history = new HistoryManager(graph);
        const pass = new CanonicalizationPass();

        const changed = pass.run(graph, history);
        expect(changed).toBe(true);
        expect(history.getHistory().length).toBeGreaterThan(0);

        // Fixed point check
        const secondRunChanged = pass.run(graph, history);
        expect(secondRunChanged).toBe(false);
    });

    it("discovers available opportunities via TransformationRegistry", () => {
        const graph = createSimpleReluGraph();
        const registry = new TransformationRegistry([new Recipes.canonicalization.Relu()]);

        const opps = registry.findAllOpportunities(graph);
        expect(opps.length).toBe(1);
        expect(opps[0].recipeName).toBe("LowerRelu");

        const builder = new GraphBuilder(graph);
        expect(() => opps[0].apply(builder)).not.toThrow();
    });

    it("applies transforms and converges to fixed-point via OrchestratorPass", () => {
        const graph = createSimpleReluGraph();
        const pass = new OrchestratorPass("OrchestratorTest", [
            new Recipes.canonicalization.Relu(),
        ]);

        const changed = pass.run(graph);
        expect(changed).toBe(true);

        const opTypes = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(opTypes).not.toContain("Relu");
    });

    it("executes InferShapesPass without mutating graph topology", () => {
        const graph = createSimpleReluGraph();
        const pass = new InferShapesPass();

        expect(pass.name).toBe("InferShapes");
        const changed = pass.run(graph);
        expect(changed).toBe(false);
    });

    it("executes full transformation pipeline through OnnxGraphTransformer", () => {
        const graph = createSimpleReluGraph();
        const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

        const transformer = new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: false,
            decomposeForCgra: false,
        });

        const transformedGraph = transformer.apply(graph, true);
        expect(transformedGraph).toBe(graph);

        logSpy.mockRestore();
    });

    it("falls back to secondary complementary recipes in CanonicalizationPass and OrchestratorPass", () => {
        const neverMatchRelu: DecompositionRecipe = {
            name: "NeverMatchRelu",
            targetOp: "Relu",
            exposesControlFlow: false,
            exposesDataAccess: false,
            phase: "canonicalization" as const,
            precision: "exact" as const,
            priority: 50,
            shapeSupport: "dynamic" as const,
            producedOps: [],
            match: () => false,
            apply: () => {
                throw new Error("Should not be called");
            },
        };
        const fallbackRelu = new Recipes.canonicalization.Relu();

        // 1. CanonicalizationPass fallback
        const g1 = createSimpleReluGraph();
        const h1 = new HistoryManager(g1);
        const canonPass = new CanonicalizationPass([neverMatchRelu, fallbackRelu]);
        expect(canonPass.run(g1, h1)).toBe(true);
        expect(
            g1
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).not.toContain("Relu");
        expect(h1.getHistory()[0].opportunityId).toContain("LowerRelu");

        // 2. OrchestratorPass fallback
        const g2 = createSimpleReluGraph();
        const orchPass = new OrchestratorPass("FallbackOrch", [neverMatchRelu, fallbackRelu]);
        expect(orchPass.run(g2)).toBe(true);
        expect(
            g2
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).not.toContain("Relu");
    });

    it("preserves outer tensors captured by Loop RegionArgumentNodes during DeadCodeEliminationPass", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "dce_capture");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const [reluOut] = b.createOp("Relu", [x], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
        const [negOut] = b.createOp("Neg", [reluOut], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        if (negOut.is(TensorNode)) negOut.as(TensorNode).setType("output");

        const out = new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            coalesce: true,
            recurse: true,
            optimize: true,
            decomposeForCgra: false,
            indexingMode: IndexingMode.Pointer,
        }).apply(g, false);

        const outerOps = out
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(outerOps).toContain("Loop");
        expect(outerOps).toContain("Reshape");
    });

    it("lowers 0D scalar Exp/Div/Sqrt inside Loop bodies when recurse=true and runs optimize by default", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "rec_sm");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const [out] = b.createOp("Det", [x], {}, [{ type: DataType.FLOAT, shape: [] }]);
        if (out.is(TensorNode)) out.as(TensorNode).setType("output");

        // Verify optimize=true runs in default OnnxGraphTransformer() without leaving orphaned ops
        const defOut = new OnnxGraphTransformer().apply(g, false);
        expect(
            defOut
                .getOperationNodes()
                .toArray()
                .some((o) => o.type === "Det"),
        ).toBe(false);

        // Build 4x4 Det with recurse=true: inner scalar Div inside Loop must be canonicalized away
        const gRec = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bRec = new GraphBuilder(gRec, "rec_det");
        const xRec = gRec
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 4], "input"))
            .as(TensorNode);
        const [detOut] = bRec.createOp("Det", [xRec], {}, [{ type: DataType.FLOAT, shape: [] }]);
        if (detOut.is(TensorNode)) detOut.as(TensorNode).setType("output");

        const lowered = new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            recurse: true,
            optimize: true,
        }).apply(gRec, false);

        const collectTypes = (gr: OnnxGraph.Class, acc: string[] = []): string[] => {
            for (const op of gr.getOperationNodes()) {
                acc.push(op.type);
                for (const reg of op.regions) collectTypes(reg, acc);
            }
            return acc;
        };
        const allTypes = collectTypes(lowered);
        expect(allTypes).not.toContain("Div");
        expect(allTypes).not.toContain("Reciprocal");
    });

    it("recursively lowers tensor ops inside outer Loop regions into nested explicit Loops and matches ORT", async () => {
        const buildRnnGraph = (): OnnxGraph.Class => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "rnn_rec");
            // Dynamic seq_len forces LowerRNNLoopRecipe (outer time-step Loop containing MatMul/Add/Tanh)
            const x = g
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [-1, 1, 3], "input"))
                .as(TensorNode);
            const w = b.createConstant(
                "W",
                makeTensorProto(DataType.FLOAT, [1, 2, 3], [0.2, -0.1, 0.3, -0.2, 0.4, 0.1]),
            );
            const r = b.createConstant(
                "R",
                makeTensorProto(DataType.FLOAT, [1, 2, 2], [0.1, -0.2, 0.3, 0.15]),
            );
            const [y, yh] = b.createOp("RNN", [x, w, r], { hidden_size: 2 }, [
                { type: DataType.FLOAT, shape: [-1, 1, 1, 2] },
                { type: DataType.FLOAT, shape: [1, 1, 2] },
            ]);
            if (y.is(TensorNode)) y.as(TensorNode).setType("output");
            if (yh.is(TensorNode)) yh.as(TensorNode).setType("output");
            return g;
        };

        const origGraph = buildRnnGraph();
        const loweredGraph = buildRnnGraph();

        new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            coalesce: true,
            recurse: true,
            optimize: true,
            indexingMode: IndexingMode.Pointer,
        }).apply(loweredGraph, false);

        expect(() => validateGraph(loweredGraph)).not.toThrow();

        // Confirm nested Loops (depth >= 2) were synthesized and no MatMul ops remain anywhere
        const maxLoopDepth = (gr: OnnxGraph.Class, depth: number = 0): number => {
            let maxD = depth;
            for (const op of gr.getOperationNodes()) {
                const nextD = op.type === "Loop" ? depth + 1 : depth;
                if (nextD > maxD) maxD = nextD;
                for (const reg of op.regions) {
                    maxD = Math.max(maxD, maxLoopDepth(reg, nextD));
                }
            }
            return maxD;
        };
        const collectTypes = (gr: OnnxGraph.Class, acc: string[] = []): string[] => {
            for (const op of gr.getOperationNodes()) {
                acc.push(op.type);
                for (const reg of op.regions) collectTypes(reg, acc);
            }
            return acc;
        };

        expect(maxLoopDepth(loweredGraph)).toBeGreaterThanOrEqual(2);
        expect(collectTypes(loweredGraph)).not.toContain("MatMul");
        expect(collectTypes(loweredGraph)).not.toContain("RNN");

        // Verify numerical equivalence against ONNX Runtime
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-rec-loop-"));
        try {
            const origPath = path.join(tmpDir, "orig_rnn.onnx");
            const loweredPath = path.join(tmpDir, "lowered_rnn.onnx");
            await exportGraphToOnnx(origGraph, origPath);
            await exportGraphToOnnx(loweredGraph, loweredPath);

            const feeds = buildFeeds(
                [{ name: "X", dtype: "float32", shape: [2, 1, 3], gen: "negmix" }],
                "rec_rnn_seed",
            );
            const origSession = await ort.InferenceSession.create(origPath);
            const loweredSession = await ort.InferenceSession.create(loweredPath);
            const origOut = await origSession.run(feeds);
            const loweredOut = await loweredSession.run(feeds);

            const cmp = compareOrtOutputs(origOut, loweredOut, { tol: 1e-3 });
            expect(cmp.passed, cmp.message).toBe(true);
        } finally {
            if (fs.existsSync(tmpDir)) {
                fs.rmSync(tmpDir, { recursive: true, force: true });
            }
        }
    });
});
