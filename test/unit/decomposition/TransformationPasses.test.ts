import { describe, it, expect, vi } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { CanonicalizationPass } from "../../../src/Onnx/transformation/canonicalization/CanonicalizationPass.js";
import { InferShapesPass } from "../../../src/Onnx/transformation/InferShapesPass.js";
import { OrchestratorPass } from "../../../src/Onnx/transformation/OrchestratorPass.js";
import { TransformationRegistry } from "../../../src/Onnx/transformation/TransformationRegistry.js";
import OnnxGraphTransformer from "../../../src/Onnx/transformation/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import type { DecompositionRecipe } from "../../../src/Onnx/transformation/Recipe.js";

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
});
