import { describe, it, expect, vi } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import OperationNode from "../../../src/onnx/ir/OperationNode.js";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import { BaseGraphPass, type GraphPass } from "../../../src/onnx/transformation/GraphPass.js";
import { PassManager } from "../../../src/onnx/transformation/PassManager.js";
import { BaseCanonicalizationRecipe } from "../../../src/onnx/transformation/canonicalization/BaseCanonicalizationRecipe.js";
import { getRecipeMetadata } from "../../../src/onnx/transformation/Recipe.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import { HistoryManager } from "../../../src/onnx/transformation/tracking/HistoryManager.js";
import { topologicalSortOperationNodes } from "../../../src/onnx/utils/index.js";

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

    it("BaseGraphPass.runOnRegions visits nested control-flow regions recursively", () => {
        const visitedGraphs: OnnxGraph.Class[] = [];
        class RegionVisitorPass extends BaseGraphPass {
            public override readonly name = "RegionVisitor";
            public override run(graph: OnnxGraph.Class): boolean {
                visitedGraphs.push(graph);
                return this.runOnRegions(graph);
            }
        }

        const outer = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const inner = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        outer.addNode("loop0").init(new OperationNode.Builder("Loop", [], {}, [inner]));

        const pass = new RegionVisitorPass();
        pass.run(outer);
        expect(visitedGraphs).toEqual([outer, inner]);
    });

    it("BaseCanonicalizationRecipe supplies standard metadata defaults and replaceWith helper", () => {
        class IdentityNegRecipe extends BaseCanonicalizationRecipe {
            public override readonly name = "IdentityNegTest";
            public override readonly targetOp = "Neg";
            public override readonly producedOps = ["Identity"];

            public override match(node: OperationNode.Class): boolean {
                return node.type === "Neg";
            }

            public override apply(node: OperationNode.Class, builder: GraphBuilder): void {
                const [idOut] = builder.createOp("Identity", [node.getInputs()![0]]);
                this.replaceWith(node, idOut, builder);
            }
        }

        const recipe = new IdentityNegRecipe();
        const meta = getRecipeMetadata(recipe);
        expect(meta.phase).toBe("canonicalization");
        expect(meta.precision).toBe("exact");
        expect(meta.priority).toBe(0);
        expect(meta.shapeSupport).toBe("dynamic");
        expect(meta.exposesControlFlow).toBe(false);
        expect(meta.exposesDataAccess).toBe(false);

        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "base_recipe");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const [negOut] = b.createOp("Neg", [x], {}, [{ type: DataType.FLOAT, shape: [2] }]);
        if (negOut.is(TensorNode)) negOut.as(TensorNode).setType("output");

        const negOp = g.getOperationNodes().first()!;
        expect(recipe.match(negOp)).toBe(true);
        recipe.apply(negOp, b);

        expect(
            g
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toEqual(["Identity"]);
    });

    it("keeps initializeSchemaRegistry and PassManager silent by default unless verbose is enabled", () => {
        const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
        try {
            initializeSchemaRegistry();
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const pm = new PassManager();
            pm.addPass({ name: "NoOp", run: () => false });
            pm.run(g);

            expect(logSpy).not.toHaveBeenCalled();

            initializeSchemaRegistry(true);
            pm.run(g, undefined, 1, true);
            expect(logSpy).toHaveBeenCalled();
        } finally {
            logSpy.mockRestore();
        }
    });
});
