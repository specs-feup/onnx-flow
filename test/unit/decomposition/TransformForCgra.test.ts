import { describe, it, expect, vi } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import transformForCgra from "../../../examples/cgra-decomposition-example/TransformForCgra.js";
import { AddGridDecompositionRecipe } from "../../../examples/cgra-decomposition-example/Add.js";
import { ReluGridDecompositionRecipe } from "../../../examples/cgra-decomposition-example/Relu.js";
import { applyRecipe } from "../../helpers/recipeHarness.js";

describe("CGRA Example Grid Decomposition", () => {
    initializeSchemaRegistry();

    it("decomposes 2D MatMul via transformForCgra (with intermediate & output consumers)", () => {
        const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
        try {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "cgra");
            const a = g
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                .as(TensorNode);
            const w = g
                .addNode("W")
                .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
                .as(TensorNode);
            const bias = g
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
                .as(TensorNode);
            const [mm] = b.createOp("MatMul", [a, w], {}, [
                { type: DataType.FLOAT, shape: [2, 2] },
            ]);
            const [out] = b.createOp("Add", [mm, bias], {}, [
                { type: DataType.FLOAT, shape: [2, 2] },
            ]);
            if (out.is(TensorNode)) out.as(TensorNode).setType("output");

            transformForCgra(g);

            const opTypes = g
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            expect(opTypes).not.toContain("MatMul");
            expect(opTypes).toContain("Mul");
            expect(opTypes).toContain("ReduceSum");
            expect(opTypes).toContain("Concat");
        } finally {
            logSpy.mockRestore();
        }
    });

    it("decomposes 2D Add and 2D Relu via CGRA grid recipes", () => {
        const addRes = applyRecipe(new AddGridDecompositionRecipe(), "Add", [
            [2, 3],
            [2, 3],
        ]);
        expect(addRes.producedOps).toContain("Gather");
        expect(addRes.producedOps).toContain("Concat");

        const reluRes = applyRecipe(new ReluGridDecompositionRecipe(), "Relu", [[2, 3]]);
        expect(reluRes.producedOps).not.toContain("Relu");
        expect(reluRes.producedOps).toContain("Where");
    });
});
