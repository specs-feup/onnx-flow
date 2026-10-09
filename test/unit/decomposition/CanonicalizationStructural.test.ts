import { describe, it, expect } from "vitest";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { applyRecipe } from "./harness.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import Graph from "@specs-feup/flow/graph/Graph";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { int64Vec } from "../../../src/Onnx/Utils.js";

describe("Canonicalization - Structural & Normalization Recipes", () => {
    initializeSchemaRegistry();

    it("decomposes Softmax", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Softmax(), "Softmax", [
            [2, 4],
        ]);
        expect(producedOps).not.toContain("Softmax");
    });

    it("decomposes AveragePool", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.AveragePool(),
            "AveragePool",
            [[1, 1, 4, 4]],
            {
                kernel_shape: [2, 2],
                strides: [1, 1], // Different from kernel_shape so it bypasses the return false check
                pads: [0, 0, 0, 0],
            },
        );
        expect(producedOps).not.toContain("AveragePool");
    });

    it("decomposes Concat", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.Concat(),
            "Concat",
            [
                [2, 3],
                [2, 3],
            ],
            { axis: 0 },
        );
        expect(producedOps).not.toContain("Concat");
    });

    it("decomposes QuantizeLinear and DequantizeLinear", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.QuantizeLinear(), "QuantizeLinear", [
                [2, 2],
                [],
                [],
            ]).producedOps,
        ).not.toContain("QuantizeLinear");

        expect(
            applyRecipe(new Recipes.canonicalization.DequantizeLinear(), "DequantizeLinear", [
                [2, 2],
                [],
                [],
            ]).producedOps,
        ).not.toContain("DequantizeLinear");
    });

    it("decomposes BatchNormalization", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.BatchNormalization(),
            "BatchNormalization",
            [[1, 4, 2, 2], [4], [4], [4], [4]],
        );
        expect(producedOps).not.toContain("BatchNormalization");
    });

    it("decomposes LayerNormalization", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.LayerNormalization(),
            "LayerNormalization",
            [[2, 4], [4], [4]],
            { axis: -1, epsilon: 1e-5 },
        );
        expect(producedOps).not.toContain("LayerNormalization");
    });

    it("decomposes InstanceNormalization", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.InstanceNormalization(),
            "InstanceNormalization",
            [[1, 4, 2, 2], [4], [4]],
            { epsilon: 1e-5 },
        );
        expect(producedOps).not.toContain("InstanceNormalization");
    });

    it("decomposes GroupNormalization", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.GroupNormalization(),
            "GroupNormalization",
            [[1, 4, 2, 2], [4], [4]],
            { num_groups: 2, epsilon: 1e-5 },
        );
        expect(producedOps).not.toContain("GroupNormalization");
    });

    it("decomposes Gemm", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.Gemm(),
            "Gemm",
            [
                [2, 3],
                [3, 4],
                [2, 4],
            ],
            { alpha: 1.0, beta: 1.0 },
        );
        expect(producedOps).not.toContain("Gemm");
    });

    it("decomposes Expand", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Expand(), "Expand", [
            [3],
            { shape: [2], dataType: DataType.INT64 },
        ]);
        expect(producedOps).not.toContain("Expand");
    });

    it("decomposes Pad", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "test");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 1, 4, 4], "input"))
            .as(TensorNode);

        // Input 1 must be a ConstantNode containing the pad values [0, 0, 1, 1, 0, 0, 1, 1]
        const pads = builder.createConstant("pads", int64Vec([0, 0, 1, 1, 0, 0, 1, 1]));

        builder.createOp("Pad", [x, pads], { mode: "constant" });

        const targetOp = graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Pad")!;

        const recipe = new Recipes.canonicalization.Pad();
        expect(recipe.match(targetOp)).toBe(true);
        recipe.apply(targetOp, builder);

        const producedOps = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(producedOps).not.toContain("Pad");
    });

    it("decomposes Slice", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "test");

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 4], "input"))
            .as(TensorNode);

        // Inputs 1 and 2 must be ConstantNodes with integer vectors
        const starts = builder.createConstant("starts", int64Vec([0, 0]));
        const ends = builder.createConstant("ends", int64Vec([2, 2]));

        builder.createOp("Slice", [x, starts, ends]);

        const targetOp = graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Slice")!;

        const recipe = new Recipes.canonicalization.Slice();
        expect(recipe.match(targetOp)).toBe(true);
        recipe.apply(targetOp, builder);

        const producedOps = graph
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(producedOps).not.toContain("Slice");
    });
});
