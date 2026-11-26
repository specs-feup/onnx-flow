import { describe, it, expect } from "vitest";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { Recipes } from "../../../src/onnx/transformation/recipes.js";
import { applyRecipe } from "../../helpers/recipeHarness.js";
import { DataType } from "../../../src/onnx/ir/OnnxTypes.js";
import Graph from "@specs-feup/flow/graph/Graph";
import { GraphBuilder } from "../../../src/onnx/ir/GraphBuilder.js";
import OnnxGraph from "../../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../../src/onnx/ir/TensorNode.js";
import { int64Vec } from "../../../src/onnx/utils/index.js";

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

    it("decomposes AveragePool with auto_pad VALID + ceil_mode=1 and SAME_UPPER into explicit Conv pads", () => {
        const validCeil = applyRecipe(
            new Recipes.canonicalization.AveragePool(),
            "AveragePool",
            [[1, 1, 1, 4]],
            {
                kernel_shape: [1, 3],
                strides: [1, 2],
                auto_pad: "VALID",
                ceil_mode: 1,
                count_include_pad: 0,
            },
        );
        expect(validCeil.producedOps).toContain("Conv");
        const convNodes = validCeil.graph
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Conv");
        // Expect 2 Conv ops (sum + mask) with ceil-augmented pads [0, 0, 0, 2]
        expect(convNodes.length).toBe(2);
        expect(convNodes[0].attributes["auto_pad"]).toBeUndefined();
        expect(convNodes[0].attributes["pads"]).toEqual([0, 0, 0, 2]);

        const sameUpper = applyRecipe(
            new Recipes.canonicalization.AveragePool(),
            "AveragePool",
            [[1, 1, 4, 4]],
            {
                kernel_shape: [3, 3],
                strides: [2, 2],
                auto_pad: "SAME_UPPER",
            },
        );
        const suConv = sameUpper.graph
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Conv")!;
        expect(suConv.attributes["auto_pad"]).toBeUndefined();
        expect(suConv.attributes["pads"]).toEqual([0, 0, 1, 1]);
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
                { shape: [], dataType: DataType.UINT8 },
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

    it("decomposes pointwise 1x1 Conv to MatMul (transpose-free when N=1 and general when N>1)", () => {
        const recipe = new Recipes.canonicalization.Conv1x1ToMatMul();

        // N = 1: transpose-free path
        const g1 = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b1 = new GraphBuilder(g1, "c1");
        const x1 = g1
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 3, 4, 4], "input"))
            .as(TensorNode);
        const w1 = g1
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [5, 3, 1, 1], "input"))
            .as(TensorNode);
        b1.createOp("Conv", [x1, w1], { kernel_shape: [1, 1] }, [
            { type: DataType.FLOAT, shape: [1, 5, 4, 4] },
        ]);
        const op1 = g1
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Conv")!;
        expect(recipe.match(op1)).toBe(true);
        recipe.apply(op1, b1);
        const ops1 = g1
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(ops1).toContain("MatMul");
        expect(ops1).not.toContain("Transpose");
        expect(ops1).not.toContain("Conv");

        // N = -1, spatial = [-1, -1]: dynamic general path
        const gDyn = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bDyn = new GraphBuilder(gDyn, "cdyn");
        const xDyn = gDyn
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 3, -1, -1], "input"))
            .as(TensorNode);
        const wDyn = gDyn
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 3, 1, 1], "input"))
            .as(TensorNode);
        bDyn.createOp("Conv", [xDyn, wDyn], { kernel_shape: [1, 1], auto_pad: "SAME_UPPER" }, [
            { type: DataType.FLOAT, shape: [-1, 4, -1, -1] },
        ]);
        const opDyn = gDyn
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Conv")!;
        expect(recipe.match(opDyn)).toBe(true);
        recipe.apply(opDyn, bDyn);
        const opsDyn = gDyn
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(opsDyn).toContain("MatMul");
        expect(opsDyn).toContain("Transpose");
        expect(opsDyn).not.toContain("Conv");
    });
});
