import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { scalarFloat, floatVec, topologicalSortOperationNodes } from "../../../src/Onnx/Utils.js";
import { applyRecipe } from "../../helpers/recipeHarness.js";

describe("Canonicalization - Binary & Comparison Recipes", () => {
    initializeSchemaRegistry();

    it("decomposes Div, Mod, and Pow", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.Div(), "Div", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Div");
        expect(
            applyRecipe(new Recipes.canonicalization.Mod(), "Mod", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Mod");
        expect(
            applyRecipe(new Recipes.canonicalization.Pow(), "Pow", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Pow");
    });

    it("decomposes Min and Max", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.Min(), "Min", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Min");
        expect(
            applyRecipe(new Recipes.canonicalization.Max(), "Max", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Max");
    });

    it("decomposes Comparisons (Less, LessOrEqual, GreaterOrEqual)", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.Less(), "Less", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Less");
        expect(
            applyRecipe(new Recipes.canonicalization.LessOrEqual(), "LessOrEqual", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("LessOrEqual");
        expect(
            applyRecipe(new Recipes.canonicalization.GreaterOrEqual(), "GreaterOrEqual", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("GreaterOrEqual");
    });

    it("decomposes Logical Or", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.Or(),
            "Or",
            [
                [2, 2],
                [2, 2],
            ],
            {},
            DataType.BOOL,
        );
        expect(producedOps).not.toContain("Or");
    });

    it("decomposes Sub", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Sub(), "Sub", [
            [2, 2],
            [2, 2],
        ]);
        expect(producedOps).not.toContain("Sub");
    });

    it("decomposes Clip", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Clip(), "Clip", [
            [2, 2],
            [],
            [],
        ]);
        expect(producedOps).not.toContain("Clip");
    });

    it("decomposes PRelu, Sum, Mean, CastLike, and Scatter", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.PRelu(), "PRelu", [[2, 2], [2]]).producedOps,
        ).not.toContain("PRelu");
        expect(
            applyRecipe(new Recipes.canonicalization.Sum(), "Sum", [
                [2, 2],
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Sum");
        expect(
            applyRecipe(new Recipes.canonicalization.Mean(), "Mean", [
                [2, 2],
                [2, 2],
            ]).producedOps,
        ).not.toContain("Mean");
        expect(
            applyRecipe(new Recipes.canonicalization.CastLike(), "CastLike", [
                { shape: [2, 2], dataType: DataType.INT32 },
                { shape: [2, 2], dataType: DataType.FLOAT },
            ]).producedOps,
        ).toContain("Cast");
        expect(
            applyRecipe(
                new Recipes.canonicalization.Scatter(),
                "Scatter",
                [
                    { shape: [2, 4], dataType: DataType.FLOAT },
                    { shape: [2, 2], dataType: DataType.INT64 },
                    { shape: [2, 2], dataType: DataType.FLOAT },
                ],
                { axis: 1 },
            ).producedOps,
        ).toContain("ScatterElements");
    });

    it("decomposes 1-input and 3-input variadic Min and Max", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.VariadicMin(), "Min", [[2, 2]]).producedOps,
        ).toContain("Identity");
        expect(
            applyRecipe(new Recipes.canonicalization.VariadicMax(), "Max", [
                [2, 2],
                [2, 2],
                [2, 2],
            ]).producedOps.filter((t) => t === "Max").length,
        ).toBe(2);
    });

    it("lowers Div by constant via LowerDivByConstant into Mul or Identity without Reciprocal", () => {
        const buildDiv = (val: number | number[]) => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "divc");
            const a = g
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
                .as(TensorNode);
            const c = b.createConstant("c", Array.isArray(val) ? floatVec(val) : scalarFloat(val));
            b.createOp("Div", [a, c], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);
            const op = g
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Div")!;
            return { g, b, op };
        };

        const recipe = new Recipes.canonicalization.DivByConstant();
        const { g: g2, b: b2, op: op2 } = buildDiv(4.0);
        expect(recipe.match(op2)).toBe(true);
        recipe.apply(op2, b2);
        const ops2 = g2
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(ops2).toEqual(["Mul"]);

        const { g: g1, b: b1, op: op1 } = buildDiv(1.0);
        expect(recipe.match(op1)).toBe(true);
        recipe.apply(op1, b1);
        expect(
            g1
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toEqual(["Identity"]);

        const { op: op0 } = buildDiv(0.0);
        expect(recipe.match(op0)).toBe(false);
    });

    it("lowers Pow with small constant exponents via LowerPowSmallInt into Mul/Sqrt/Reciprocal", () => {
        const buildPow = (expVal: number) => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "pows");
            const a = g
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                .as(TensorNode);
            const e = b.createConstant("e", scalarFloat(expVal));
            b.createOp("Pow", [a, e], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
            const op = g
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Pow")!;
            return { g, b, op };
        };

        const recipe = new Recipes.canonicalization.PowSmallInt();

        const { g: gSq, b: bSq, op: opSq } = buildPow(2);
        expect(recipe.match(opSq)).toBe(true);
        recipe.apply(opSq, bSq);
        expect(
            gSq
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toEqual(["Mul"]);

        const { g: gCu, b: bCu, op: opCu } = buildPow(3);
        recipe.apply(opCu, bCu);
        expect(
            gCu
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toEqual(["Mul", "Mul"]);

        const { g: gInvSqrt, b: bInvSqrt, op: opInvSqrt } = buildPow(-0.5);
        recipe.apply(opInvSqrt, bInvSqrt);
        expect(topologicalSortOperationNodes(gInvSqrt).map((o) => o.type)).toEqual([
            "Sqrt",
            "Reciprocal",
        ]);

        const { op: opNonInt } = buildPow(2.7);
        expect(recipe.match(opNonInt)).toBe(false);
    });

    it("lowers Div by exact power-of-two constants as exact precision via LowerDivByExactPowerOfTwo", () => {
        const buildDiv = (val: number) => {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "divp2");
            const a = g
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
                .as(TensorNode);
            const c = b.createConstant("c", scalarFloat(val));
            b.createOp("Div", [a, c], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);
            const op = g
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Div")!;
            return { g, b, op };
        };

        const exactRecipe = new Recipes.canonicalization.DivByExactPowerOfTwo();
        expect(exactRecipe.precision).toBe("exact");

        const { g: g4, b: b4, op: op4 } = buildDiv(4.0);
        expect(exactRecipe.match(op4)).toBe(true);
        exactRecipe.apply(op4, b4);
        expect(
            g4
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toEqual(["Mul"]);

        const { op: op3 } = buildDiv(3.0);
        expect(exactRecipe.match(op3)).toBe(false);
    });
});
