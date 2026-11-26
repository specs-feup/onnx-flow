import { describe, it, expect } from "vitest";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
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
});
