import { describe, it, expect } from "vitest";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { applyRecipe } from "./harness.js";

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
});
