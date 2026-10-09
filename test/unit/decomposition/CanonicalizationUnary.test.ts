import { describe, it, expect } from "vitest";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { applyRecipe } from "./harness.js";
import { LowerExpRecipe } from "../../../src/Onnx/transformation/canonicalization/recipes/LowerExpRecipe.js";
import { LowerReciprocalRecipe } from "../../../src/Onnx/transformation/canonicalization/recipes/LowerReciprocalRecipe.js";

describe("Canonicalization - Unary & Activation Recipes", () => {
    initializeSchemaRegistry();

    it("decomposes Abs", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Abs(), "Abs", [[2]]);
        expect(producedOps).not.toContain("Abs");
    });

    it("decomposes Sign", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Sign(), "Sign");
        expect(producedOps).not.toContain("Sign");
    });

    it("decomposes Ceil", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Ceil(), "Ceil");
        expect(producedOps).not.toContain("Ceil");
    });

    it("decomposes Sqrt", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Sqrt(), "Sqrt");
        expect(producedOps).not.toContain("Sqrt");
    });

    it("decomposes Log", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Log(), "Log");
        expect(producedOps).not.toContain("Log");
    });

    it("decomposes Trigonometric ops (Sin, Cos, Tan)", () => {
        expect(applyRecipe(new Recipes.canonicalization.Sin(), "Sin").producedOps).not.toContain(
            "Sin",
        );
        expect(applyRecipe(new Recipes.canonicalization.Cos(), "Cos").producedOps).not.toContain(
            "Cos",
        );
        expect(applyRecipe(new Recipes.canonicalization.Tan(), "Tan").producedOps).not.toContain(
            "Tan",
        );
    });

    it("decomposes Hyperbolic ops (Sinh, Cosh)", () => {
        expect(applyRecipe(new Recipes.canonicalization.Sinh(), "Sinh").producedOps).not.toContain(
            "Sinh",
        );
        expect(applyRecipe(new Recipes.canonicalization.Cosh(), "Cosh").producedOps).not.toContain(
            "Cosh",
        );
    });

    it("decomposes Inverse Trig ops (Asin, Acos, Atan)", () => {
        expect(applyRecipe(new Recipes.canonicalization.Asin(), "Asin").producedOps).not.toContain(
            "Asin",
        );
        expect(applyRecipe(new Recipes.canonicalization.Acos(), "Acos").producedOps).not.toContain(
            "Acos",
        );
        expect(applyRecipe(new Recipes.canonicalization.Atan(), "Atan").producedOps).not.toContain(
            "Atan",
        );
    });

    it("decomposes Inverse Hyperbolic ops (Asinh, Acosh, Atanh)", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.Asinh(), "Asinh").producedOps,
        ).not.toContain("Asinh");
        expect(
            applyRecipe(new Recipes.canonicalization.Acosh(), "Acosh").producedOps,
        ).not.toContain("Acosh");
        expect(
            applyRecipe(new Recipes.canonicalization.Atanh(), "Atanh").producedOps,
        ).not.toContain("Atanh");
    });

    it("decomposes Activations (LeakyRelu, Elu, Celu, Selu, Softplus, Softsign, Mish)", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.LeakyRelu(), "LeakyRelu").producedOps,
        ).not.toContain("LeakyRelu");
        expect(applyRecipe(new Recipes.canonicalization.Elu(), "Elu").producedOps).not.toContain(
            "Elu",
        );
        expect(applyRecipe(new Recipes.canonicalization.Celu(), "Celu").producedOps).not.toContain(
            "Celu",
        );
        expect(applyRecipe(new Recipes.canonicalization.Selu(), "Selu").producedOps).not.toContain(
            "Selu",
        );
        expect(
            applyRecipe(new Recipes.canonicalization.Softplus(), "Softplus").producedOps,
        ).not.toContain("Softplus");
        expect(
            applyRecipe(new Recipes.canonicalization.Softsign(), "Softsign").producedOps,
        ).not.toContain("Softsign");
        expect(applyRecipe(new Recipes.canonicalization.Mish(), "Mish").producedOps).not.toContain(
            "Mish",
        );
    });

    it("decomposes Sigmoid variants and Erf", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.Sigmoid(), "Sigmoid").producedOps,
        ).not.toContain("Sigmoid");
        expect(
            applyRecipe(new Recipes.canonicalization.HardSigmoid(), "HardSigmoid").producedOps,
        ).not.toContain("HardSigmoid");
        expect(
            applyRecipe(new Recipes.canonicalization.TanhSigmoid(), "TanhSigmoid").producedOps,
        ).not.toContain("TanhSigmoid");
        expect(
            applyRecipe(new Recipes.canonicalization.TanhDiv(), "TanhDiv").producedOps,
        ).not.toContain("TanhDiv");
        expect(applyRecipe(new Recipes.canonicalization.Erf(), "Erf").producedOps).not.toContain(
            "Erf",
        );
    });

    it("decomposes IsNaN", () => {
        // ONNX operator name uses capital 'N'
        const { producedOps } = applyRecipe(new Recipes.canonicalization.IsNan(), "IsNaN");
        expect(producedOps).not.toContain("IsNaN");
    });

    it("decomposes IsInf", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.IsInf(), "IsInf");
        expect(producedOps).not.toContain("IsInf");
    });

    it("decomposes Not", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.Not(),
            "Not",
            [[2]],
            {},
            DataType.BOOL,
        );
        expect(producedOps).not.toContain("Not");
    });

    it("decomposes Round variants (RoundAdd, RoundStep)", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.RoundAdd(), "Round").producedOps,
        ).not.toContain("Round");
        expect(
            applyRecipe(new Recipes.canonicalization.RoundStep(), "Round").producedOps,
        ).not.toContain("Round");
    });

    it("decomposes Relu", () => {
        const { producedOps } = applyRecipe(new Recipes.canonicalization.Relu(), "Relu", [[2, 2]]);
        expect(producedOps).not.toContain("Relu");
    });

    it("decomposes Exp", () => {
        const { producedOps } = applyRecipe(new LowerExpRecipe(), "Exp", [[2, 2]]);
        expect(producedOps).not.toContain("Exp");
    });

    it("decomposes Reciprocal", () => {
        const { producedOps } = applyRecipe(new LowerReciprocalRecipe(), "Reciprocal", [[2, 2]]);
        expect(producedOps).not.toContain("Reciprocal");
    });
});
