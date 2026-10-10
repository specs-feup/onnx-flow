import { describe, it, expect } from "vitest";
import fs from "fs";
import { executeEquivalenceTest, TESTS, resolveModelPath } from "../compatibility_test.js";

describe("Integration: ONNX Model Compatibility & Lowering Equivalence", () => {
    it.each(TESTS)(
        "produces equivalent outputs for: $label",
        async (config) => {
            const absOrig = resolveModelPath(config.originalPath);
            if (!fs.existsSync(absOrig)) {
                throw new Error(`Model not found at resolved path: ${absOrig}`);
            }

            const result = await executeEquivalenceTest(config, {
                skipCli: false,
                showStats: process.env["TEST_STATS"] === "true",
                verboseOnFailure: true,
            });

            expect(result.status, result.message ?? "Expected model outputs to match").toBe("pass");
        },
        30000,
    );
});
