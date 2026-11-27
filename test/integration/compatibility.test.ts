import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { executeEquivalenceTest } from "../helpers/OrtHarness.js";
import { COMPATIBILITY_TESTS } from "../helpers/ModelRegistry.js";
import { ensureModelExists } from "../helpers/ModelFactory.js";

const TEST_TIMEOUT = 150000; //ms

describe("Integration: ONNX Model Compatibility & Lowering Equivalence", () => {
    let tmpDir: string;

    beforeAll(() => {
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-compat-"));
    });

    afterAll(() => {
        if (tmpDir && fs.existsSync(tmpDir)) {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it.each(COMPATIBILITY_TESTS)(
        "produces equivalent outputs for: $label",
        async (config) => {
            const absOrig = await ensureModelExists(config.originalPath);
            expect(fs.existsSync(absOrig), `Model not found at resolved path: ${absOrig}`).toBe(
                true,
            );

            const result = await executeEquivalenceTest(config, {
                skipCli: false,
                showStats: process.env["TEST_STATS"] === "true",
                verboseOnFailure: true,
                workDir: tmpDir,
            });

            expect(result.status, result.message ?? "Expected model outputs to match").toBe("pass");
        },
        TEST_TIMEOUT,
    );
});
