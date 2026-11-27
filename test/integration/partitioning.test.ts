import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { exportGraphToOnnx } from "../helpers/OrtHarness.js";
import { runPartitionTest } from "../helpers/PartitionHarness.js";
import {
    buildResidualSkipForkGraph,
    buildSplitMultiBranchConcatGraph,
    ensureModelExists,
} from "../helpers/ModelFactory.js";
import { PARTITION_TESTS } from "../helpers/ModelRegistry.js";

const TEST_TIMEOUT = 150000; //ms

describe("Integration: Graph Partitioning Equivalence", () => {
    it.each(PARTITION_TESTS)(
        "partitions graph and matches original execution for: $label",
        async (tc) => {
            const absPath = await ensureModelExists(tc.originalPath);
            expect(fs.existsSync(absPath), `Model not found at resolved path: ${absPath}`).toBe(
                true,
            );

            await expect(runPartitionTest({ ...tc, originalPath: absPath })).resolves.not.toThrow();
        },
        TEST_TIMEOUT,
    );

    describe("Multi-Branch Subgraph Partitioning", () => {
        let tmpDir: string;

        beforeAll(() => {
            tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-part-combo-"));
        });

        afterAll(() => {
            if (tmpDir && fs.existsSync(tmpDir)) {
                fs.rmSync(tmpDir, { recursive: true, force: true });
            }
        });

        it("partitions a Residual Skip-Fork DAG (skip tensor crosses cut boundary)", async () => {
            const modelPath = path.join(tmpDir, "res_skip_fork.onnx");
            await exportGraphToOnnx(buildResidualSkipForkGraph("part_res"), modelPath);

            await expect(
                runPartitionTest({
                    label: "combo_part_residual_skip_fork",
                    originalPath: modelPath,
                    specs: [{ name: "X", dtype: "float32", shape: [1, 2, 4, 4], gen: "negmix" }],
                    tol: 1e-4,
                }),
            ).resolves.not.toThrow();
        });

        it("partitions a Split -> Asymmetric Branches -> Concat DAG", async () => {
            const modelPath = path.join(tmpDir, "split_branch_concat.onnx");
            await exportGraphToOnnx(buildSplitMultiBranchConcatGraph("part_sc"), modelPath);

            await expect(
                runPartitionTest({
                    label: "combo_part_split_branch_concat",
                    originalPath: modelPath,
                    specs: [{ name: "X", dtype: "float32", shape: [2, 4], gen: "negmix" }],
                    tol: 1e-5,
                }),
            ).resolves.not.toThrow();
        });
    });
});
