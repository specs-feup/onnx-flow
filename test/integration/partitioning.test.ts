import { describe, it, expect } from "vitest";
import fs from "node:fs";
import { runPartitionTest } from "../helpers/partitionHarness.js";
import { ensureModelExists } from "../helpers/modelFactory.js";
import { PARTITION_TESTS } from "../helpers/modelRegistry.js";

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
        30000,
    );
});
