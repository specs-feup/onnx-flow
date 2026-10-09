import { describe, it, expect } from "vitest";
import fs from "fs";
import { runPartitionTest } from "../partitioning_test_utils.js";
import { resolveModelPath } from "../compatibility_test.js";
import { TESTS as PARTITION_TESTS } from "../partition_test.js";

describe("Integration: Graph Partitioning Equivalence", () => {
    it.each(PARTITION_TESTS)(
        "partitions graph and matches original execution for: $label",
        async (tc) => {
            const absPath = resolveModelPath(tc.originalPath);
            if (!fs.existsSync(absPath)) {
                throw new Error(`Model not found at resolved path: ${absPath}`);
            }

            await expect(runPartitionTest({ ...tc, originalPath: absPath })).resolves.not.toThrow();
        },
        30000,
    );
});
