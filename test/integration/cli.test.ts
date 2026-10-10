import { describe, it, expect, beforeAll, afterEach } from "vitest";
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";
import { ensureProtoExists, PROJECT_ROOT, resolveModelPath } from "../compatibility_test.js";

describe("CLI & Partitioning End-to-End Tests", () => {
    const cliEntry = path.resolve(PROJECT_ROOT, "out/src/index.js");
    const testModelRel = "examples/onnx/add_chain_standard.onnx";
    const testModelAbs = resolveModelPath(testModelRel);

    const generatedFiles: string[] = [];

    beforeAll(() => {
        ensureProtoExists();
        expect(fs.existsSync(cliEntry), "Build out/src/index.js before running CLI tests").toBe(
            true,
        );
    });

    afterEach(() => {
        for (const file of generatedFiles) {
            if (fs.existsSync(file)) {
                try {
                    fs.unlinkSync(file);
                } catch {
                    /* cleanup */
                }
            }
        }
        generatedFiles.length = 0;
    });

    function runCli(args: string): string {
        return execSync(`node "${cliEntry}" ${args}`, {
            cwd: PROJECT_ROOT,
            encoding: "utf-8",
            stdio: ["pipe", "pipe", "pipe"],
        });
    }

    it("runs default reconversion and outputs artifacts", () => {
        const outBase = path.resolve(PROJECT_ROOT, "temp_cli_standard");
        const outDot = `${outBase}.dot`;
        generatedFiles.push(outDot);

        const output = runCli(`"${testModelAbs}" --format dot -vz 0 -v 1 -o "${outDot}"`);
        expect(output).toContain("Reconverted ONNX written to");
        expect(fs.existsSync(outDot)).toBe(true);
    });

    it("fails validation when invalid partition arguments are passed", () => {
        expect(() => {
            runCli(`"${testModelAbs}" --partition -vz 0`);
        }).toThrow(/Argument check failed: --partition requires either 1 argument/);

        expect(() => {
            runCli(`"${testModelAbs}" --partition Op 1 extra -vz 0`);
        }).toThrow(/Argument check failed: --partition requires either 1 argument/);
    });

    it("fails gracefully when a non-existent operation instance is requested", () => {
        expect(() => {
            runCli(`"${testModelAbs}" --partition NonExistentOp 1 -vz 0`);
        }).toThrow(/Could not find instance 1 of operation 'NonExistentOp'/);

        expect(() => {
            runCli(`"${testModelAbs}" --partition Add notANumber -vz 0`);
        }).toThrow(/Invalid instance number: notANumber/);
    });

    it("partitions graph using <OpType> <Instance> and executes head/tail in ONNX Runtime", async () => {
        const baseName = path.resolve(PROJECT_ROOT, "temp_part_op");
        const headOnnx = `${baseName}_head.onnx`;
        const tailOnnx = `${baseName}_tail.onnx`;
        const headJson = `${baseName}_head.json`;
        const tailJson = `${baseName}_tail.json`;

        generatedFiles.push(headOnnx, tailOnnx, headJson, tailJson);

        const cliOutput = runCli(
            `"${testModelAbs}" --partition Add 1 -o "${baseName}.json" -vz 0 -v 1`,
        );

        expect(cliOutput).toContain("Partitioning graph at node:");
        expect(cliOutput).toContain("Created ONNX partition:");
        expect(fs.existsSync(headOnnx)).toBe(true);
        expect(fs.existsSync(tailOnnx)).toBe(true);

        // Verify end-to-end numeric equivalence through ORT
        const feeds = {
            A: new ort.Tensor("float32", new Float32Array([1, 2, 3, 4]), [4]),
            B: new ort.Tensor("float32", new Float32Array([5, 6, 7, 8]), [4]),
            C: new ort.Tensor("float32", new Float32Array([1, 1, 1, 1]), [4]),
            D: new ort.Tensor("float32", new Float32Array([2, 2, 2, 2]), [4]),
        };

        // 1. Run full reference model
        const origSession = await ort.InferenceSession.create(testModelAbs);
        const origResults = await origSession.run(feeds);
        const origOutput = Array.from(Object.values(origResults)[0].data as Float32Array);

        // 2. Run Head partition
        const headSession = await ort.InferenceSession.create(headOnnx);
        const headFeeds: Record<string, ort.Tensor> = {};
        for (const inputName of headSession.inputNames) {
            if (inputName in feeds) headFeeds[inputName] = feeds[inputName as keyof typeof feeds];
        }
        const headResults = await headSession.run(headFeeds);

        // 3. Run Tail partition with inputs from original feeds + Head outputs
        const tailSession = await ort.InferenceSession.create(tailOnnx);
        const tailFeeds: Record<string, ort.Tensor> = {};
        for (const inputName of tailSession.inputNames) {
            if (inputName in headResults) {
                tailFeeds[inputName] = headResults[inputName];
            } else if (inputName in feeds) {
                tailFeeds[inputName] = feeds[inputName as keyof typeof feeds];
            }
        }
        const tailResults = await tailSession.run(tailFeeds);
        const tailOutput = Array.from(Object.values(tailResults)[0].data as Float32Array);

        // 4. Compare outputs
        expect(tailOutput.length).toBe(origOutput.length);
        for (let i = 0; i < origOutput.length; i++) {
            expect(tailOutput[i]).toBeCloseTo(origOutput[i], 4);
        }
    });

    it("partitions graph using explicit <nodeId>", () => {
        const baseName = path.resolve(PROJECT_ROOT, "temp_part_id");
        const headOnnx = `${baseName}_head.onnx`;
        const tailOnnx = `${baseName}_tail.onnx`;
        generatedFiles.push(headOnnx, tailOnnx);

        // Node ID "0" is the first operator in initGraph topological indexing
        runCli(`"${testModelAbs}" --partition 0 -o "${baseName}.json" -vz 0 -v 0`);

        expect(fs.existsSync(headOnnx)).toBe(true);
        expect(fs.existsSync(tailOnnx)).toBe(true);
    });
});
