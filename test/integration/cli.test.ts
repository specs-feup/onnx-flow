import { describe, it, expect, beforeAll, afterEach } from "vitest";
import { execSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import * as ort from "onnxruntime-web";
import { ensureProtoExists, PROJECT_ROOT } from "../helpers/OrtHarness.js";
import { ensureModelExists } from "../helpers/ModelFactory.js";

describe("CLI & Partitioning End-to-End Tests", () => {
    const cliEntry = path.resolve(PROJECT_ROOT, "out/src/index.js");
    const testModelRel = "examples/onnx/add_chain_standard.onnx";
    let testModelAbs = "";
    let tmpDir = "";

    beforeAll(async () => {
        ensureProtoExists();
        expect(fs.existsSync(cliEntry), "Build out/src/index.js before running CLI tests").toBe(
            true,
        );
        testModelAbs = await ensureModelExists(testModelRel);
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-cli-e2e-"));
    });

    afterEach(() => {
        if (tmpDir && fs.existsSync(tmpDir)) {
            for (const entry of fs.readdirSync(tmpDir)) {
                fs.rmSync(path.join(tmpDir, entry), { recursive: true, force: true });
            }
        }
    });

    function stageInputModel(srcPath: string = testModelAbs): string {
        const dst = path.join(tmpDir, `${Date.now()}_${path.basename(srcPath)}`);
        fs.copyFileSync(srcPath, dst);
        return dst;
    }

    function runCli(args: string): string {
        return execSync(`node "${cliEntry}" ${args}`, {
            cwd: PROJECT_ROOT,
            encoding: "utf-8",
            stdio: ["pipe", "pipe", "pipe"],
        });
    }

    it("runs default reconversion and outputs artifacts", () => {
        const stagedInput = stageInputModel();
        const outDot = path.join(tmpDir, "cli_standard.dot");

        const output = runCli(`"${stagedInput}" --format dot -vz 0 -v 1 -o "${outDot}"`);
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
        const baseName = path.join(tmpDir, "part_op");
        const headOnnx = `${baseName}_head.onnx`;
        const tailOnnx = `${baseName}_tail.onnx`;

        const cliOutput = runCli(
            `"${testModelAbs}" --partition Add 1 -o "${baseName}.json" -vz 0 -v 1`,
        );

        expect(cliOutput).toContain("Partitioning graph at node:");
        expect(cliOutput).toContain("Created ONNX partition:");
        expect(fs.existsSync(headOnnx)).toBe(true);
        expect(fs.existsSync(tailOnnx)).toBe(true);

        const feeds = {
            A: new ort.Tensor("float32", new Float32Array([1, 2, 3, 4]), [4]),
            B: new ort.Tensor("float32", new Float32Array([5, 6, 7, 8]), [4]),
            C: new ort.Tensor("float32", new Float32Array([1, 1, 1, 1]), [4]),
            D: new ort.Tensor("float32", new Float32Array([2, 2, 2, 2]), [4]),
        };

        const origSession = await ort.InferenceSession.create(testModelAbs);
        const origResults = await origSession.run(feeds);
        const origOutput = Array.from(Object.values(origResults)[0].data as Float32Array);

        const headSession = await ort.InferenceSession.create(headOnnx);
        const headFeeds: Record<string, ort.Tensor> = {};
        for (const inputName of headSession.inputNames) {
            if (inputName in feeds) headFeeds[inputName] = feeds[inputName as keyof typeof feeds];
        }
        const headResults = await headSession.run(headFeeds);

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

        expect(tailOutput.length).toBe(origOutput.length);
        for (let i = 0; i < origOutput.length; i++) {
            expect(tailOutput[i]).toBeCloseTo(origOutput[i], 4);
        }
    });

    it("partitions graph using explicit <nodeId>", () => {
        const baseName = path.join(tmpDir, "part_id");
        const headOnnx = `${baseName}_head.onnx`;
        const tailOnnx = `${baseName}_tail.onnx`;

        runCli(`"${testModelAbs}" --partition 0 -o "${baseName}.json" -vz 0 -v 0`);

        expect(fs.existsSync(headOnnx)).toBe(true);
        expect(fs.existsSync(tailOnnx)).toBe(true);
    });

    it("supports transformation flags (--no-canonicalize, --no-loop-lowering, --optimize, --no-fuse, --no-coalesce, --tileSize, --decomposeForCgra)", async () => {
        const matmulPath = await ensureModelExists("examples/onnx/matmul_standard.onnx");

        const stagedNoCanon = stageInputModel();
        runCli(
            `"${stagedNoCanon}" --no-canonicalize --no-fuse --no-coalesce --optimize -im radix -vz 0 -v 0`,
        );
        expect(fs.existsSync(stagedNoCanon.replace(/\.onnx$/, "_reconverted.onnx"))).toBe(true);

        const stagedNoLoop = stageInputModel();
        runCli(`"${stagedNoLoop}" --no-loop-lowering -vz 0 -v 0`);
        expect(fs.existsSync(stagedNoLoop.replace(/\.onnx$/, "_reconverted.onnx"))).toBe(true);

        const stagedTiled = stageInputModel();
        runCli(`"${stagedTiled}" --tileSize 2 -vz 0 -v 0`);
        const tiledReconv = stagedTiled.replace(/\.onnx$/, "_reconverted.onnx");
        expect(fs.existsSync(tiledReconv)).toBe(true);

        // Verify tiled reconverted model executes accurately in ORT
        const feeds = {
            A: new ort.Tensor("float32", new Float32Array([1, 2, 3, 4]), [4]),
            B: new ort.Tensor("float32", new Float32Array([5, 6, 7, 8]), [4]),
            C: new ort.Tensor("float32", new Float32Array([1, 1, 1, 1]), [4]),
            D: new ort.Tensor("float32", new Float32Array([2, 2, 2, 2]), [4]),
        };
        const sess = await ort.InferenceSession.create(tiledReconv);
        const res = await sess.run(feeds);
        expect(Array.from(Object.values(res)[0].data as Float32Array)).toEqual([9, 11, 13, 15]);

        const stagedCgra = stageInputModel(matmulPath);
        const cgraDot = path.join(tmpDir, "cgra.dot");
        runCli(
            `"${stagedCgra}" --decomposeForCgra --formatter cgra --format dot -o "${cgraDot}" -vz 0 -v 0`,
        );
        expect(fs.existsSync(cgraDot)).toBe(true);
        expect(fs.existsSync(stagedCgra.replace(/\.onnx$/, "_reconverted.onnx"))).toBe(true);
    });
});
