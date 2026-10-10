import { describe, it, expect, beforeAll, afterEach } from "vitest";
import fs from "fs";
import path from "path";
import { runCli } from "../../../src/index.js";
import { onnx2json } from "../../../src/onnx2json.js";
import { ensureProtoExists, PROJECT_ROOT, resolveModelPath } from "../../compatibility_test.js";

describe("CLI in-process execution & partition verification", () => {
    const testModel = resolveModelPath("examples/onnx/add_chain_standard.onnx");
    const testJsonModel = resolveModelPath("examples/onnx/vector_add_standard.onnx");
    const cleanupFiles: string[] = [];

    beforeAll(() => {
        ensureProtoExists();
    });

    afterEach(() => {
        for (const file of cleanupFiles) {
            if (fs.existsSync(file)) {
                try {
                    fs.unlinkSync(file);
                } catch {
                    /* cleanup */
                }
            }
        }
        cleanupFiles.length = 0;
    });

    it("verifies partition files do not clobber each other and suppresses unpartitioned output", async () => {
        const outBase = path.resolve(PROJECT_ROOT, "temp_unit_part_test");
        const outJson = `${outBase}.json`;
        const headOnnx = `${outBase}_head.onnx`;
        const tailOnnx = `${outBase}_tail.onnx`;
        const headJson = `${outBase}_head.json`;
        const tailJson = `${outBase}_tail.json`;
        const headOnnxJson = `${outBase}_head.onnx.json`;

        cleanupFiles.push(outJson, headOnnx, tailOnnx, headJson, tailJson, headOnnxJson);

        await runCli([
            testModel,
            "--partition",
            "Add",
            "1",
            "-o",
            outJson,
            "--format",
            "json",
            "-vz",
            "0",
            "-v",
            "0",
        ]);

        expect(fs.existsSync(headOnnx)).toBe(true);
        expect(fs.existsSync(tailOnnx)).toBe(true);
        expect(fs.existsSync(headJson)).toBe(true);
        expect(fs.existsSync(outJson)).toBe(false);

        const cyData = JSON.parse(fs.readFileSync(headJson, "utf-8"));
        expect(cyData).toHaveProperty("elements");

        if (fs.existsSync(headOnnxJson)) {
            const onnxData = JSON.parse(fs.readFileSync(headOnnxJson, "utf-8"));
            expect(onnxData).toHaveProperty("graph");
        }
    });

    it("verifies single-argument partition validates node existence", async () => {
        await expect(
            runCli([testModel, "--partition", "non_existent_node_id", "-vz", "0", "-v", "0"]),
        ).rejects.toThrow(/Could not find node with id 'non_existent_node_id'/);
    });

    it("partitions correctly with valid single-argument node ID in DOT format", async () => {
        const outBase = path.resolve(PROJECT_ROOT, "temp_unit_part_dot");
        const headDot = `${outBase}_head.dot`;
        const tailDot = `${outBase}_tail.dot`;
        const headOnnx = `${outBase}_head.onnx`;
        const tailOnnx = `${outBase}_tail.onnx`;

        cleanupFiles.push(headDot, tailDot, headOnnx, tailOnnx);

        await runCli([
            testModel,
            "--partition",
            "0",
            "-o",
            `${outBase}.dot`,
            "--format",
            "dot",
            "-vz",
            "0",
            "-v",
            "0",
        ]);

        expect(fs.existsSync(headDot)).toBe(true);
        expect(fs.existsSync(tailDot)).toBe(true);

        const dotContent = fs.readFileSync(headDot, "utf-8");
        expect(dotContent).toContain("digraph");
    });

    it("executes default flow with verbose logging and without reconversion", async () => {
        const outBase = path.resolve(PROJECT_ROOT, "temp_unit_verbose");
        const outJson = `${outBase}.json`;
        cleanupFiles.push(outJson);

        await runCli([
            testModel,
            "-o",
            outJson,
            "--format",
            "json",
            "-v",
            "2",
            "-vz",
            "0",
            "--noReconversion",
        ]);

        expect(fs.existsSync(outJson)).toBe(true);
    });

    it("executes with DOT formatter and online visualization link option", async () => {
        const outBase = path.resolve(PROJECT_ROOT, "temp_unit_dot_link");
        const outDot = `${outBase}.dot`;
        cleanupFiles.push(outDot);

        await runCli([
            testModel,
            "-o",
            outDot,
            "--format",
            "dot",
            "--formatter",
            "cgra",
            "--noLowLevel",
            "-v",
            "1",
            "-vz",
            "1",
            "--noReconversion",
        ]);

        expect(fs.existsSync(outDot)).toBe(true);
    });

    it("executes with --noLowLevel flag and partition verbose branch", async () => {
        await runCli([
            testModel,
            "--partition",
            "Add",
            "1",
            "--format",
            "dot",
            "-v",
            "2",
            "-vz",
            "0",
            "--noReconversion",
        ]);
    });

    it("handles pre-parsed JSON input file as input path", async () => {
        const outBase = path.resolve(PROJECT_ROOT, "temp_unit_raw_json");
        const rawJsonPath = path.resolve(PROJECT_ROOT, "temp_vector_add_input.json");
        const outJson = `${outBase}_out.json`;
        cleanupFiles.push(rawJsonPath, outJson);

        // Ingest into a raw ONNX JSON structure
        const rawModel = await onnx2json(testJsonModel);
        fs.writeFileSync(rawJsonPath, JSON.stringify(rawModel, null, 2));

        // Pass .json input path directly to CLI
        await runCli([rawJsonPath, "-o", outJson, "-vz", "0", "-v", "0", "--noReconversion"]);

        expect(fs.existsSync(outJson)).toBe(true);
    });
});
