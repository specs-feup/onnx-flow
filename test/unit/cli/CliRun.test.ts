import { describe, it, expect, beforeAll, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { runCli } from "../../../src/index.js";
import { onnx2json } from "../../../src/io/OnnxToJson.js";
import { ensureProtoExists } from "../../helpers/ortHarness.js";
import { ensureModelExists } from "../../helpers/modelFactory.js";

describe("CLI in-process execution & partition verification", () => {
    let testModel = "";
    let testJsonModel = "";
    let tmpDir = "";

    beforeAll(async () => {
        ensureProtoExists();
        testModel = await ensureModelExists("examples/onnx/add_chain_standard.onnx");
        testJsonModel = await ensureModelExists("examples/onnx/vector_add_standard.onnx");
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-cli-unit-"));
    });

    afterEach(() => {
        if (tmpDir && fs.existsSync(tmpDir)) {
            for (const entry of fs.readdirSync(tmpDir)) {
                fs.rmSync(path.join(tmpDir, entry), { recursive: true, force: true });
            }
        }
    });

    function stageModel(srcPath: string = testModel): string {
        const dst = path.join(tmpDir, `${Date.now()}_${path.basename(srcPath)}`);
        fs.copyFileSync(srcPath, dst);
        return dst;
    }

    it("verifies partition files do not clobber each other and suppresses unpartitioned output", async () => {
        const outBase = path.join(tmpDir, "temp_unit_part_test");
        const outJson = `${outBase}.json`;
        const headOnnx = `${outBase}_head.onnx`;
        const tailOnnx = `${outBase}_tail.onnx`;
        const headJson = `${outBase}_head.json`;
        const headOnnxJson = `${outBase}_head.onnx.json`;

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
        const outBase = path.join(tmpDir, "temp_unit_part_dot");
        const headDot = `${outBase}_head.dot`;
        const tailDot = `${outBase}_tail.dot`;

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
        const outJson = path.join(tmpDir, "temp_unit_verbose.json");

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
        const outDot = path.join(tmpDir, "temp_unit_dot_link.dot");

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
        const outBase = path.join(tmpDir, "temp_unit_part_verbose");
        await runCli([
            testModel,
            "--partition",
            "Add",
            "1",
            "-o",
            `${outBase}.dot`,
            "--format",
            "dot",
            "-v",
            "2",
            "-vz",
            "0",
            "--noReconversion",
        ]);
        expect(fs.existsSync(`${outBase}_head.dot`)).toBe(true);
    });

    it("handles pre-parsed JSON input file as input path", async () => {
        const rawJsonPath = path.join(tmpDir, "temp_vector_add_input.json");
        const outJson = path.join(tmpDir, "temp_unit_raw_json_out.json");

        const rawModel = await onnx2json(testJsonModel);
        fs.writeFileSync(rawJsonPath, JSON.stringify(rawModel, null, 2));

        await runCli([rawJsonPath, "-o", outJson, "-vz", "0", "-v", "0", "--noReconversion"]);

        expect(fs.existsSync(outJson)).toBe(true);
    });

    it("supports --indexingMode pointer and radix flags via CLI", async () => {
        const outPointer = path.join(tmpDir, "temp_unit_im_pointer.json");
        const outRadix = path.join(tmpDir, "temp_unit_im_radix.json");

        await runCli([
            stageModel(),
            "--indexingMode",
            "pointer",
            "-o",
            outPointer,
            "-vz",
            "0",
            "-v",
            "0",
            "--noReconversion",
        ]);
        await runCli([
            stageModel(),
            "-im",
            "radix",
            "-o",
            outRadix,
            "-vz",
            "0",
            "-v",
            "0",
            "--noReconversion",
        ]);

        expect(fs.existsSync(outPointer)).toBe(true);
        expect(fs.existsSync(outRadix)).toBe(true);
    });
});
