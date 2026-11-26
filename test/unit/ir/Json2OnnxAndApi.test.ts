import { describe, it, expect, beforeAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { coerceNumericFields, json2onnx } from "../../../src/json2onnx.js";
import { onnx2json } from "../../../src/onnx2json.js";
import { applyRecipes, transformModel } from "../../../src/api.js";
import { createGraph } from "../../../src/initGraph.js";
import { convertFlowGraphToOnnxJson } from "../../../src/flow2json.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { ensureProtoExists } from "../../setup.js";
import { ensureModelExists } from "../../helpers/modelFactory.js";

describe("json2onnx, onnx2json & Public API", () => {
    beforeAll(() => {
        ensureProtoExists();
    });

    it("coerces numeric fields and normalizes string-encoded arrays in coerceNumericFields", () => {
        const input = {
            ints: ["1", "2", 3],
            floats: ["1.5", 2.5],
            i: "42",
            f: "3.14",
            int64Data: ["10", 20n],
            nested: {
                axis: "0",
            },
        };

        const coerced = coerceNumericFields(input) as typeof input;
        expect(coerced.ints).toEqual([1, 2, 3]);
        expect(coerced.floats).toEqual([1.5, 2.5]);
        expect(coerced.i).toBe(42);
        expect(coerced.f).toBeCloseTo(3.14);
        expect(coerced.int64Data).toEqual([10, 20]);
        expect(coerced.nested.axis).toBe(0);
    });

    it("rejects invalid file extensions in json2onnx and onnx2json", async () => {
        await expect(json2onnx("model.txt", "out.onnx")).rejects.toThrow(
            "The specified file is not a JSON file. Please provide a valid .json file.",
        );
        await expect(onnx2json("model.txt")).rejects.toBe(
            "The specified file is not an ONNX file. Please provide a valid .onnx file.",
        );
    });

    it("executes end-to-end onnx2json, json2onnx, and applyRecipes", async () => {
        const sampleModelPath = await ensureModelExists("examples/onnx/vector_add_standard.onnx");
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-api-unit-"));

        try {
            const rawJson = await onnx2json(sampleModelPath);
            expect(rawJson.graph?.node?.length).toBeGreaterThan(0);

            const graph = createGraph(rawJson);
            const count = applyRecipes(graph, [new Recipes.canonicalization.Relu()]);
            expect(typeof count).toBe("number");

            const tmpJson = path.join(tmpDir, "vector_add.json");
            const tmpOnnx = path.join(tmpDir, "vector_add_roundtrip.onnx");
            const onnxJson = convertFlowGraphToOnnxJson(graph);
            const testJson = {
                ...onnxJson,
                graph: {
                    ...onnxJson.graph,
                    initializer: [
                        ...(onnxJson.graph?.initializer ?? []),
                        {
                            name: "raw_buf_test",
                            dataType: DataType.FLOAT,
                            dims: [2],
                            rawData: { type: "Buffer", data: [0, 0, 128, 63, 0, 0, 0, 64] },
                        },
                    ],
                },
            };
            fs.writeFileSync(tmpJson, JSON.stringify(testJson, null, 2));
            await json2onnx(tmpJson, tmpOnnx);
            expect(fs.existsSync(tmpOnnx)).toBe(true);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });

    it("transforms an onnx file directly via transformModel in api.ts", async () => {
        const sampleModelPath = await ensureModelExists("examples/onnx/relu_standard.onnx");
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-api-xform-"));
        const outOnnx = path.join(tmpDir, "relu_transformed.onnx");
        try {
            const count = await transformModel(sampleModelPath, outOnnx, [
                new Recipes.canonicalization.Relu(),
            ]);
            expect(count).toBeGreaterThan(0);
            expect(fs.existsSync(outOnnx)).toBe(true);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });
});
