import { describe, it, expect, beforeAll, afterAll } from "vitest";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { coerceNumericFields, json2onnx } from "../../../src/json2onnx.js";
import { onnx2json } from "../../../src/onnx2json.js";
import { applyRecipes, transformModel } from "../../../src/api.js";
import { createGraph } from "../../../src/initGraph.js";
import { convertFlowGraphToOnnxJson } from "../../../src/flow2json.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe("json2onnx, onnx2json & Public API", () => {
    let createdProto = false;
    const targetProto = path.resolve(__dirname, "../../../../out/src/Onnx/onnx.proto");

    beforeAll(() => {
        if (!fs.existsSync(targetProto)) {
            const sourceProto = path.resolve(__dirname, "../../../src/Onnx/onnx.proto");
            if (fs.existsSync(sourceProto)) {
                fs.mkdirSync(path.dirname(targetProto), { recursive: true });
                fs.copyFileSync(sourceProto, targetProto);
                createdProto = true;
            }
        }
    });

    afterAll(() => {
        if (createdProto && fs.existsSync(targetProto)) {
            try {
                fs.unlinkSync(targetProto);
            } catch {
                /* ignore */
            }
        }
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
        const sampleModelPath = path.resolve("examples/onnx/vector_add_standard.onnx");
        if (!fs.existsSync(sampleModelPath)) return;

        // 1. onnx2json
        const rawJson = await onnx2json(sampleModelPath);
        expect(rawJson.graph?.node?.length).toBeGreaterThan(0);

        // 2. applyRecipes via in-memory graph
        const graph = createGraph(rawJson);
        const count = applyRecipes(graph, [new Recipes.canonicalization.Relu()]);
        expect(typeof count).toBe("number");

        // 3. json2onnx roundtrip with serialized Buffer for fixBuffers coverage
        const tmpJson = path.resolve("temp_vector_add.json");
        const tmpOnnx = path.resolve("temp_vector_add_roundtrip.onnx");
        try {
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
            if (fs.existsSync(tmpJson)) fs.unlinkSync(tmpJson);
            if (fs.existsSync(tmpOnnx)) fs.unlinkSync(tmpOnnx);
        }
    });

    it("transforms an onnx file directly via transformModel in api.ts", async () => {
        const sampleModelPath = path.resolve("examples/onnx/relu_standard.onnx");
        if (!fs.existsSync(sampleModelPath)) return;

        const outOnnx = path.resolve("temp_relu_transformed.onnx");
        try {
            const count = await transformModel(sampleModelPath, outOnnx, [
                new Recipes.canonicalization.Relu(),
            ]);
            expect(count).toBeGreaterThan(0);
            expect(fs.existsSync(outOnnx)).toBe(true);
        } finally {
            if (fs.existsSync(outOnnx)) fs.unlinkSync(outOnnx);
        }
    });
});
