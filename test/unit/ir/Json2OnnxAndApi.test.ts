import { describe, it, expect, beforeAll } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { coerceNumericFields, json2onnx } from "../../../src/io/JsonToOnnx.js";
import { onnx2json } from "../../../src/io/OnnxToJson.js";
import { applyRecipes, transformModel } from "../../../src/api.js";
import { createGraph } from "../../../src/io/OnnxGraphLoader.js";
import { convertFlowGraphToOnnxJson } from "../../../src/io/OnnxGraphExporter.js";
import { Recipes } from "../../../src/onnx/transformation/recipes.js";
import { DataType } from "../../../src/onnx/OnnxTypes.js";
import { GraphBuilder } from "../../../src/onnx/GraphBuilder.js";
import OnnxGraph from "../../../src/onnx/OnnxGraph.js";
import TensorNode from "../../../src/onnx/TensorNode.js";
import Graph from "@specs-feup/flow/graph/Graph";
import * as ort from "onnxruntime-web";
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

    it("roundtrips IR 14 / Opset 28 binary ONNX and executes adaptive Opset 23 / IR 11 model in ORT", async () => {
        const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-v28-"));
        try {
            // 1. Build RMSNormalization model (Opset 23 -> auto-resolved IR 11) and run in onnxruntime-web
            const graph23 = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const builder23 = new GraphBuilder(graph23);
            const xIn = graph23
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, [1, 4], "input"))
                .as(TensorNode);
            const scaleIn = builder23.createConstant("scale", {
                dataType: DataType.FLOAT,
                dims: [4],
                floatData: [1, 1, 1, 1],
            });
            const [rmsOut] = builder23.createOp("RMSNormalization", [xIn, scaleIn], {
                axis: -1,
                epsilon: 1e-5,
            });
            if (rmsOut.is(TensorNode)) {
                rmsOut.setType("output");
            }

            const json23 = convertFlowGraphToOnnxJson(graph23, "rms_graph");
            expect(json23.irVersion).toBe(11);
            expect(json23.opsetImport?.[0].version).toBe(23);

            const json23Path = path.join(tmpDir, "rms23.json");
            const onnx23Path = path.join(tmpDir, "rms23.onnx");
            fs.writeFileSync(json23Path, JSON.stringify(json23));
            await json2onnx(json23Path, onnx23Path);

            const session = await ort.InferenceSession.create(onnx23Path);
            const feeds = {
                X: new ort.Tensor("float32", new Float32Array([3, 4, 0, 0]), [1, 4]),
            };
            const outputs = await session.run(feeds);
            const outData = outputs[rmsOut.id].data as Float32Array;
            // rms([3,4,0,0]) = sqrt((9+16)/4) = sqrt(6.25) = 2.5 -> [1.2, 1.6, 0, 0]
            expect(outData[0]).toBeCloseTo(1.2, 4);
            expect(outData[1]).toBeCloseTo(1.6, 4);

            // 2. Explicit IR 14 / Opset 28 binary roundtrip
            const json28 = convertFlowGraphToOnnxJson(graph23, "rms_v28", undefined, {
                targetOpset: 28,
                irVersion: 14,
            });
            expect(json28.irVersion).toBe(14);
            expect(json28.opsetImport?.[0].version).toBe(28);

            const json28Path = path.join(tmpDir, "rms28.json");
            const onnx28Path = path.join(tmpDir, "rms28.onnx");
            fs.writeFileSync(json28Path, JSON.stringify(json28));
            await json2onnx(json28Path, onnx28Path);

            const decoded28 = await onnx2json(onnx28Path);
            expect(Number(decoded28.irVersion)).toBe(14);
            expect(Number(decoded28.opsetImport?.[0].version)).toBe(28);
            const reimported = createGraph(decoded28);
            expect(reimported.getIrVersion()).toBe(14);
            expect(reimported.getSourceOpset()).toBe(28);
        } finally {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    });
});
