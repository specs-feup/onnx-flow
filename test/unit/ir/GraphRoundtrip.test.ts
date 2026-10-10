import { describe, it, expect } from "vitest";
import { createGraph } from "../../../src/initGraph.js";
import { convertFlowGraphToOnnxJson } from "../../../src/flow2json.js";
import { DataType, type RawOnnxModel } from "../../../src/Onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("Graph Ingestion and Export Roundtrip", () => {
    initializeSchemaRegistry();

    it("converts a RawOnnxModel into OnnxGraph and serializes it back accurately", () => {
        const rawModel: RawOnnxModel = {
            irVersion: 9,
            opsetImport: [{ domain: "", version: 19 }],
            graph: {
                name: "test_pipeline",
                input: [
                    {
                        name: "A",
                        type: {
                            tensorType: {
                                elemType: DataType.FLOAT,
                                shape: { dim: [{ dimValue: 2 }, { dimValue: 2 }] },
                            },
                        },
                    },
                ],
                output: [
                    {
                        name: "B",
                        type: {
                            tensorType: {
                                elemType: DataType.FLOAT,
                                shape: { dim: [{ dimValue: 2 }, { dimValue: 2 }] },
                            },
                        },
                    },
                ],
                node: [
                    {
                        opType: "Relu",
                        input: ["A"],
                        output: ["B"],
                    },
                ],
            },
        };

        // 1. Ingestão (initGraph.ts)
        const graph = createGraph(rawModel);
        expect(graph.hasNode("A")).toBe(true);
        expect(graph.hasNode("B")).toBe(true);
        expect(graph.getOperationNodes().length).toBe(1);

        // 2. Exportação (flow2json.ts)
        const exportedJson = convertFlowGraphToOnnxJson(graph, "exported_model");
        expect(exportedJson.graph?.node?.length).toBe(1);
        expect(exportedJson.graph?.node?.[0].opType).toBe("Relu");
        expect(exportedJson.graph?.input?.length).toBe(1);
        expect(exportedJson.graph?.output?.length).toBe(1);
    });
});
