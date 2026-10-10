import { describe, it, expect, beforeAll } from "vitest";
import path from "path";
import {
    parseOnnxFile,
    loadGraph,
    renderDotToSVG,
    generateGraphvizOnlineLink,
    generateGraphCode,
} from "../../../src/index.js";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import { ensureProtoExists, PROJECT_ROOT } from "../../compatibility_test.js";

describe("index.ts Library Functionality", () => {
    beforeAll(() => {
        ensureProtoExists();
    });

    const modelPath = path.resolve(PROJECT_ROOT, "examples/onnx/vector_add_standard.onnx");

    it("parses ONNX file into RawOnnxModel", async () => {
        const raw = await parseOnnxFile(modelPath);
        expect(raw).toBeDefined();
        expect(raw.graph?.node?.length).toBeGreaterThan(0);
    });

    it("loads graph as DOT string and OnnxGraph object", async () => {
        const raw = await parseOnnxFile(modelPath);

        // Load as DOT string
        const dotString = loadGraph(raw, false, false, true);
        expect(typeof dotString).toBe("string");
        expect(dotString).toContain("digraph");

        // Load as OnnxGraph instance
        const graphObj = loadGraph(raw, false, false, false);
        expect(graphObj).toBeInstanceOf(OnnxGraph.Class);
    });

    it("renders DOT to SVG and generates Graphviz online links", async () => {
        const dot = "digraph G { A -> B; }";
        const svg = await renderDotToSVG(dot);
        expect(svg).toContain("<svg");

        const link = generateGraphvizOnlineLink(dot);
        expect(link).toContain("https://dreampuf.github.io/GraphvizOnline/#");
        expect(link).toContain(encodeURIComponent(dot));
    });

    it("stubs code generation predictably", async () => {
        const raw = await parseOnnxFile(modelPath);
        const graph = loadGraph(raw, false, false, false) as OnnxGraph.Class;
        expect(generateGraphCode(graph)).toBe("TODO");
    });
});
