import { describe, it, expect } from "vitest";
import {
    toNum,
    toStaticShape,
    prodSafe,
    computeStrides,
    normalizeAxis,
    broadcastTwoShapes,
    broadcastShapes,
    inferPoolDim,
    inferConvDim,
    shapesEqual,
} from "../../../src/Onnx/utils/ShapeMath.js";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";

describe("ShapeMath Utilities", () => {
    it("converts dimension values and shapes to numbers and static forms", () => {
        expect(toNum(5)).toBe(5);
        expect(toNum("12")).toBe(12);
        expect(toNum("batch")).toBeUndefined();

        expect(toStaticShape([1, "batch", 64])).toEqual([1, -1, 64]);
        expect(prodSafe([2, 3, 4])).toBe(24);
        expect(prodSafe([2, -1, 4])).toBe(8); // Non-positive coerced to 1
    });

    it("computes row-major contiguous strides correctly", () => {
        expect(computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
        expect(computeStrides([10])).toEqual([1]);
        expect(computeStrides([])).toEqual([]);
    });

    it("normalizes negative axes safely within rank", () => {
        expect(normalizeAxis(-1, 4)).toBe(3);
        expect(normalizeAxis(-2, 4)).toBe(2);
        expect(normalizeAxis(1, 4)).toBe(1);
        expect(normalizeAxis(0, 0)).toBe(0);
    });

    it("broadcasts two shapes according to ONNX/NumPy semantics", () => {
        expect(broadcastTwoShapes([2, 3], [3])).toEqual([2, 3]);
        expect(broadcastTwoShapes([2, 1, 4], [3, 4])).toEqual([2, 3, 4]);
        expect(broadcastTwoShapes([1, 5], [5])).toEqual([1, 5]);
        expect(broadcastShapes([2, 1, 3], [1, 4, 1], [3])).toEqual([2, 4, 3]);
    });

    it("calculates 1D spatial pooling and convolution output dimensions", () => {
        // inDim=28, kernel=3, stride=1, pad=0, dilation=1 -> 26
        expect(inferConvDim(28, 3, 1, 0, 0, 1)).toBe(26);

        // inDim=28, kernel=3, stride=2, pad=1 on each side -> floor((28 + 2 - 3)/2 + 1) = 14
        expect(inferConvDim(28, 3, 2, 1, 1, 1)).toBe(14);

        // Dilated kernel: kernel=3, dil=2 -> effectiveK = 2*(2) + 1 = 5
        expect(inferConvDim(28, 3, 1, 0, 0, 2)).toBe(24);

        // Ceil mode support in pool dim
        expect(inferPoolDim(7, 2, 2, 0, 0, 1, 1)).toBe(4);
    });

    it("compares shapes between concrete tensor nodes", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const t1 = graph
            .addNode("t1")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "intermediate"))
            .as(TensorNode);
        const t2 = graph
            .addNode("t2")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "intermediate"))
            .as(TensorNode);
        const t3 = graph
            .addNode("t3")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 8], "intermediate"))
            .as(TensorNode);

        expect(shapesEqual(t1, t2)).toBe(true);
        expect(shapesEqual(t1, t3)).toBe(false);
    });
});
