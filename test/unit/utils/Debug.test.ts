import { describe, it, expect, vi } from "vitest";
import fs from "fs";
import path from "path";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { scalarFloat } from "../../../src/Onnx/utils/TensorData.js";
import { dbg, dbgTensor, safeWriteJson } from "../../../src/Onnx/utils/Debug.js";

describe("Debug Utilities", () => {
    it("logs loop debug messages and inspects tensor nodes", () => {
        const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});

        dbg("test debug message", 123);
        expect(logSpy).toHaveBeenCalledWith("[loop-debug]", "test debug message", 123);

        dbgTensor("null_check", null);
        dbgTensor("undefined_check", undefined);

        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const tNode = graph
            .addNode("t")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        dbgTensor("tensor_check", tNode);
        expect(logSpy).toHaveBeenCalledWith(
            "[loop-debug]",
            "tensor_check",
            expect.objectContaining({ id: "t", kind: "input" }),
        );

        const cNode = graph
            .addNode("c")
            .init(new ConstantNode.Builder(scalarFloat(4.5)))
            .as(ConstantNode);
        dbgTensor("constant_check", cNode);
        expect(logSpy).toHaveBeenCalledWith(
            "[loop-debug]",
            "constant_check",
            expect.objectContaining({ id: "c", kind: "constant" }),
        );

        logSpy.mockRestore();
    });

    it("serializes nested objects safely to file with safeWriteJson", () => {
        const tmpPath = path.resolve("temp_debug_test.json");
        try {
            const payload = {
                str: "hello",
                num: 42,
                nan: NaN,
                inf: Infinity,
                bool: true,
                nil: null,
                undef: undefined,
                big: 999999999999999n,
                arr: [1, "two", false],
                nested: { key: "value" },
            };

            safeWriteJson(tmpPath, payload);
            expect(fs.existsSync(tmpPath)).toBe(true);

            const readBack = JSON.parse(fs.readFileSync(tmpPath, "utf-8"));
            expect(readBack.str).toBe("hello");
            expect(readBack.num).toBe(42);
            expect(readBack.nan).toBeNull();
            expect(readBack.inf).toBeNull();
            expect(readBack.bool).toBe(true);
            expect(readBack.nil).toBeNull();
            expect(readBack.undef).toBeNull();
            expect(readBack.big).toBe(999999999999999);
            expect(readBack.arr).toEqual([1, "two", false]);
            expect(readBack.nested).toEqual({ key: "value" });
        } finally {
            if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
        }
    });

    it("throws an error upon encountering cyclic structures in safeWriteJson", () => {
        const tmpPath = path.resolve("temp_debug_cyclic.json");
        const cyclicObj: Record<string, unknown> = { a: 1 };
        cyclicObj["self"] = cyclicObj;

        try {
            expect(() => safeWriteJson(tmpPath, cyclicObj)).toThrow(
                /safeWriteJson: cyclic reference/,
            );
        } finally {
            if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
        }
    });
});
