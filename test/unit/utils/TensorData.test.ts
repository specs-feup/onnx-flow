import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import {
    bool,
    int32Vec,
    int64Vec,
    floatVec,
    scalarFloat,
    scalarInt64,
    makeTensorProto,
    zeroTensor,
    totalSizeFromDims,
    isInt64Type,
    decodeIntegerVectorFromTensorProto,
    readScalarFromTensorNode,
    readVectorFromTensorNode,
    readTensorData,
} from "../../../src/Onnx/utils/TensorData.js";

describe("TensorData Utilities", () => {
    it("creates scalar and vector tensor protos with expected fields", () => {
        const b = bool(true);
        expect(b.dataType).toBe(DataType.BOOL);
        expect(b.dims).toEqual([]);
        expect(b.int32Data).toEqual([1]);

        const i32 = int32Vec([10, 20]);
        expect(i32.dataType).toBe(DataType.INT32);
        expect(i32.dims).toEqual([2]);
        expect(i32.int32Data).toEqual([10, 20]);

        const i64 = scalarInt64(42);
        expect(i64.dataType).toBe(DataType.INT64);
        expect(i64.int64Data).toEqual([42]);

        const f32 = floatVec([1.5, -2.5]);
        expect(f32.dataType).toBe(DataType.FLOAT);
        expect(f32.dims).toEqual([2]);
        expect(f32.floatData).toEqual([1.5, -2.5]);
    });

    it("handles makeTensorProto fallback for raw buffers", () => {
        const rawProto = makeTensorProto(DataType.UNDEFINED, [2], [1.0, 2.0]);
        expect(rawProto.rawData).toBeDefined();
        expect(rawProto.rawData?.type).toBe("Buffer");
        expect(Array.isArray(rawProto.rawData?.data)).toBe(true);
    });

    it("allocates correct zero-filled tensors", () => {
        const zeroFloat = zeroTensor(DataType.FLOAT, [2, 3]);
        expect(zeroFloat.dims).toEqual([2, 3]);
        expect(zeroFloat.floatData).toEqual([0, 0, 0, 0, 0, 0]);

        const zeroInt = zeroTensor(DataType.INT64, [4]);
        expect(zeroInt.int64Data).toEqual([0, 0, 0, 0]);
    });

    it("calculates total sizes and identifies int64 types", () => {
        expect(totalSizeFromDims(1, [2, 3, 4])).toBe(24);
        expect(totalSizeFromDims(10, [])).toBe(10);
        expect(isInt64Type(DataType.INT64)).toBe(true);
        expect(isInt64Type("INT64")).toBe(true);
        expect(isInt64Type(DataType.FLOAT)).toBe(false);
    });

    it("decodes integer vectors from int32, int64, and raw byte buffers", () => {
        const p64 = int64Vec([100, 200]);
        expect(decodeIntegerVectorFromTensorProto(p64)).toEqual([100, 200]);

        // Raw buffer test (two little-endian 32-bit ints: 5 and 9)
        const buf = Buffer.alloc(8);
        buf.writeInt32LE(5, 0);
        buf.writeInt32LE(9, 4);
        const rawProto = {
            dataType: DataType.INT32,
            dims: [2],
            rawData: { type: "Buffer", data: Array.from(buf) },
        };
        expect(decodeIntegerVectorFromTensorProto(rawProto)).toEqual([5, 9]);
    });

    it("reads numeric scalars and vectors directly from ConstantNode instances", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const fNode = graph
            .addNode("c_float")
            .init(new ConstantNode.Builder(scalarFloat(3.14)))
            .as(ConstantNode);
        expect(readScalarFromTensorNode(fNode)).toBeCloseTo(3.14);

        const vecNode = graph
            .addNode("c_vec")
            .init(new ConstantNode.Builder(floatVec([1, 2, 3])))
            .as(ConstantNode);
        expect(readVectorFromTensorNode(vecNode)).toEqual([1, 2, 3]);
    });

    it("decodes raw Float16 values accurately in readTensorData", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        // 0x3C00 is 1.0 in IEEE 754 half-precision float (Float16)
        const buf = Buffer.alloc(2);
        buf.writeUInt16LE(0x3c00, 0);

        const f16Node = graph
            .addNode("c_f16")
            .init(
                new ConstantNode.Builder({
                    dataType: DataType.FLOAT16,
                    dims: [1],
                    rawData: { type: "Buffer", data: Array.from(buf) },
                }),
            )
            .as(ConstantNode);

        const data = readTensorData(f16Node);
        expect(data).toBeDefined();
        expect(data?.[0]).toBeCloseTo(1.0);
    });
});
