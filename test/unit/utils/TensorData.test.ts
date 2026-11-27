import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import ConstantNode from "../../../src/Onnx/ConstantNode.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
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
    resolveAutoPads,
    buildResolvedConvAttrs,
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

    it("resolves auto_pad modes accurately via resolveAutoPads", () => {
        expect(resolveAutoPads("NOTSET", [5, 5], [3, 3], [1, 1], [1, 1], [1, 2, 3, 4])).toEqual([
            1, 2, 3, 4,
        ]);
        expect(resolveAutoPads("NOTSET", [5, 5], [3, 3], [1, 1], [1, 1])).toEqual([0, 0, 0, 0]);
        expect(resolveAutoPads("VALID", [5, 5], [3, 3], [1, 1], [1, 1], [1, 2, 3, 4])).toEqual([
            0, 0, 0, 0,
        ]);
        // Asymmetric padding: input=4, kernel=3, stride=2, dil=1 => out=2, totalPad=(2-1)*2+3-4=1
        expect(resolveAutoPads("SAME_UPPER", [4, 5], [3, 3], [2, 2], [1, 1])).toEqual([0, 1, 1, 1]);
        expect(resolveAutoPads("SAME_LOWER", [4, 5], [3, 3], [2, 2], [1, 1])).toEqual([1, 1, 0, 1]);
        // With dilations: input=5, kernel=3, stride=1, dil=2 => effK=5, out=5, totalPad=4
        expect(resolveAutoPads("SAME_UPPER", [5], [3], [1], [2])).toEqual([2, 2]);
    });

    it("builds resolved Conv attributes without conflicting auto_pad and pads", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        const xStatic = graph
            .addNode("xStatic")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 4, 4], "input"))
            .as(TensorNode);
        const xDyn = graph
            .addNode("xDyn")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 2, -1, -1], "input"))
            .as(TensorNode);

        const [y1] = builder.createOp("AveragePool", [xStatic], {
            kernel_shape: [2, 2],
            strides: [2, 2],
            auto_pad: "SAME_UPPER",
            pads: [9, 9, 9, 9],
        });
        const op1 = y1.incomers.sources.filterIs(OperationNode).first()!;
        const attrs1 = buildResolvedConvAttrs(op1, [4, 4], [2, 2], { fixedGroup: 2 });
        expect(attrs1["auto_pad"]).toBeUndefined();
        expect(attrs1["pads"]).toEqual([0, 0, 0, 0]);
        expect(attrs1["group"]).toBe(2);

        const [y2] = builder.createOp("AveragePool", [xDyn], {
            kernel_shape: [3, 3],
            strides: [2, 2],
            auto_pad: "SAME_LOWER",
            pads: [1, 1, 1, 1],
        });
        const op2 = y2.incomers.sources.filterIs(OperationNode).first()!;
        const attrs2 = buildResolvedConvAttrs(op2, [-1, -1], [3, 3]);
        expect(attrs2["auto_pad"]).toBe("SAME_LOWER");
        expect(attrs2["pads"]).toBeUndefined();
    });

    it("packs and unpacks INT4, UINT4, INT2, and UINT2 sub-byte tensors in makeTensorProto and readTensorData", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const i4Vals = [-8, -1, 0, 5, 7];
        const i4Proto = makeTensorProto(DataType.INT4, [5], i4Vals);
        expect(i4Proto.rawData?.data.length).toBe(3);
        const i4Node = graph
            .addNode("c_i4")
            .init(new ConstantNode.Builder(i4Proto))
            .as(ConstantNode);
        expect(readTensorData(i4Node)).toEqual(i4Vals);
        expect(decodeIntegerVectorFromTensorProto(i4Proto)).toEqual(i4Vals);

        const u4Vals = [0, 3, 12, 15];
        const u4Proto = makeTensorProto(DataType.UINT4, [4], u4Vals);
        const u4Node = graph
            .addNode("c_u4")
            .init(new ConstantNode.Builder(u4Proto))
            .as(ConstantNode);
        expect(readTensorData(u4Node)).toEqual(u4Vals);

        const i2Vals = [-2, -1, 0, 1, -2];
        const i2Proto = makeTensorProto(DataType.INT2, [5], i2Vals);
        expect(i2Proto.rawData?.data.length).toBe(2);
        const i2Node = graph
            .addNode("c_i2")
            .init(new ConstantNode.Builder(i2Proto))
            .as(ConstantNode);
        expect(readTensorData(i2Node)).toEqual(i2Vals);

        const u2Vals = [0, 1, 2, 3];
        const u2Proto = makeTensorProto(DataType.UINT2, [4], u2Vals);
        const u2Node = graph
            .addNode("c_u2")
            .init(new ConstantNode.Builder(u2Proto))
            .as(ConstantNode);
        expect(readTensorData(u2Node)).toEqual(u2Vals);

        const z4 = zeroTensor(DataType.INT4, [2, 3]);
        expect(z4.rawData?.data).toEqual([0, 0, 0]);
        expect(decodeIntegerVectorFromTensorProto(z4)).toEqual([0, 0, 0, 0, 0, 0]);
    });
});
