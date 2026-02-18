import fs from "fs";
import BaseNode from "@specs-feup/flow/graph/BaseNode";
import OnnxEdge from "./OnnxEdge.js";
import OnnxGraph from "./OnnxGraph.js";
import { TensorProto, DataType } from "./OnnxTypes.js";
import OperationNode from "./OperationNode.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import RegionArgumentNode from "./RegionArgumentNode.js";

// =====================================================================================
// SECTION 1: TYPES & CONSTANTS
// =====================================================================================

export type Dim = number | string;
export type Shape = Dim[];

export const typeSizeMap: Record<number, number> = {
    0: 0, // UNDEFINED
    1: 4, // FLOAT
    2: 1, // UINT8
    3: 1, // INT8
    4: 2, // UINT16
    5: 2, // INT16
    6: 4, // INT32
    7: 8, // INT64
    8: -1, // STRING
    9: 1, // BOOL
    10: 2, // FLOAT16
    11: 8, // DOUBLE
    12: 4, // UINT32
    13: 8, // UINT64
    14: 8, // COMPLEX64
    15: 16, // COMPLEX128
    16: 2, // BFLOAT16
    17: 1, // FLOAT8E4M3FN
    18: 1, // FLOAT8E4M3FNUZ
    19: 2, // FLOAT8E5M2
    20: 2, // FLOAT8E5M2FNUZ
    21: 1, // UINT4
    22: 1, // INT4
};

// =====================================================================================
// SECTION 2: TENSOR PROTO CREATION HELPERS
// =====================================================================================

export function bool(v: boolean): TensorProto {
    return { dataType: DataType.BOOL, dims: [], int32Data: [v ? 1 : 0] };
}

export function scalarInt32(v: number): TensorProto {
    return { dataType: DataType.INT32, dims: [], int32Data: [Number(v)] };
}

export function int32Vec(arr: number[]): TensorProto {
    return { dataType: DataType.INT32, dims: [arr.length], int32Data: arr.map(Number) };
}

export function scalarInt64(v: number): TensorProto {
    return { dataType: DataType.INT64, dims: [], int64Data: [Number(v)] };
}

export function int64Vec(arr: number[]): TensorProto {
    return { dataType: DataType.INT64, dims: [arr.length], int64Data: arr.map(Number) };
}

export function scalarFloat(v: number): TensorProto {
    return { dataType: DataType.FLOAT, dims: [], floatData: [Number(v)] };
}

export function floatVec(arr: number[]): TensorProto {
    return { dataType: DataType.FLOAT, dims: [arr.length], floatData: arr.map(Number) };
}

export function makeTensorProto(dtype: DataType, dims: number[], values: number[]): TensorProto {
    const t: TensorProto = { dataType: dtype, dims };

    switch (dtype) {
        case DataType.FLOAT:
            t.floatData = values;
            break;
        case DataType.DOUBLE:
            t.doubleData = values;
            break;
        case DataType.INT32:
            t.int32Data = values.map((v) => v | 0);
            break;
        case DataType.INT64:
            t.int64Data = values.map((v) => Number(v));
            break;
        case DataType.UINT64:
            t.uint64Data = values.map((v) => Number(v));
            break;
        case DataType.BOOL:
            t.int32Data = values.map((v) => (v ? 1 : 0));
            break;
        default: {
            // Fallback: encode as raw little-endian 32-bit floats or similar
            const buf = Buffer.alloc(values.length * 4);
            const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
            values.forEach((x, i) => dv.setFloat32(i * 4, x, true));
            t.rawData = { type: "Buffer", data: Array.from(buf) };
            break;
        }
    }
    return t;
}

export function zeroTensor(elemType: DataType, shape: number[]): TensorProto {
    const safeShape = shape && shape.length ? shape.map((d) => (d != null && d > 0 ? d : 1)) : [1];
    const n = safeShape.reduce((a, b) => a * b, 1);
    const base: TensorProto = { dataType: elemType, dims: safeShape };

    switch (elemType) {
        case DataType.FLOAT:
            return { ...base, floatData: new Array<number>(n).fill(0) };
        case DataType.DOUBLE:
            return { ...base, doubleData: new Array<number>(n).fill(0) };
        case DataType.INT64:
            return { ...base, int64Data: new Array<number>(n).fill(0) };
        case DataType.UINT64:
            return { ...base, uint64Data: new Array<number>(n).fill(0) };
        default:
            return { ...base, int32Data: new Array<number>(n).fill(0) };
    }
}

// =====================================================================================
// SECTION 3: CONSTANT NODE FACTORIES (Phase 3 Updated)
// =====================================================================================

export function uniq(g: OnnxGraph.Class, base: string): string {
    let i = 0,
        id = base;
    while (g.hasNode(id)) id = `${base}_${++i}`;
    return id;
}

/** Create a ConstantNode for a rank-0 scalar of given type. */
export function scalarOfType(
    g: OnnxGraph.Class,
    name: string,
    v: number,
    dtype: DataType,
): ConstantNode.Class {
    const proto = makeTensorProto(dtype, [], [v]);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Create a ConstantNode for a rank-0 INT64 scalar. */
export function scalarI64(g: OnnxGraph.Class, name: string, v: number): ConstantNode.Class {
    return scalarOfType(g, name, v, DataType.INT64);
}

/** Create a ConstantNode for a rank-0 scalar zero. */
export function scalarZeroOfType(
    g: OnnxGraph.Class,
    name: string,
    dtype: DataType,
): ConstantNode.Class {
    return scalarOfType(g, name, 0, dtype);
}

/** Create a ConstantNode for a 1D INT64 vector. */
export function constI64(g: OnnxGraph.Class, name: string, vals: number[]): ConstantNode.Class {
    const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Create a ConstantNode for a 1D FLOAT vector. */
export function constF32(g: OnnxGraph.Class, name: string, vals: number[]): ConstantNode.Class {
    const proto = makeTensorProto(DataType.FLOAT, [vals.length], vals);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Create a ConstantNode filled with ones. */
export function tensorOnesConst(
    g: OnnxGraph.Class,
    name: string,
    dtype: DataType,
    shape: number[],
): ConstantNode.Class {
    const size = shape.reduce((a, b) => a * b, 1);
    const ones = new Array<number>(size).fill(1);
    const proto = makeTensorProto(dtype, shape, ones);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Generic helper to create a ConstantNode from a TensorProto. */
export function makeTensorConst(
    g: OnnxGraph.Class,
    id: string,
    proto: TensorProto,
): ConstantNode.Class {
    return g.addNode(uniq(g, id)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

export function makeValueScalar1(
    g: OnnxGraph.Class,
    name: string,
    dtype: DataType,
    v: number,
): ConstantNode.Class {
    const proto = makeTensorProto(dtype, [1], [v]);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

export function makeI64ShapeConst(
    g: OnnxGraph.Class,
    name: string,
    vals: number[],
): ConstantNode.Class {
    return constI64(g, name, vals);
}

// Helper to create a scalar ConstantNode Builder
export const constBuilder = (val: number) => {
    return new ConstantNode.Builder(makeTensorProto(DataType.INT64, [], [val]));
};

// =====================================================================================
// SECTION 4: COMPUTATIONAL HELPERS (Create Operations)
// =====================================================================================

/** Creates a Shape op + intermediate output tensor. */
export function shapeOf(
    g: OnnxGraph.Class,
    x: TensorNode.Class | ConstantNode.Class,
    name: string,
): TensorNode.Class {
    const sop = g
        .addNode(uniq(g, `${name}_op`))
        .init(new OperationNode.Builder("Shape", [x], {}))
        .as(OperationNode);
    const s = g
        .addNode(uniq(g, `${name}`))
        .init(new TensorNode.Builder(DataType.INT64, [x.shape.length], "intermediate"))
        .as(TensorNode);
    addEdge(g, sop, s, DataType.INT64, [x.shape.length]);
    return s;
}

/** Creates ScatterElements to edit a specific dimension of a shape tensor. */
export function editShapeDim(
    g: OnnxGraph.Class,
    baseShape: TensorNode.Class,
    axis: number,
    size1D: TensorNode.Class | ConstantNode.Class,
    name: string,
): TensorNode.Class {
    const idx = makeI64ShapeConst(g, `${name}_idx`, [axis]);

    const shapeOne = makeI64ShapeConst(g, `${name}_vec_shape`, [1]);

    const reshapeOp = g
        .addNode(uniq(g, `${name}_ensure_vec_op`))
        .init(new OperationNode.Builder("Reshape", [size1D, shapeOne]))
        .as(OperationNode);

    const updateVec = g
        .addNode(uniq(g, `${name}_ensure_vec`))
        .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
        .as(TensorNode);

    addEdge(g, reshapeOp, updateVec, DataType.INT64, [1]);

    const sc = g
        .addNode(uniq(g, `${name}_sc`))
        .init(
            new OperationNode.Builder("ScatterElements", [baseShape, idx, updateVec], { axis: 0 }),
        )
        .as(OperationNode);
    const out = g
        .addNode(uniq(g, `${name}_out`))
        .init(
            new TensorNode.Builder(DataType.INT64, [baseShape.shape[0] as number], "intermediate"),
        )
        .as(TensorNode);
    addEdge(g, sc, out, DataType.INT64, [baseShape.shape[0] as number]);
    return out;
}

// =====================================================================================
// SECTION 5: GRAPH MANIPULATION & QUERY
// =====================================================================================

export function formatId(name: string, nodeId: string): string {
    return `${name}_${nodeId}`;
}

export function addEdge(
    g: OnnxGraph.Class,
    srcOp: OperationNode.Class,
    dstTensor: TensorNode.Class,
    dtype: DataType,
    shape?: Array<number | string | undefined>,
) {
    g.addEdge(srcOp, dstTensor)
        .init(new OnnxEdge.Builder(dtype, shape ?? dstTensor.shape))
        .as(OnnxEdge);
}

export function toArrayLike<T = any>(nc: any): T[] {
    return nc?.toArray?.() ?? nc ?? [];
}

/** Remove an initializer entry from the graph metadata (cleanup). */
export function removeInitializerByName(g: OnnxGraph.Class, name?: string) {
    if (!name) return;
    const anyG: any = g as any;
    const model = anyG?.rawModel ?? anyG?.model;
    const graph = model?.graph ?? anyG?.graph;
    if (!graph) return;
    for (const f of ["initializer", "sparse_initializer", "input", "value_info"]) {
        if (Array.isArray(graph[f])) graph[f] = graph[f].filter((x: any) => x?.name !== name);
    }
}

/** Removes a ConstantNode if it has no consumers. */
export function maybeRemoveOrphanConstant(g: OnnxGraph.Class, node?: BaseNode.Class) {
    if (!node) return;
    // Strict check for ConstantNode (Phase 3)
    if (node.is(ConstantNode)) {
        const consumers = toArrayLike(node.outgoers?.targets);
        if (consumers.length === 0) {
            const onnxName = node.id;
            node.remove();
            removeInitializerByName(g, onnxName);
        }
    }
}

/** Looks up a tensor-like node (TensorNode or ConstantNode) by ID or original name. */
export function findTensorByOnnxName(
    g: OnnxGraph.Class,
    name?: string,
): TensorNode.Class | ConstantNode.Class | undefined {
    if (!name) return undefined;

    // Check Constants
    const constants = g.nodes.filterIs(ConstantNode).toArray() as ConstantNode.Class[];
    const tConst = constants.find((n) => n.id === name || n.constantValue.name === name);
    if (tConst) return tConst;

    // Check Tensors
    const tensors = (g.getTensorNodes?.().toArray?.() ?? []) as TensorNode.Class[];
    const t = tensors.find((n) => n.id === name);
    return t;
}

export function findConstantProducerAsTensor(
    g: OnnxGraph.Class,
    onnxName?: string,
): ConstantNode.Class | undefined {
    if (!onnxName) return undefined;
    // In Phase 3, constants are just ConstantNodes.
    return findTensorByOnnxName(g, onnxName)?.tryAs(ConstantNode);
}

// =====================================================================================
// SECTION 6: DATA READERS (Phase 3 Updated)
// =====================================================================================

export function toU8(raw: any): Uint8Array | undefined {
    if (!raw) return undefined;
    if (raw instanceof Uint8Array) return raw;
    if (Array.isArray(raw)) return Uint8Array.from(raw);
    if ((globalThis as any).Buffer?.isBuffer(raw)) {
        const b: Buffer = raw as any;
        return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    }
    const inner = raw.data ?? undefined;
    if (inner) return toU8(inner);
    return undefined;
}

export function totalSizeFromDims(
    fallbackElems: number,
    dims?: (number | string)[] | undefined,
): number {
    if (!Array.isArray(dims) || dims.length === 0) return fallbackElems;
    return dims.map((d) => Number(d)).reduce((a, b) => a * b, 1);
}

export function isInt64Type(dt: number | string | undefined): boolean {
    return dt === 7 || dt === "INT64";
}

export function decodeIntegerVectorFromTensorProto(tv: TensorProto): number[] | undefined {
    if (!tv) return undefined;

    if (Array.isArray(tv.int64Data) && tv.int64Data.length) return tv.int64Data.map(Number);
    if (Array.isArray(tv.int32Data) && tv.int32Data.length) return tv.int32Data.map(Number);
    if (Array.isArray(tv.uint64Data) && tv.uint64Data.length) return tv.uint64Data.map(Number);

    const u8 = toU8(tv.rawData ?? undefined);
    if (!u8) return undefined;

    const i64 = isInt64Type(tv.dataType);
    const elemBytes = i64 ? 8 : 4;
    const n = totalSizeFromDims(Math.floor(u8.byteLength / elemBytes), tv.dims);
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
        const off = i * elemBytes;
        out.push(i64 ? Number(dv.getBigInt64(off, true)) : dv.getInt32(off, true));
    }
    return out;
}

/** * Reads integer vector from a node.
 * Supports ConstantNode directly.
 */
export function readConstIntegerVectorFromTensorNode(node?: BaseNode.Class): number[] | undefined {
    if (!node) return undefined;

    let tv: TensorProto;
    if (node.is(ConstantNode)) {
        tv = node.as(ConstantNode).constantValue;
    }

    if (!tv) return undefined;
    return decodeIntegerVectorFromTensorProto(tv);
}

/** * Reads a scalar from a node.
 * Supports ConstantNode directly.
 */
export function readScalarFromTensorNode(node?: BaseNode.Class): number | undefined {
    if (!node) return undefined;

    let tv: TensorProto | undefined;
    if (node.is(ConstantNode)) {
        tv = node.as(ConstantNode).constantValue;
    }

    if (!tv) return undefined;

    if (Array.isArray(tv.floatData) && tv.floatData.length) return Number(tv.floatData[0]);
    if (Array.isArray(tv.doubleData) && tv.doubleData.length) return Number(tv.doubleData[0]);
    if (Array.isArray(tv.int64Data) && tv.int64Data.length) return Number(tv.int64Data[0]);
    if (Array.isArray(tv.int32Data) && tv.int32Data.length) return Number(tv.int32Data[0]);

    const u8 = toU8(tv.rawData ?? undefined);
    if (u8) {
        if (u8.byteLength === 8) {
            const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
            // Try float64 then int64
            const f = dv.getFloat64(0, true);
            if (!isNaN(f) && Number.isFinite(f)) return f;
            return Number(dv.getBigInt64(0, true));
        } else if (u8.byteLength === 4) {
            const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
            const f = dv.getFloat32(0, true);
            if (!isNaN(f) && Number.isFinite(f)) return Number(f);
            return dv.getInt32(0, true);
        }
    }
    return undefined;
}

/**
 * Universal helper to read ANY numeric tensor as a JavaScript number array.
 * Handles Float, Double, Int32, Int64, and raw byte buffers.
 */
export function readVectorFromTensorNode(node?: BaseNode.Class): number[] | undefined {
    if (!node) return undefined;

    // 1. Get the TensorProto
    let tv: TensorProto | undefined;
    if (node.is(ConstantNode)) {
        tv = node.as(ConstantNode).constantValue;
    }

    if (!tv) return undefined;

    // 2. Try explicit fields first (Fastest)
    if (Array.isArray(tv.floatData) && tv.floatData.length) return tv.floatData.map(Number);
    if (Array.isArray(tv.doubleData) && tv.doubleData.length) return tv.doubleData.map(Number);
    if (Array.isArray(tv.int64Data) && tv.int64Data.length) return tv.int64Data.map(Number);
    if (Array.isArray(tv.int32Data) && tv.int32Data.length) return tv.int32Data.map(Number);
    if (Array.isArray(tv.uint64Data) && tv.uint64Data.length) return tv.uint64Data.map(Number);

    // 3. Fallback to Raw Data Parsing
    const u8 = toU8(tv.rawData ?? undefined);
    if (!u8) return undefined;

    const dt = tv.dataType;
    const len = totalSizeFromDims(0, tv.dims);
    // If dims is missing/empty, try to guess from byte length for common types
    const elemCount = len > 0 ? len : 0;

    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const out: number[] = [];

    // Helper to estimate count if unknown
    const getCount = (bytes: number) =>
        elemCount > 0 ? elemCount : Math.floor(u8.byteLength / bytes);

    // 1 = FLOAT
    if (dt === 1) {
        const n = getCount(4);
        for (let i = 0; i < n; i++) out.push(dv.getFloat32(i * 4, true));
    }
    // 11 = DOUBLE
    else if (dt === 11) {
        const n = getCount(8);
        for (let i = 0; i < n; i++) out.push(dv.getFloat64(i * 8, true));
    }
    // 7 = INT64
    else if (dt === 7) {
        const n = getCount(8);
        for (let i = 0; i < n; i++) out.push(Number(dv.getBigInt64(i * 8, true)));
    }
    // 6 = INT32
    else if (dt === 6) {
        const n = getCount(4);
        for (let i = 0; i < n; i++) out.push(dv.getInt32(i * 4, true));
    }
    // 2 = UINT8
    else if (dt === 2) {
        return Array.from(u8);
    }
    // Add others if needed (Float16, Int8, etc)

    return out.length > 0 ? out : undefined;
}

// Helper for Float16 decoding (IEEE 754 half-precision)
function decodeFloat16(binary: number): number {
    const exponent = (binary & 0x7c00) >> 10;
    const fraction = binary & 0x03ff;
    const sign = binary >> 15 === 0 ? 1 : -1;

    if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
    if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/**
 * Reads any numeric tensor (Scalar or Vector) into a JavaScript number array.
 * Automatically handles Float vs Int parsing based on the node's stored DataType.
 */
export function readTensorData(node?: BaseNode.Class): number[] | undefined {
    if (!node) return undefined;

    // 1. Get TensorProto
    let tv: TensorProto | undefined;
    if (node.is(ConstantNode)) {
        tv = node.as(ConstantNode).constantValue;
    }

    if (!tv) return undefined;

    // 2. Try explicit fields (Fastest)
    if (Array.isArray(tv.floatData) && tv.floatData.length) return tv.floatData.map(Number);
    if (Array.isArray(tv.int64Data) && tv.int64Data.length) return tv.int64Data.map(Number);
    if (Array.isArray(tv.int32Data) && tv.int32Data.length) return tv.int32Data.map(Number);
    if (Array.isArray(tv.doubleData) && tv.doubleData.length) return tv.doubleData.map(Number);
    if (Array.isArray(tv.uint64Data) && tv.uint64Data.length) return tv.uint64Data.map(Number);

    // 3. Fallback: Parse Raw Bytes
    const u8 = toU8(tv.rawData ?? undefined);
    if (!u8) return undefined;

    const dt = tv.dataType;
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const out: number[] = [];

    // Calculate count
    let count = totalSizeFromDims(0, tv.dims);
    if (count <= 0) {
        // Estimate based on type size
        const bytesPerElem =
            dt === 1 || dt === 6 || dt === 12
                ? 4 // 32-bit
                : dt === 7 || dt === 11 || dt === 13
                  ? 8 // 64-bit
                  : dt === 4 || dt === 5 || dt === 10 || dt === 16
                    ? 2 // 16-bit
                    : 1; // 8-bit
        count = Math.floor(u8.byteLength / bytesPerElem);
    }

    for (let i = 0; i < count; i++) {
        switch (dt) {
            // --- 32-bit ---
            case 1: // FLOAT
                out.push(dv.getFloat32(i * 4, true));
                break;
            case 6: // INT32
                out.push(dv.getInt32(i * 4, true));
                break;
            case 12: // UINT32
                out.push(dv.getUint32(i * 4, true));
                break;

            // --- 64-bit ---
            case 7: // INT64
                out.push(Number(dv.getBigInt64(i * 8, true)));
                break;
            case 11: // DOUBLE
                out.push(dv.getFloat64(i * 8, true));
                break;
            case 13: // UINT64
                out.push(Number(dv.getBigUint64(i * 8, true)));
                break;

            // --- 16-bit ---
            case 5: // INT16
                out.push(dv.getInt16(i * 2, true));
                break;
            case 4: // UINT16
                out.push(dv.getUint16(i * 2, true));
                break;
            case 10: // FLOAT16
                out.push(decodeFloat16(dv.getUint16(i * 2, true)));
                break;
            case 16: // BFLOAT16 (Rough approximation: high 16 bits of F32)
                // JS doesn't have native BFloat16, usually we treat as Float32 truncated
                // Shift left 16 to simulate F32, then read.
                // Simple hack: Just cast to UInt16 for now if you don't need math precision
                // or implement full decoder if critical.
                out.push(dv.getUint16(i * 2, true));
                break;

            // --- 8-bit ---
            case 2: // UINT8
                out.push(u8[i]);
                break;
            case 3: // INT8
                out.push(new Int8Array(u8.buffer, u8.byteOffset)[i]);
                break;
            case 9: // BOOL
                out.push(u8[i] ? 1 : 0);
                break;

            default:
                return undefined; // Complex, String, or 4-bit types not supported for numeric array
        }
    }

    return out;
}

// =====================================================================================
// SECTION 7: SHAPE & MATH HELPERS
// =====================================================================================

export function isNumeric(dtype: DataType): boolean {
    return !(dtype === DataType.STRING || dtype === DataType.BOOL);
}

export function toNum(x: number | string | undefined): number | undefined {
    if (typeof x === "number") return x;
    // FIX: Parse strings like "378" correctly
    if (typeof x === "string") {
        const n = Number(x);
        return isNaN(n) ? undefined : n;
    }
    return undefined;
}

export function toNumShape(
    s?: Array<number | string | undefined>,
): Array<number | undefined> | undefined {
    if (!s) return undefined;
    return s.map(toNum);
}

export function asStaticDims(shape: (number | string)[]): number[] {
    return shape.map((d) => {
        const n = toNum(d);
        return n !== undefined && n > 0 ? n : 1;
    });
}

export function toStaticShape(shape: Shape): number[] {
    return shape.map((d) => {
        const n = toNum(d);
        return n !== undefined ? n : -1;
    });
}

export function prodSafe(dims: number[]): number {
    return dims.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

export function computeStrides(dims: number[]): number[] {
    const n = dims.length;
    const strides = new Array(n);
    let acc = 1;
    for (let i = n - 1; i >= 0; --i) {
        const d = dims[i] > 0 ? dims[i] : 1;
        strides[i] = acc;
        acc *= d;
    }
    return strides;
}

export function normalizeAxis(axis: number, rank: number): number {
    if (rank <= 0) return axis;
    return ((axis % rank) + rank) % rank;
}

export function broadcastTwoShapes(a: number[], b: number[]): number[] {
    const ra = a.length,
        rb = b.length;
    const r = Math.max(ra, rb);
    const out = new Array<number>(r);
    for (let i = 0; i < r; i++) {
        const da = a[ra - 1 - i] ?? 1;
        const db = b[rb - 1 - i] ?? 1;
        if (da === 1) out[r - 1 - i] = db;
        else if (db === 1) out[r - 1 - i] = da;
        else if (da === db) out[r - 1 - i] = da;
        else out[r - 1 - i] = Math.max(da, db);
    }
    return out;
}

export function broadcastShapes(...shapes: number[][]): number[] {
    return shapes.reduce((acc, s) => broadcastTwoShapes(acc, s), []);
}

export function inferPoolDim(
    inDim: number,
    k: number,
    stride: number,
    padHead: number,
    padTail: number,
    dil: number,
) {
    const effectiveK = dil * (k - 1) + 1;
    return Math.floor((inDim + padHead + padTail - effectiveK) / stride + 1);
}

export function inferConvDim(
    inDim: number,
    k: number,
    stride: number,
    padHead: number,
    padTail: number,
    dil: number,
) {
    return inferPoolDim(inDim, k, stride, padHead, padTail, dil);
}

export function getAttr(node: any, name: string, def?: any) {
    const v = node.getAttributes?.[name] ?? node.attributes?.[name];
    return v === undefined ? def : v;
}

export function getSmallestRankShape(tensors: TensorNode.Class[]): Shape {
    if (tensors.length === 0) return [];
    let smallest = tensors[0].shape;
    for (let i = 1; i < tensors.length; i++) {
        if (tensors[i].shape.length < smallest.length) smallest = tensors[i].shape;
    }
    return smallest;
}

export function getLargestRankShape(
    tensors: (TensorNode.Class | ConstantNode.Class | RegionArgumentNode.Class)[],
): Shape {
    if (tensors.length === 0) return [];
    let largest = tensors[0].shape;
    for (let i = 1; i < tensors.length; i++) {
        if (tensors[i].shape.length > largest.length) largest = tensors[i].shape;
    }
    return largest;
}

// =====================================================================================
// SECTION 8: GRAPH ALGORITHMS
// =====================================================================================

export function topologicalSortOperationNodes(graph: OnnxGraph.Class): OperationNode.Class[] {
    const sorted: OperationNode.Class[] = [];
    const visited = new Set<string>();
    const temp = new Set<string>();

    const opNodes = graph.getOperationNodes().toArray();

    // Map tensor id -> producing op
    const tensorProducers = new Map<string, OperationNode.Class>();
    for (const op of opNodes) {
        const outTensors =
            op.getOutgoers?.targets?.filter((n) => n.is(TensorNode)).toArray?.() ?? [];
        for (const t of outTensors as TensorNode.Class[]) {
            tensorProducers.set(t.id, op);
        }
    }

    // Extra deps from implicit subgraph captures (Phase 4: Explicit RegionArgumentNode)
    const extraDeps = new Map<string, Set<OperationNode.Class>>();

    for (const op of opNodes) {
        // Iterate over strict regions
        const regions = op.regions ?? [];

        for (const sg of regions) {
            if (!sg) continue;

            // Find explicit captures via RegionArgumentNode
            const nodes = sg.getNodes().toArray();
            for (const node of nodes) {
                if (node.is(RegionArgumentNode)) {
                    const arg = node.as(RegionArgumentNode);
                    const parentName = arg.originalName;

                    const parentProd = tensorProducers.get(parentName);

                    // If the parent node is produced by an op in the current graph, we depend on it.
                    if (parentProd && parentProd.id !== op.id) {
                        let deps = extraDeps.get(op.id);
                        if (!deps) {
                            deps = new Set<OperationNode.Class>();
                            extraDeps.set(op.id, deps);
                        }
                        deps.add(parentProd);
                    }
                }
            }
        }
    }

    const visit = (node: OperationNode.Class) => {
        if (visited.has(node.id) || !graph.hasNode(node.id)) return;
        if (temp.has(node.id)) {
            console.warn(`[TopoSort] Cycle detected at ${node.id}`);
            return;
        }
        temp.add(node.id);

        // 1. Explicit Captures dependencies
        const implicitPreds = extraDeps.get(node.id);
        if (implicitPreds) implicitPreds.forEach(visit);

        // 2. Explicit dependencies (Inputs)
        const checkPred = (n: BaseNode.Class) => {
            if (n.is(OperationNode)) {
                const op = n.as(OperationNode);
                // Follow intermediate tensor inputs recursively
                for (const input of op.getInputs?.() ?? []) {
                    if (
                        input &&
                        input.is(TensorNode) &&
                        input.as(TensorNode).type === "intermediate"
                    ) {
                        checkPred(input);
                    }
                }
            }
            // Check incomers (edges)
            const incomers = n.incomers?.toArray?.() ?? [];
            for (const edge of incomers) {
                const src = edge?.source;
                if (!src) continue;
                if (src.is(OperationNode)) visit(src.as(OperationNode));
                else if (src.is(TensorNode) && src.as(TensorNode).type === "intermediate")
                    checkPred(src);
            }
        };

        checkPred(node);

        temp.delete(node.id);
        visited.add(node.id);
        sorted.push(node);
    };

    opNodes.forEach(visit);
    return sorted;
}

// =====================================================================================
// SECTION 9: DEBUG UTILS
// =====================================================================================

export function dbg(...args: any[]): void {
    console.log("[loop-debug]", ...args);
}

export function dbgTensor(label: string, t: BaseNode.Class | null | undefined): void {
    if (!t) return;
    if (t.is(TensorNode)) {
        const tn = t.as(TensorNode);
        dbg(label, { id: tn.id, kind: tn.type, elemType: tn.literalType, shape: tn.shape });
    } else if (t.is(ConstantNode)) {
        const cn = t.as(ConstantNode);
        dbg(label, { id: cn.id, kind: "constant", elemType: cn.literalType, shape: cn.shape });
    }
}

export function safeWriteJson(filePath: string, obj: any) {
    const fd = fs.openSync(filePath, "w");
    const BUFFER_LIMIT = 1 << 20;
    let buffer = "";

    const flush = () => {
        if (buffer.length > 0) {
            fs.writeSync(fd, buffer);
            buffer = "";
        }
    };
    const write = (s: string) => {
        buffer += s;
        if (buffer.length >= BUFFER_LIMIT) flush();
    };

    const seen = new Set<any>();

    const writeValue = (value: any) => {
        if (value === null || value === undefined) {
            write("null");
            return;
        }
        const t = typeof value;
        if (t === "number" || t === "boolean") {
            write(String(value));
            return;
        }
        if (t === "string") {
            write(JSON.stringify(value));
            return;
        }

        if (Array.isArray(value)) {
            write("[");
            for (let i = 0; i < value.length; i++) {
                if (i > 0) write(",");
                writeValue(value[i]);
            }
            write("]");
            return;
        }
        if (t === "object") {
            if (seen.has(value)) throw new Error("safeWriteJson: cyclic reference");
            seen.add(value);
            const keys = Object.keys(value);
            write("{");
            for (let i = 0; i < keys.length; i++) {
                if (i > 0) write(",");
                write(JSON.stringify(keys[i]));
                write(":");
                writeValue((value as any)[keys[i]]);
            }
            write("}");
            seen.delete(value);
        }
    };

    try {
        writeValue(obj);
        flush();
    } finally {
        fs.closeSync(fd);
    }
}
