import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../ConstantNode.js";
import type { TensorProto, KnownShape } from "../OnnxTypes.js";
import { DataType } from "../OnnxTypes.js";

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
    const safeShape = shape.length ? shape.map((d) => (d > 0 ? d : 1)) : [1];
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

export function toU8(raw: unknown): Uint8Array | undefined {
    if (raw === undefined) return undefined;
    if (raw instanceof Uint8Array) return raw;
    if (Array.isArray(raw)) return Uint8Array.from(raw);
    const globalEnv = globalThis as unknown as { Buffer?: { isBuffer: (v: unknown) => boolean } };
    if (globalEnv.Buffer !== undefined && globalEnv.Buffer.isBuffer(raw)) {
        const b = raw as Buffer;
        return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    }
    const inner = (raw as Record<string, unknown>)["data"] ?? undefined;
    if (inner) return toU8(inner);
    return undefined;
}

export function totalSizeFromDims(fallbackElems: number, dims?: KnownShape | undefined): number {
    if (!Array.isArray(dims) || dims.length === 0) return fallbackElems;
    return dims.map((d) => Number(d)).reduce((a, b) => a * b, 1);
}

export function isInt64Type(dt: DataType | string | undefined): boolean {
    return dt === DataType.INT64 || dt === "INT64";
}

export function decodeIntegerVectorFromTensorProto(tv: TensorProto): number[] | undefined {
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
        if (off + elemBytes > dv.byteLength) break;
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
        return decodeIntegerVectorFromTensorProto(tv);
    }

    return undefined;
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
