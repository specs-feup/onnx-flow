import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../ir/ConstantNode.js";
import type OperationNode from "../ir/OperationNode.js";
import type { TensorProto, KnownShape, AttributeMap } from "../ir/OnnxTypes.js";
import { DataType } from "../ir/OnnxTypes.js";

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

export function packSubByteInts(values: number[], bits: 2 | 4, signed: boolean): number[] {
    const perByte = bits === 4 ? 2 : 4;
    const mask = bits === 4 ? 0x0f : 0x03;
    const byteLen = Math.ceil(values.length / perByte);
    const bytes = new Array<number>(byteLen).fill(0);
    for (let i = 0; i < values.length; i++) {
        let v = Math.trunc(values[i]);
        if (signed) {
            const min = -(1 << (bits - 1));
            const max = (1 << (bits - 1)) - 1;
            v = Math.max(min, Math.min(max, v));
            if (v < 0) v = (1 << bits) + v;
        } else {
            const max = (1 << bits) - 1;
            v = Math.max(0, Math.min(max, v));
        }
        const byteIdx = Math.floor(i / perByte);
        const shift = (i % perByte) * bits;
        bytes[byteIdx] |= (v & mask) << shift;
    }
    return bytes;
}

export function unpackSubByteInts(
    u8: Uint8Array,
    bits: 2 | 4,
    signed: boolean,
    count?: number,
): number[] {
    const perByte = bits === 4 ? 2 : 4;
    const mask = bits === 4 ? 0x0f : 0x03;
    const signBit = 1 << (bits - 1);
    const mod = 1 << bits;
    const n = count !== undefined && count > 0 ? count : u8.byteLength * perByte;
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
        const byteIdx = Math.floor(i / perByte);
        if (byteIdx >= u8.byteLength) break;
        const shift = (i % perByte) * bits;
        const raw = (u8[byteIdx] >> shift) & mask;
        out.push(signed && raw >= signBit ? raw - mod : raw);
    }
    return out;
}

export function makeTensorProto(dtype: DataType, dims: number[], values: number[]): TensorProto {
    const t: TensorProto = { dataType: dtype, dims };

    switch (dtype) {
        case DataType.FLOAT:
            if (values.some((v) => !Number.isFinite(v))) {
                const buf = Buffer.alloc(values.length * 4);
                const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
                values.forEach((x, i) => dv.setFloat32(i * 4, x, true));
                t.rawData = { type: "Buffer", data: Array.from(buf) };
            } else {
                t.floatData = values;
            }
            break;
        case DataType.DOUBLE:
            if (values.some((v) => !Number.isFinite(v))) {
                const buf = Buffer.alloc(values.length * 8);
                const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
                values.forEach((x, i) => dv.setFloat64(i * 8, x, true));
                t.rawData = { type: "Buffer", data: Array.from(buf) };
            } else {
                t.doubleData = values;
            }
            break;
        case DataType.UINT8:
        case DataType.INT8:
        case DataType.UINT16:
        case DataType.INT16:
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
        case DataType.UINT4:
            t.rawData = { type: "Buffer", data: packSubByteInts(values, 4, false) };
            break;
        case DataType.INT4:
            t.rawData = { type: "Buffer", data: packSubByteInts(values, 4, true) };
            break;
        case DataType.UINT2:
            t.rawData = { type: "Buffer", data: packSubByteInts(values, 2, false) };
            break;
        case DataType.INT2:
            t.rawData = { type: "Buffer", data: packSubByteInts(values, 2, true) };
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
        case DataType.UINT4:
        case DataType.INT4:
            return {
                ...base,
                rawData: { type: "Buffer", data: new Array<number>(Math.ceil(n / 2)).fill(0) },
            };
        case DataType.UINT2:
        case DataType.INT2:
            return {
                ...base,
                rawData: { type: "Buffer", data: new Array<number>(Math.ceil(n / 4)).fill(0) },
            };
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

    if (tv.dataType === DataType.UINT4 || tv.dataType === DataType.INT4) {
        const count = totalSizeFromDims(0, tv.dims);
        return unpackSubByteInts(
            u8,
            4,
            tv.dataType === DataType.INT4,
            count > 0 ? count : undefined,
        );
    }
    if (tv.dataType === DataType.UINT2 || tv.dataType === DataType.INT2) {
        const count = totalSizeFromDims(0, tv.dims);
        return unpackSubByteInts(
            u8,
            2,
            tv.dataType === DataType.INT2,
            count > 0 ? count : undefined,
        );
    }

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
 * Delegates to `readTensorData` to support all ONNX numeric data types uniformly.
 */
export function readVectorFromTensorNode(node?: BaseNode.Class): number[] | undefined {
    const data = readTensorData(node);
    return data !== undefined && data.length > 0 ? data : undefined;
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
 * Automatically handles explicit typed arrays and raw little-endian byte buffers.
 */
export function readTensorData(node?: BaseNode.Class): number[] | undefined {
    if (!node || !node.is(ConstantNode)) return undefined;
    const tv = node.as(ConstantNode).constantValue;

    if (Array.isArray(tv.floatData) && tv.floatData.length > 0) return tv.floatData.map(Number);
    if (Array.isArray(tv.int64Data) && tv.int64Data.length > 0) return tv.int64Data.map(Number);
    if (Array.isArray(tv.int32Data) && tv.int32Data.length > 0) return tv.int32Data.map(Number);
    if (Array.isArray(tv.doubleData) && tv.doubleData.length > 0) return tv.doubleData.map(Number);
    if (Array.isArray(tv.uint64Data) && tv.uint64Data.length > 0) return tv.uint64Data.map(Number);

    const u8 = toU8(tv.rawData ?? undefined);
    if (!u8) return undefined;

    const dt = tv.dataType;
    if (dt === DataType.UINT4 || dt === DataType.INT4) {
        const elemCount = totalSizeFromDims(0, tv.dims);
        return unpackSubByteInts(
            u8,
            4,
            dt === DataType.INT4,
            elemCount > 0 ? elemCount : undefined,
        );
    }
    if (dt === DataType.UINT2 || dt === DataType.INT2) {
        const elemCount = totalSizeFromDims(0, tv.dims);
        return unpackSubByteInts(
            u8,
            2,
            dt === DataType.INT2,
            elemCount > 0 ? elemCount : undefined,
        );
    }

    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const out: number[] = [];

    let count = totalSizeFromDims(0, tv.dims);
    if (count <= 0) {
        const bytesPerElem =
            dt === DataType.FLOAT || dt === DataType.INT32 || dt === DataType.UINT32
                ? 4
                : dt === DataType.INT64 || dt === DataType.DOUBLE || dt === DataType.UINT64
                  ? 8
                  : dt === DataType.UINT16 ||
                      dt === DataType.INT16 ||
                      dt === DataType.FLOAT16 ||
                      dt === DataType.BFLOAT16
                    ? 2
                    : 1;
        count = Math.floor(u8.byteLength / bytesPerElem);
    }

    for (let i = 0; i < count; i++) {
        switch (dt) {
            case DataType.FLOAT:
                out.push(dv.getFloat32(i * 4, true));
                break;
            case DataType.INT32:
                out.push(dv.getInt32(i * 4, true));
                break;
            case DataType.UINT32:
                out.push(dv.getUint32(i * 4, true));
                break;
            case DataType.INT64:
                out.push(Number(dv.getBigInt64(i * 8, true)));
                break;
            case DataType.DOUBLE:
                out.push(dv.getFloat64(i * 8, true));
                break;
            case DataType.UINT64:
                out.push(Number(dv.getBigUint64(i * 8, true)));
                break;
            case DataType.INT16:
                out.push(dv.getInt16(i * 2, true));
                break;
            case DataType.UINT16:
                out.push(dv.getUint16(i * 2, true));
                break;
            case DataType.FLOAT16:
                out.push(decodeFloat16(dv.getUint16(i * 2, true)));
                break;
            case DataType.BFLOAT16:
                out.push(dv.getUint16(i * 2, true));
                break;
            case DataType.UINT8:
                out.push(u8[i]);
                break;
            case DataType.INT8:
                out.push(new Int8Array(u8.buffer, u8.byteOffset)[i]);
                break;
            case DataType.BOOL:
                out.push(u8[i] ? 1 : 0);
                break;
            default:
                return undefined;
        }
    }

    return out;
}

export function resolveAutoPads(
    autoPad: string,
    inputSpatial: number[],
    kernelShape: number[],
    strides: number[],
    dilations: number[],
    explicitPads?: number[],
): number[] {
    const k = inputSpatial.length;
    if (!autoPad || autoPad === "NOTSET") {
        return explicitPads && explicitPads.length === 2 * k
            ? explicitPads.map(Number)
            : new Array<number>(2 * k).fill(0);
    }
    if (autoPad === "VALID") {
        return new Array<number>(2 * k).fill(0);
    }
    const padHead = new Array<number>(k).fill(0);
    const padTail = new Array<number>(k).fill(0);
    for (let d = 0; d < k; d++) {
        const stride = strides[d] ?? 1;
        const dil = dilations[d] ?? 1;
        const effK = (kernelShape[d] - 1) * dil + 1;
        const outDim = Math.ceil(inputSpatial[d] / stride);
        const totalPad = Math.max(0, (outDim - 1) * stride + effK - inputSpatial[d]);
        if (autoPad === "SAME_LOWER") {
            padHead[d] = Math.ceil(totalPad / 2);
            padTail[d] = Math.floor(totalPad / 2);
        } else {
            // SAME_UPPER
            padHead[d] = Math.floor(totalPad / 2);
            padTail[d] = Math.ceil(totalPad / 2);
        }
    }
    return [...padHead, ...padTail];
}

/**
 * Builds canonical Conv-compatible attributes from an op with auto_pad/pads,
 * resolving auto_pad into explicit `pads` whenever spatial and kernel dimensions
 * are statically known (or for NOTSET/VALID), and avoiding conflicting `auto_pad`
 * + `pads` attributes on the lowered node.
 */
export function buildResolvedConvAttrs(
    op: OperationNode.Class,
    inputSpatial: number[],
    kernelShape: number[],
    options?: { includeGroup?: boolean; fixedGroup?: number },
): AttributeMap {
    const attrs =
        (
            op as {
                getAttributes?: () => Record<string, unknown>;
                attributes?: Record<string, unknown>;
            }
        ).getAttributes?.() ??
        (op as { attributes?: Record<string, unknown> }).attributes ??
        {};

    const spatialRank = inputSpatial.length;
    const autoPad = typeof attrs["auto_pad"] === "string" ? attrs["auto_pad"] : "NOTSET";
    const explicitPads = Array.isArray(attrs["pads"]) ? (attrs["pads"] as number[]) : undefined;
    const strides = Array.isArray(attrs["strides"])
        ? (attrs["strides"] as number[]).map(Number)
        : new Array<number>(spatialRank).fill(1);
    const dilations = Array.isArray(attrs["dilations"])
        ? (attrs["dilations"] as number[]).map(Number)
        : new Array<number>(spatialRank).fill(1);

    const convAttrs: AttributeMap = {};
    if (options?.fixedGroup !== undefined) {
        convAttrs["group"] = options.fixedGroup;
    } else if (options?.includeGroup === true && attrs["group"] !== undefined) {
        convAttrs["group"] = attrs["group"] as AttributeMap[string];
    }
    if (attrs["dilations"] !== undefined) convAttrs["dilations"] = dilations;
    if (attrs["kernel_shape"] !== undefined) {
        convAttrs["kernel_shape"] = (attrs["kernel_shape"] as number[]).map(Number);
    }
    if (attrs["strides"] !== undefined) convAttrs["strides"] = strides;

    const hasStaticSpatial =
        spatialRank > 0 &&
        inputSpatial.every((d) => d > 0) &&
        kernelShape.length === spatialRank &&
        kernelShape.every((k) => k > 0);

    if (autoPad === "NOTSET") {
        if (explicitPads !== undefined) {
            convAttrs["pads"] =
                spatialRank > 0
                    ? resolveAutoPads(
                          "NOTSET",
                          inputSpatial,
                          kernelShape,
                          strides,
                          dilations,
                          explicitPads,
                      )
                    : explicitPads.map(Number);
        }
    } else if (autoPad === "VALID") {
        if (spatialRank > 0) {
            convAttrs["pads"] = resolveAutoPads(
                "VALID",
                inputSpatial,
                kernelShape,
                strides,
                dilations,
            );
        } else {
            convAttrs["auto_pad"] = "VALID";
        }
    } else if (hasStaticSpatial) {
        convAttrs["pads"] = resolveAutoPads(
            autoPad,
            inputSpatial,
            kernelShape,
            strides,
            dilations,
            explicitPads,
        );
    } else {
        convAttrs["auto_pad"] = autoPad;
    }

    return convAttrs;
}
