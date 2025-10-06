import { DataType, TensorProto } from "../OnnxTypes.js";

export const typeSizeMap: Record<number, number> = {
    0: 0,    // onnx.TensorProto.UNDEFINED
    1: 4,    // onnx.TensorProto.FLOAT
    2: 1,    // onnx.TensorProto.UINT8
    3: 1,    // onnx.TensorProto.INT8
    4: 2,    // onnx.TensorProto.UINT16
    5: 2,    // onnx.TensorProto.INT16
    6: 4,    // onnx.TensorProto.INT32
    7: 8,    // onnx.TensorProto.INT64
    8: -1,   // onnx.TensorProto.STRING (Variable size)
    9: 1,    // onnx.TensorProto.BOOL
    10: 2,   // onnx.TensorProto.FLOAT16
    11: 8,   // onnx.TensorProto.DOUBLE
    12: 4,   // onnx.TensorProto.UINT32
    13: 8,   // onnx.TensorProto.UINT64
    14: 8,   // onnx.TensorProto.COMPLEX64
    15: 16,  // onnx.TensorProto.COMPLEX128
    16: 2,   // onnx.TensorProto.BFLOAT16
    17: 1,   // onnx.TensorProto.FLOAT8E4M3FN
    18: 1,   // onnx.TensorProto.FLOAT8E4M3FNUZ
    19: 2,   // onnx.TensorProto.FLOAT8E5M2
    20: 2,   // onnx.TensorProto.FLOAT8E5M2FNUZ
    21: 1,   // onnx.TensorProto.UINT4
    22: 1    // onnx.TensorProto.INT4
};

export type AnyTensorProto = {
  dataType?: number | string;
  dims?: (number | string)[];
  rawData?: { data?: Uint8Array | number[] | Buffer } | Uint8Array | number[] | Buffer;
  int64Data?: (number | bigint)[];
  int32Data?: number[];
  uint64Data?: number[];
  floatData?: number[];
  doubleData?: number[];
};

export function formatId(name : string, nodeId : string) : string {
    return `${name}_${nodeId}`;
}

/* boolean tensor                                                       */
export function bool(v: boolean): TensorProto {
  return { dataType: DataType.BOOL, dims: [], int32Data: [v ? 1 : 0] };
}

/* scalar INT64 tensor                                                       */
export function scalarInt32(v: number): TensorProto {
  return { dataType: DataType.INT32, dims: [], int32Data: [Number(v)] };
}

/* 1-D INT64 tensor                                                          */
export function int32Vec(arr: (number)[]): TensorProto {
  return { dataType: DataType.INT32, dims: [arr.length], int32Data: arr.map(Number) };
}

/* scalar INT64 tensor                                                       */
export function scalarInt64(v: number): TensorProto {
  return { dataType: DataType.INT64, dims: [], int64Data: [Number(v)] };
}

/* 1-D INT64 tensor                                                          */
export function int64Vec(arr: (number)[]): TensorProto {
  return { dataType: DataType.INT64, dims: [arr.length], int64Data: arr.map(Number) };
}

/* scalar float tensor                                                       */
export function scalarFloat(v: number): TensorProto {
  return { dataType: DataType.FLOAT, dims: [], floatData: [Number(v)] };
}

/* 1-D float tensor                                                          */
export function floatVec(arr: number[]): TensorProto {
  return { dataType: DataType.FLOAT, dims: [arr.length], floatData: arr.map(Number) };
}

/* zero tensor that matches elemType + shape                                 */
export function zeroTensor(elemType: DataType, shape: number[]): TensorProto {
  const n = shape.reduce((a, b) => a * b, 1);
  switch (elemType) {
    case DataType.FLOAT:
    case DataType.DOUBLE:
      return { dataType: elemType, dims: shape, floatData: Array(n).fill(0) };
    case DataType.INT32:
      return { dataType: elemType, dims: shape, int32Data: Array(n).fill(0) };
    default:
      return { dataType: DataType.INT64, dims: shape, int64Data: Array(n).fill(Number(0)) };
  }
}

// Build a minimal ONNX-compatible TensorProto for numeric data
export function makeTensorProto(dtype: DataType, dims: number[], values: number[]): TensorProto {
  const t: TensorProto = { dataType: dtype, dims };

  switch (dtype) {
    case DataType.FLOAT:   t.floatData  = values; break;
    case DataType.DOUBLE:  t.doubleData = values; break;
    case DataType.INT32:   t.int32Data  = values.map(v => (v|0)); break;
    case DataType.INT64:   t.int64Data  = values.map(v => Number(v)); break;
    case DataType.UINT64:  t.uint64Data = values.map(v => Number(v)); break; // if you actually need u64
    // add other dtypes here if your graphs require them
    default:
      // Fallback: encode as raw little-endian 32-bit floats
      const buf = Buffer.alloc(values.length * 4);
      const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
      values.forEach((x, i) => dv.setFloat32(i * 4, x, true));
      t.rawData = { type: "Buffer", data: Array.from(buf) };
      break;
  }
  return t;
}

export function toU8(raw: any): Uint8Array | undefined {
  if (!raw) return undefined;
  if (raw instanceof Uint8Array) return raw;
  if (Array.isArray(raw)) return Uint8Array.from(raw);
  if ((globalThis as any).Buffer?.isBuffer(raw)) {
    const b: Buffer = raw as any;
    return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  }
  // handle { data: ... }
  const inner = (raw.data ?? undefined);
  if (inner) return toU8(inner);
  return undefined;
}

export function totalSizeFromDims(fallbackElems: number, dims?: (number | string)[] | undefined): number {
  if (!Array.isArray(dims) || dims.length === 0) return fallbackElems;
  return dims.map(d => Number(d)).reduce((a, b) => a * b, 1);
}

export function isInt64Type(dt: number | string | undefined): boolean {
  // 7 is ONNX INT64; some wrappers store strings like "INT64"
  return dt === 7 || dt === "INT64";
}

/** Decode an integer vector (pref INT64, else INT32) from a TensorProto-like object. */
export function decodeIntegerVectorFromTensorProto(tv: AnyTensorProto | undefined): number[] | undefined {
  if (!tv) return undefined;

  // fast paths
  if (Array.isArray(tv.int64Data) && tv.int64Data.length) return tv.int64Data.map(Number);
  if (Array.isArray(tv.int32Data) && tv.int32Data.length) return tv.int32Data.map(Number);
  if (Array.isArray(tv.uint64Data) && tv.uint64Data.length) return tv.uint64Data.map(Number);

  // rawData path
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

