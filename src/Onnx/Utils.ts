import OnnxEdge from "./OnnxEdge.js";
import OnnxGraph from "./OnnxGraph.js";
import { TensorProto, DataType } from "./OnnxTypes.js";
import OperationNode from "./OperationNode.js";
import TensorNode from "./TensorNode.js";


export type Dim = number | String;
export type Shape = Dim[];

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
export function decodeIntegerVectorFromTensorProto(tv: AnyTensorProto): number[] | undefined {
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

export function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}

export function toArrayLike<T = any>(nc: any): T[] {
  return nc?.toArray?.() ?? nc ?? [];
}

export function scalarOfType(
  g: OnnxGraph.Class,
  name: string,
  v: number,
  dtype: DataType
): TensorNode.Class {
  const proto = makeTensorProto(dtype, [], [v]);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, [], "constant", proto))
    .as(TensorNode);
}

export function tensorOnesConst(
  g: OnnxGraph.Class,
  name: string,
  dtype: DataType,
  shape: number[]
): TensorNode.Class {
  const size = shape.reduce((a, b) => a * b, 1);
  const ones = new Array<number>(size).fill(1);
  const proto = makeTensorProto(dtype, shape, ones);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, shape, "constant", proto))
    .as(TensorNode);
}

export function addEdge(
  g: OnnxGraph.Class,
  srcOp: OperationNode.Class,
  dstTensor: TensorNode.Class,
  dtype: DataType,
  shape?: Array<number | String | undefined>
) {
  g.addEdge(srcOp, dstTensor)
    .init(new OnnxEdge.Builder(dtype, shape ?? dstTensor.shape))
    .as(OnnxEdge);
}

export function removeInitializerByName(g: OnnxGraph.Class, name?: string) {
  if (!name) return;
  const anyG: any = g as any;
  const model = anyG?.rawModel ?? anyG?.model;
  const graph = model?.graph ?? anyG?.graph;
  if (!graph) return;
  for (const f of ["initializer","sparse_initializer","input","value_info"]) {
    if (Array.isArray(graph[f])) graph[f] = graph[f].filter((x: any) => x?.name !== name);
  }
}

export function maybeRemoveOrphanConstant(g: OnnxGraph.Class, tn?: TensorNode.Class) {
  if (!tn) return;
  const isConstLike =
    (tn as any).type === "constant" ||
    (tn as any).constantValue != null ||
    (tn as any).originalInitializer != null ||
    (tn as any).initializer != null;
  if (!isConstLike) return;
  const consumers = toArrayLike<OperationNode.Class>(tn.getOutgoers?.targets?.filterIs?.(OperationNode));
  if (consumers.length > 0) return;

  // remove upstream Constant op if it's now orphan
  const srcOps = toArrayLike<OperationNode.Class>(tn.getIncomers?.sources?.filterIs?.(OperationNode));
  for (const src of srcOps) {
    if (src.type !== "Constant") continue;
    const outs = toArrayLike<TensorNode.Class>(src.getOutgoers?.targets?.filterIs?.(TensorNode));
    const stillUsed = outs.some(t => toArrayLike(src.getOutgoers?.targets?.filterIs?.(OperationNode)).length > 0);
    if (!stillUsed) src.remove();
  }

  const onnxName = tn.id;
  tn.remove();
  removeInitializerByName(g, onnxName);
}

// rank-0 INT64 scalar constant
export function scalarI64(g: OnnxGraph.Class, name: string, v: number): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [], [v]);
  return g
    .addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [], "constant", proto))
    .as(TensorNode);
}

// rank-0 scalar of the given numeric dtype
export function scalarZeroOfType(
  g: OnnxGraph.Class,
  name: string,
  dtype: DataType
): TensorNode.Class {
  const proto = makeTensorProto(dtype, [], [0]);
  return g
    .addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, [], "constant", proto))
    .as(TensorNode);
}

// 1-D INT64 constant (axes helper too)
export function constI64(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
  return g
    .addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [vals.length], "constant", proto))
    .as(TensorNode);
}

export function constF32(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(
      DataType.FLOAT,
      [vals.length],
      "constant",
      makeTensorProto(DataType.FLOAT, [vals.length], vals))
    ).as(TensorNode);
}

export function isNumeric(dtype: DataType): boolean {
  // Guard: avoid STRING/BOOL tensors — this rewrite is numeric-only
  return !(
    dtype === (DataType as any).STRING ||
    dtype === (DataType as any).BOOL
  );
}

export function toNum(x: number | String | undefined): number | undefined {
  if (typeof x === "number") return x;
  if (typeof x === "string" && /^[0-9]+$/.test(x)) return Number(x);
  return undefined;
}

export function toNumShape(s?: Array<number | String | undefined>): Array<number | undefined> | undefined {
  if (!s) return undefined;
  return s.map(toNum);
}

// Look up tensors by original ONNX name (initializer/constant/id)
export function findTensorByOnnxName(g: OnnxGraph.Class, name?: string): TensorNode.Class | undefined {
  if (!name) return undefined;
  const pool = (g.getTensorNodes?.().toArray?.() ?? []) as TensorNode.Class[];

  let t = pool.find(n => (n as any).originalInitializer?.name === name);
  if (t) return t;

  t = pool.find(n => (n as any).constantValue?.name === name);
  if (t) return t;

  t = pool.find(n => n.id === name);
  if (t) return t;

  t = pool.find(n => (n as any).extraAttrs?.some?.((a: any) => a.name === "name" && a.s === name));
  return t;
}

export function findConstantProducerAsTensor(g: OnnxGraph.Class, onnxName?: string): TensorNode.Class | undefined {
  if (!onnxName) return undefined;
  const allOps = (g.getOperationNodes?.().toArray?.() ?? []) as OperationNode.Class[];
  for (const c of allOps) {
    if (c.type !== "Constant") continue;
    const outs: string[] =
      (c as any).outputs ??
      (c as any).output ??
      (c as any).getOutputNames?.() ?? [];
    if (outs[0] === onnxName) {
      const t = findTensorByOnnxName(g, onnxName);
      if (t) return t;
    }
  }
  return undefined;
}

/* Read an INT64 (or INT32/UINT64) vector from a TensorNode's constantValue/initializer/value. */
export function readConstIntegerVectorFromTensorNode(tn?: TensorNode.Class): number[] | undefined {
  if (!tn) return undefined;
  const tv: any =
    (tn as any).constantValue ??
    (tn as any).initializer ??
    (tn as any).value ??
    (tn as any).data;
  if (!tv) return undefined;

  // 1) direct int64Data
  if (Array.isArray(tv.int64Data) && tv.int64Data.length) {
    return tv.int64Data.map(Number);
  }
  // 2) common alternates
  if (Array.isArray(tv.int32Data) && tv.int32Data.length) {
    return tv.int32Data.map(Number);
  }
  if (Array.isArray(tv.uint64Data) && tv.uint64Data.length) {
    return tv.uint64Data.map((x: any) => Number(x));
  }

  // 3) rawData (Node Buffer or Uint8Array), little-endian
  const raw = (tv.rawData && (tv.rawData.data ?? tv.rawData)) as any;
  if (raw) {
    // Normalize to a Uint8Array view
    let u8: Uint8Array;
    if (raw instanceof Uint8Array) u8 = raw;
    else if (Buffer.isBuffer(raw)) u8 = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
    else if (Array.isArray(raw)) u8 = Uint8Array.from(raw);
    else return undefined;

    // Decide element width: prefer INT64 (8 bytes) when dataType==7 (ONNX INT64)
    const isI64 = tv.dataType === 7 /* TensorProto.DataType.INT64 */;
    const elemBytes = isI64 ? 8 : 4;
    const n =
      (Array.isArray(tv.dims) && tv.dims.length
        ? tv.dims.map((d: any) => Number(d)).reduce((a: number, b: number) => a * b, 1)
        : Math.floor(u8.byteLength / elemBytes));

    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      const off = i * elemBytes;
      if (isI64) out.push(Number(dv.getBigInt64(off, true)));     // little-endian int64
      else       out.push(dv.getInt32(off, true));                // fallback
    }
    return out;
  }

  return undefined;
}

export function readScalarFromTensorNode(tn?: TensorNode.Class): number | undefined {
  if (!tn) return undefined;
  const tv: any =
    (tn as any).constantValue ??
    (tn as any).originalInitializer ??
    (tn as any).initializer ??
    (tn as any).pads ??
    (tn as any).data;
  if (!tv) return undefined;

  if (Array.isArray(tv.floatData) && tv.floatData.length) return Number(tv.floatData[0]);
  if (Array.isArray(tv.doubleData) && tv.doubleData.length) return Number(tv.doubleData[0]);
  if (Array.isArray(tv.int64Data) && tv.int64Data.length) return Number(tv.int64Data[0]);
  if (Array.isArray(tv.int32Data) && tv.int32Data.length) return Number(tv.int32Data[0]);

  const raw = (tv.rawData && (tv.rawData.data ?? tv.rawData)) as any;
  if (raw) {
    let u8: Uint8Array;
    if (raw instanceof Uint8Array) u8 = raw;
    else if ((globalThis as any).Buffer?.isBuffer(raw)) u8 = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
    else if (Array.isArray(raw)) u8 = Uint8Array.from(raw);
    else return undefined;
    if (u8.byteLength === 8) {
      const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
      try { return Number(dv.getFloat64(0, true)); } catch { /* noop */ }
      try { return Number(dv.getBigInt64(0, true)); } catch { /* noop */ }
    } else if (u8.byteLength === 4) {
      const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
      const f = dv.getFloat32(0, true);
      if (Number.isFinite(f)) return Number(f);
      return dv.getInt32(0, true);
    }
  }
  return undefined;
}

export function makeValueScalar1(g: OnnxGraph.Class, name: string, dtype: DataType, v: number): TensorNode.Class {
  const proto = makeTensorProto(dtype, [1], [v]);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, [1], "constant", proto))
    .as(TensorNode);
}

export function makeI64ShapeConst(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [vals.length], "constant", proto))
    .as(TensorNode);
}

export function shapeOf(g: OnnxGraph.Class, x: TensorNode.Class, name: string): TensorNode.Class {
  const sop = g.addNode(uniq(g, `${name}_op`)).init(new OperationNode.Builder("Shape", [x], {})).as(OperationNode);
  const s = g.addNode(uniq(g, `${name}`)).init(new TensorNode.Builder(DataType.INT64, [x.shape.length], "intermediate")).as(TensorNode);
  addEdge(g, sop, s, DataType.INT64, [x.shape.length]);
  return s;
}

export function editShapeDim(
  g: OnnxGraph.Class,
  baseShape: TensorNode.Class,
  axis: number,
  size1D: TensorNode.Class,
  name: string
): TensorNode.Class {
  const idx = makeI64ShapeConst(g, `${name}_idx`, [axis]);
  const sc = g.addNode(uniq(g, `${name}_sc`)).init(new OperationNode.Builder("ScatterElements", [baseShape, idx, size1D], { axis: 0 })).as(OperationNode);
  const out = g.addNode(uniq(g, `${name}_out`)).init(new TensorNode.Builder(DataType.INT64, [baseShape.shape[0] as number], "intermediate")).as(TensorNode);
  addEdge(g, sc, out, DataType.INT64, [baseShape.shape[0] as number]);
  return out;
}

export function getSmallestRankShape(tensors: TensorNode.Class[]): Shape {
  if (tensors.length === 0) return [];

  let smallest = tensors[0].shape;
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].shape.length < smallest.length) {
      smallest = tensors[i].shape;
    }
  }
  return smallest;
}

export function getLargestRankShape(tensors: TensorNode.Class[]): Shape {
  if (tensors.length === 0) return [];
  let largest = tensors[0].shape;
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].shape.length > largest.length) {
      largest = tensors[i].shape;
    }
  }
  return largest;
}

export function makeTensorConst(
  g: OnnxGraph.Class, id: string, dataType: DataType,
  tensorKind: TensorNode.TensorKind, proto: TensorProto
) {
  const builder = tensorKind === "constant" ? new TensorNode.Builder(dataType, proto.dims!, tensorKind, proto) : new TensorNode.Builder(dataType, proto.dims!, tensorKind, undefined, proto);
  return g.addNode(uniq(g, id)).init(builder).as(TensorNode);
}

export function asStaticDims(shape: (number | string)[]): number[] {
  return shape.map(d => (typeof d === 'number' && d > 0) ? d : 1);
}

export function isNum(d: Dim): d is number {
  return typeof d === "number" && Number.isFinite(d);
}

// Convert (number|string)[] → number[]; unknown/symbolic become -1
export function toStaticShape(shape: Shape): number[] {
  return shape.map(d => (typeof d === "number" ? d : -1));
}

// Product of positive dims; treat -1 (unknown) as 1 for decoding purposes
export function prodSafe(dims: number[]): number {
  return dims.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

// Compute strides for a fully-known numeric shape
export function computeStrides(dims: number[]): number[] {
  const n = dims.length;
  const strides = new Array(n);
  let acc = 1;
  for (let i = n - 1; i >= 0; --i) {
    const d = dims[i] > 0 ? dims[i] : 1;  // unknown/0 -> 1
    strides[i] = acc;
    acc *= d;
  }
  return strides;
}

export function swap<A>(a: A[], i: number, j: number) { const x=a[i]; a[i]=a[j]; a[j]=x; return a; }

/**
 * Topologically sorts OperationNodes of a graph.
 *
 * Extended to account for implicit dependencies coming from subgraphs:
 * if an op's body/subgraph uses a tensor defined in the parent graph,
 * we treat the producer of that tensor as a predecessor of the op.
 *
 * This lets us correctly order Loops/Ifs whose bodies read outer values
 * (implicit inputs), without requiring extra wiring changes in the
 * lowering passes.
 */
export function topologicalSortOperationNodes(
  graph: OnnxGraph.Class
): OperationNode.Class[] {
  const sorted: OperationNode.Class[] = [];
  const visited = new Set<string>();
  const temp = new Set<string>();

  const opNodes = graph.getOperationNodes().toArray();

  // Map tensor id -> producing op (in this graph)
  const tensorProducers = new Map<string, OperationNode.Class>();
  for (const op of opNodes) {
    const outTensors =
      op.getOutgoers?.targets?.filter(n => n.is(TensorNode)).toArray?.() ?? [];
    for (const t of outTensors as TensorNode.Class[]) {
      tensorProducers.set(t.id, op);
    }
  }

  // Extra deps: op.id -> set of predecessor ops (from subgraph implicit inputs)
  const extraDeps = new Map<string, Set<OperationNode.Class>>();

  // Discover implicit dependencies from subgraphs (Loop/If/etc bodies)
  for (const op of opNodes) {
    const subgraphs: Record<string, OnnxGraph.Class> = {
      ...(op.getSubgraphs?.() ?? {}),
    };

    const body = op.getBodySubgraph?.();
    if (body) {
      subgraphs["__body"] = body;
    }

    for (const key of Object.keys(subgraphs)) {
      const sg = subgraphs[key];
      if (!sg) continue;

      const sgOps = sg.getOperationNodes().toArray();

      for (const bOp of sgOps) {
        // Use getInputs() if available; fallback to incomers if needed
        const inputs = (bOp.getInputs?.() ?? []) as any[];

        for (const inp of inputs) {
          const t = inp?.tryAs?.(TensorNode);
          if (!t) continue;

          // If tensor lives in parent graph but not in this subgraph,
          // it's an implicit/outer value for this op.
          const isFromParent = graph.hasNode(t.id);
          const isInSubgraph = sg.hasNode(t.id);

          if (isFromParent && !isInSubgraph) {
            const prod = tensorProducers.get(t.id);
            if (!prod || prod.id === op.id) continue;

            let deps = extraDeps.get(op.id);
            if (!deps) {
              deps = new Set<OperationNode.Class>();
              extraDeps.set(op.id, deps);
            }
            deps.add(prod);
          }
        }
      }
    }
  }

  const visit = (node: OperationNode.Class) => {
    if (visited.has(node.id) || !graph.hasNode(node.id)) return;

    if (temp.has(node.id)) {
      console.warn(`[TopoSort] Cycle or back-edge detected at node: ${node.id}`);
      return;
    }

    temp.add(node.id);

    // 1) Enforce extra deps from subgraphs (implicit inputs)
    const implicitPreds = extraDeps.get(node.id);
    if (implicitPreds) {
      for (const pred of implicitPreds) {
        visit(pred);
      }
    }

    // 2) Existing predecessor logic (within this graph)
    const checkPred = (n: TensorNode.Class | OperationNode.Class) => {
      if (n.is(OperationNode)) {
        const op = n.as(OperationNode);
        // Follow intermediate inputs
        for (const input of op.getInputs?.() ?? []) {
          if (!input) continue;
          const t = input.tryAs?.(TensorNode);
          if (t && t.type === "intermediate") {
            checkPred(t);
          }
        }
      }

      const incomers = n.incomers?.toArray?.() ?? [];
      for (const edge of incomers) {
        const src = edge?.source;
        if (!src) continue;

        if (src.is?.(OperationNode)) {
          const pred = src.as(OperationNode);
          visit(pred);
        } else if (src.is?.(TensorNode)) {
          const tensorPred = src.as(TensorNode);
          if (tensorPred?.type === "intermediate") {
            checkPred(tensorPred);
          }
        }
      }
    };

    checkPred(node);

    temp.delete(node.id);
    visited.add(node.id);
    sorted.push(node);
  };

  for (const node of opNodes) {
    visit(node);
  }

  return sorted;
}

export function normalizeAxis(axis: number, rank: number): number {
  if (rank <= 0) return axis;
  return ((axis % rank) + rank) % rank;
}

export function broadcastTwoShapes(a: number[], b: number[]): number[] {
  const ra = a.length, rb = b.length;
  const r = Math.max(ra, rb);
  const out = new Array<number>(r);
  for (let i = 0; i < r; i++) {
    const da = a[ra - 1 - i] ?? 1;
    const db = b[rb - 1 - i] ?? 1;
    if (da === 1) out[r - 1 - i] = db;
    else if (db === 1) out[r - 1 - i] = da;
    else if (da === db) out[r - 1 - i] = da;
    else {
      // Keep your lax behavior
      console.warn(`Broadcast mismatch at dim ${r - 1 - i}: ${da} vs ${db}. Guessing max.`);
      out[r - 1 - i] = Math.max(da, db);
    }
  }
  return out;
}

export function broadcastShapes(...shapes: number[][]): number[] {
  return shapes.reduce((acc, s) => broadcastTwoShapes(acc, s), []);
}

export function getAttr(node: any, name: string, def?: any) {
  const v = node.getAttributes?.[name];
  return v === undefined ? def : v;
}

export function inferPoolDim(inDim: number, k: number, stride: number, padHead: number, padTail: number, dil: number) {
  // ONNX: floor((in + padHead + padTail - dil*(k-1) - 1)/stride + 1)
  const effectiveK = dil * (k - 1) + 1;
  return Math.floor((inDim + padHead + padTail - effectiveK) / stride + 1);
}

export function inferConvDim(inDim: number, k: number, stride: number, padHead: number, padTail: number, dil: number) {
  // Same as pooling
  return inferPoolDim(inDim, k, stride, padHead, padTail, dil);
}

export function dbg(...args: any[]): void {
  console.log("[loop-debug]", ...args);
}

export function dbgTensor(label: string, t: TensorNode.Class | null | undefined): void {
  if (!t) return;
  dbg(label, {
    id: t.id,
    kind: t.type,
    elemType: t.literalType,
    shape: t.shape,
  });
}
