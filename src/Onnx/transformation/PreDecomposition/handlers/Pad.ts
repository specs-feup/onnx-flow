import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, AnyTensorProto, decodeIntegerVectorFromTensorProto } from "../../Utilities.js";

// ---------- small utils ----------
function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}

function toArrayLike<T = any>(nc: any): T[] {
  return nc?.toArray?.() ?? nc ?? [];
}

function removeInitializerByName(g: OnnxGraph.Class, name?: string) {
  if (!name) return;
  const anyG: any = g as any;
  const model = anyG?.rawModel ?? anyG?.model;
  const graph = model?.graph ?? anyG?.graph;
  if (!graph) return;

  const fields = [
    "initializer",          // TensorProto[]
    "sparse_initializer",   // Optional
    "input",                // ValueInfoProto[] (graph inputs)
    "value_info"            // ValueInfoProto[] (misc)
  ];

  for (const f of fields) {
    if (Array.isArray(graph[f])) {
      graph[f] = graph[f].filter((x: any) => x?.name !== name);
    }
  }
}


function readPadsVectorFromTensorInput(
  g: OnnxGraph.Class,
  tn?: TensorNode.Class
): number[] | undefined {
  if (!tn) return undefined;

  // 1) read off the tensor node itself
  const tv1: AnyTensorProto | undefined =
    (tn as any).constantValue ??
    (tn as any).originalInitializer ??
    (tn as any).initializer ??
    (tn as any).value ??
    (tn as any).data;
  let vec = decodeIntegerVectorFromTensorProto(tv1);
  if (vec && vec.length) return vec;

  // 2) fallback: upstream Constant op
  const srcOpsNC = tn.getIncomers?.sources?.filterIs?.(OperationNode);
  const srcOps = toArrayLike<OperationNode.Class>(srcOpsNC);
  const constOp = srcOps.find(op => op.type === "Constant");
  if (constOp) {
    const valueAttr: AnyTensorProto | undefined =
      (constOp as any).getAttribute?.("value") ??
      ((constOp as any).getAttributes?.() ?? {})["value"];
    vec = decodeIntegerVectorFromTensorProto(valueAttr);
    if (vec && vec.length) return vec;
  }

  // 3) fallback: graph initializer with matching name (common for Pad)
  const name = (tn as any).extraAttrs?.onnxName ?? (tn as any).name ?? (tn as any).id ?? (tn as any).getName?.();
  const anyG: any = g as any;

  // try common places the raw ONNX model is stored
  const initLists: any[][] = [
    anyG?.rawModel?.graph?.initializer,
    anyG?.model?.graph?.initializer,
    anyG?.graph?.initializer,
  ].filter(Boolean);

  for (const list of initLists) {
    const found = Array.isArray(list) ? list.find((t: any) => t?.name === name) : undefined;
    if (found) {
      vec = decodeIntegerVectorFromTensorProto(found);
      if (vec && vec.length) return vec;
    }
  }

  return undefined;
}

// Read a scalar numeric (float/int) from a TensorNode constant
function readScalarFromTensorNode(tn?: TensorNode.Class): number | undefined {
  if (!tn) return undefined;
  const tv: any =
    (tn as any).constantValue ??
    (tn as any).originalInitializer ??   
    (tn as any).initializer ??           // (optional, for old paths)
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
      // try float32 then int32
      const f = dv.getFloat32(0, true);
      if (Number.isFinite(f)) return Number(f);
      return dv.getInt32(0, true);
    }
  }
  return undefined;
}

// Make a 1-D length-1 constant tensor for ConstantOfShape.value (ORT prefers rank-1)
function makeValueScalar1(g: OnnxGraph.Class, name: string, dtype: DataType, v: number): TensorNode.Class {
  const proto = makeTensorProto(dtype, [1], [v]);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, [1], "constant", proto))
    .as(TensorNode);
}

// Make a constant INT64 shape vector (e.g., pads seed shape)
function makeI64ShapeConst(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [vals.length], "constant", proto))
    .as(TensorNode);
}

function maybeRemoveOrphanConstant(g: OnnxGraph.Class, tn?: TensorNode.Class) {
  if (!tn) return;
  const isConstLike =
    (tn as any).type === "constant" ||
    (tn as any).constantValue != null ||
    (tn as any).originalInitializer != null ||
    (tn as any).initializer != null;

  if (!isConstLike) return;

  const consumersNC = tn.getOutgoers?.targets?.filterIs?.(OperationNode);
  const consumers = toArrayLike<OperationNode.Class>(consumersNC);
  if (consumers.length > 0) return;

  // If produced by a Constant op and it's orphan, remove that op too
  const srcOpsNC = tn.getIncomers?.sources?.filterIs?.(OperationNode);
  const srcOps = toArrayLike<OperationNode.Class>(srcOpsNC);
  for (const src of srcOps) {
    if (src.type !== "Constant") continue;
    const outsNC = src.getOutgoers?.targets?.filterIs?.(TensorNode);
    const outs = toArrayLike<TensorNode.Class>(outsNC);
    const stillUsed = outs.some(t => {
      const consNC2 = t.getOutgoers?.targets?.filterIs?.(OperationNode);
      const cons2 = toArrayLike<OperationNode.Class>(consNC2);
      return cons2.length > 0;
    });
    if (!stillUsed) src.remove();
  }

  function getOnnxName(tn?: TensorNode.Class): string | undefined {
    if (!tn) return undefined;
    const a: any = tn as any;
    return a.extraAttrs?.onnxName ?? a.name ?? a.id ?? a.getName?.();
  }

  // Remove the TensorNode from the graph
  const onnxName = getOnnxName(tn);
  tn.remove();

  // Proactively scrub the model’s initializer / inputs / value_info entries
  removeInitializerByName(g, onnxName);
}

// ---------- main handler ----------
export default function padHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "Pad") return false;

  const ins = op.getInputs?.() ?? [];
  const Xn = ins[0];
  if (!Xn?.is?.(TensorNode)) return false;

  const Xin = Xn.as(TensorNode);
  const inShape = Xin.shape.map(d => (typeof d === "number" ? d : 1));
  const rank = inShape.length;

  // pads: prefer input[1] constant; fallback to attribute "pads" if present
  let pads: number[] | undefined = undefined;
  const padsNode = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  if (padsNode) pads = readPadsVectorFromTensorInput(g, padsNode);

  if (!pads) {
    const a = (op as any).getAttributes?.() ?? (op as any)["attributes"] ?? {};
    if (Array.isArray(a.pads)) pads = a.pads.map((x: any) => Number(x));
  }

  const padsLen = pads?.length ?? -1;
  if (!pads || pads.length !== 2 * rank) return false;

  // reject negative pads (crop) for now
  if (pads.some(p => p < 0)) return false;

  // constant_value: input[2] scalar, else default 0
  let padValue = 0;
  if (ins[2]?.is?.(TensorNode)) {
    const s = readScalarFromTensorNode(ins[2].as(TensorNode));
    if (typeof s === "number" && Number.isFinite(s)) padValue = s;
  }

  // Output tensor Y (reuse original)
  const outsNC = op.getOutgoers?.targets?.filterIs?.(TensorNode);
  const outs = toArrayLike<TensorNode.Class>(outsNC);
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // dtype for pad slabs must match X dtype
  const dtype = Xin.literalType as DataType;

  let cur = Xin;

  // Process each axis in ascending order
  for (let ax = 0; ax < rank; ax++) {
    const pBeg = pads[ax];
    const pEnd = pads[ax + rank];
    if (pBeg === 0 && pEnd === 0) continue;

    const ensurePadSlab = (size: number, suffix: "B" | "E") => {
      // 1) Get dynamic shape of current tensor
      const shp = g.addNode(uniq(g, `pad_shape_${op.id}_${ax}_${suffix}`))
        .init(new OperationNode.Builder("Shape", [cur], {}))
        .as(OperationNode);
      const shpT = g.addNode(uniq(g, `pad_shapeT_${op.id}_${ax}_${suffix}`))
        .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
        .as(TensorNode);
      g.addEdge(shp, shpT).init(new OnnxEdge.Builder(shpT.literalType, shpT.shape)).as(OnnxEdge);

      // 2) Create indices=[ax] and updates=[size] (both INT64, shape [1])
      const idx = makeI64ShapeConst(g, `pad_idx_${op.id}_${ax}_${suffix}`, [ax]);
      const upd = makeI64ShapeConst(g, `pad_upd_${op.id}_${ax}_${suffix}`, [size]);

      // 3) newShape = ScatterElements(shpT, idx, upd, axis=0)
      const sc = g.addNode(uniq(g, `pad_scatter_${op.id}_${ax}_${suffix}`))
        .init(new OperationNode.Builder("ScatterElements", [shpT, idx, upd], { axis: 0 }))
        .as(OperationNode);
      const newShapeT = g.addNode(uniq(g, `pad_newShapeT_${op.id}_${ax}_${suffix}`))
        .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
        .as(TensorNode);
      g.addEdge(sc, newShapeT).init(new OnnxEdge.Builder(DataType.INT64, [rank])).as(OnnxEdge);

      // 4) Make a constant TensorNode directly (avoids empty Constant.value attribute)
      const kT = g.addNode(uniq(g, `pad_kT_${op.id}_${ax}_${suffix}`))
        .init(new TensorNode.Builder(
          dtype,
          [1],
          "constant",
          makeTensorProto(dtype, [1], [padValue])
        ))
        .as(TensorNode);

      const exp = g.addNode(uniq(g, `pad_expand_${op.id}_${ax}_${suffix}`))
        .init(new OperationNode.Builder("Expand", [kT, newShapeT], {}))
        .as(OperationNode);

      const slab = g.addNode(uniq(g, `pad_slab_${op.id}_${ax}_${suffix}`))
        .init(new TensorNode.Builder(dtype, Array(rank).fill(undefined), "intermediate"))
        .as(TensorNode);
      g.addEdge(exp, slab).init(new OnnxEdge.Builder(dtype, slab.shape)).as(OnnxEdge);

      return slab;
    };

    // begin slab
    let left: TensorNode.Class | undefined;
    if (pBeg > 0) left = ensurePadSlab(pBeg, "B");

    // end slab
    let right: TensorNode.Class | undefined;
    if (pEnd > 0) right = ensurePadSlab(pEnd, "E");

    // Concat along this axis
    const parts: (TensorNode.Class)[] = [];
    if (left) parts.push(left);
    parts.push(cur);
    if (right) parts.push(right);

    const cc = g.addNode(uniq(g, `pad_concat_${op.id}_${ax}`))
      .init(new OperationNode.Builder("Concat", parts, { axis: ax }))
      .as(OperationNode);

    if (ax === rank - 1) {
      // last axis → output must be Y only
      g.addEdge(cc, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
      cur = Y;
    } else {
      const mid = g.addNode(uniq(g, `pad_mid_${op.id}_${ax}`))
        .init(new TensorNode.Builder(dtype, Array(rank).fill(undefined), "intermediate"))
        .as(TensorNode);
      g.addEdge(cc, mid).init(new OnnxEdge.Builder(dtype, mid.shape)).as(OnnxEdge);
      cur = mid;
    }
  }

  // Final wiring: if no axis changed, copy X→Y with Identity; else connect last concat to Y directly
  if (cur === Xin) {
    const id = g.addNode(uniq(g, `pad_id_${op.id}`))
      .init(new OperationNode.Builder("Identity", [Xin], {}))
      .as(OperationNode);
    g.addEdge(id, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
  }

  // remove original Pad op
  g.getNodeById(op.id).remove();

  // clean up constant inputs (pads, value)
  const padsTN = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  const valTN  = ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined;
  maybeRemoveOrphanConstant(g, padsTN);
  maybeRemoveOrphanConstant(g, valTN);

  return true;
}
