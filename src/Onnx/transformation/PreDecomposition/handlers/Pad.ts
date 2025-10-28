import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto, AnyTensorProto, decodeIntegerVectorFromTensorProto } from "../../Utilities.js";

/* ------------------------------- utils -------------------------------- */
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
  for (const f of ["initializer", "sparse_initializer", "input", "value_info"]) {
    if (Array.isArray(graph[f])) graph[f] = graph[f].filter((x: any) => x?.name !== name);
  }
}

function readPadsVectorFromTensorInput(
  g: OnnxGraph.Class,
  tn?: TensorNode.Class
): number[] | undefined {
  if (!tn) return undefined;
  const tv1: AnyTensorProto | undefined =
    (tn as any).constantValue ??
    (tn as any).originalInitializer ??
    (tn as any).initializer ??
    (tn as any).value ??
    (tn as any).data;
  let vec = decodeIntegerVectorFromTensorProto(tv1);
  if (vec && vec.length) return vec;

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

  const name = (tn as any).extraAttrs?.onnxName ?? (tn as any).name ?? (tn as any).id ?? (tn as any).getName?.();
  const anyG: any = g as any;
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

function readScalarFromTensorNode(tn?: TensorNode.Class): number | undefined {
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

function makeValueScalar1(g: OnnxGraph.Class, name: string, dtype: DataType, v: number): TensorNode.Class {
  const proto = makeTensorProto(dtype, [1], [v]);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(dtype, [1], "constant", proto))
    .as(TensorNode);
}

function makeI64ShapeConst(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [vals.length], "constant", proto))
    .as(TensorNode);
}

function scalarI64(g: OnnxGraph.Class, name: string, v: number): TensorNode.Class {
  const proto = makeTensorProto(DataType.INT64, [], [v]);
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [], "constant", proto))
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

  const onnxName = getOnnxName(tn);
  tn.remove();
  removeInitializerByName(g, onnxName);
}

/* ------------------------------ helpers ------------------------------- */
function addEdge(
  g: OnnxGraph.Class,
  srcOp: OperationNode.Class,
  dstTensor: TensorNode.Class,
  dtype: DataType,
  shape?: Array<number | String | undefined>
) {
  g.addEdge(srcOp, dstTensor).init(new OnnxEdge.Builder(dtype, shape ?? dstTensor.shape)).as(OnnxEdge);
}

function shapeOf(g: OnnxGraph.Class, x: TensorNode.Class, name: string): TensorNode.Class {
  const sop = g.addNode(uniq(g, `${name}_op`)).init(new OperationNode.Builder("Shape", [x], {})).as(OperationNode);
  const s = g.addNode(uniq(g, `${name}`)).init(new TensorNode.Builder(DataType.INT64, [x.shape.length], "intermediate")).as(TensorNode);
  addEdge(g, sop, s, DataType.INT64, [x.shape.length]);
  return s;
}

function editShapeDim(
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

function ensurePadSlabConst(
  g: OnnxGraph.Class,
  cur: TensorNode.Class,
  axis: number,
  size: number,
  dtype: DataType,
  padValue: number,
  tag: string
): TensorNode.Class {
  // shape(cur) with axis replaced by size
  const shp = shapeOf(g, cur, `pad_shape_${tag}`);
  const size1D = makeI64ShapeConst(g, `pad_size_${tag}`, [size]);
  const newShape = editShapeDim(g, shp, axis, size1D, `pad_shape_edit_${tag}`);

  const kT = makeValueScalar1(g, `pad_val_${tag}`, dtype, padValue);
  const exp = g.addNode(uniq(g, `pad_expand_${tag}`)).init(new OperationNode.Builder("Expand", [kT, newShape], {})).as(OperationNode);
  const slab = g.addNode(uniq(g, `pad_slab_${tag}`)).init(new TensorNode.Builder(dtype, Array(cur.shape.length).fill(undefined), "intermediate")).as(TensorNode);
  addEdge(g, exp, slab, dtype, slab.shape);
  return slab;
}

function ensureEdgeSlab(
  g: OnnxGraph.Class,
  cur: TensorNode.Class,
  axis: number,
  size: number,
  tag: string
): TensorNode.Class | undefined {
  if (size <= 0) return undefined;
  const rank = cur.shape.length;
  const shp = shapeOf(g, cur, `edge_shape_${tag}`);
  const axisIdx = makeI64ShapeConst(g, `edge_axis_${tag}`, [axis]);
  const gdim = g.addNode(uniq(g, `edge_gather_${tag}`)).init(new OperationNode.Builder("Gather", [shp, axisIdx], { axis: 0 })).as(OperationNode);
  const dim1D = g.addNode(uniq(g, `edge_dim1D_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  addEdge(g, gdim, dim1D, DataType.INT64, [1]);

  // Build Slice for single element (left: [0:1], right: [dim-1:dim])
  const zero1 = makeI64ShapeConst(g, `edge_zero_${tag}`, [0]);
  const one1 = makeI64ShapeConst(g, `edge_one_${tag}`, [1]);
  const axes1 = makeI64ShapeConst(g, `edge_axes_${tag}`, [axis]);

  let starts: TensorNode.Class; let ends: TensorNode.Class;
  if (tag.endsWith("L")) {
    starts = zero1;
    ends = one1;
  } else { // Right slab
    const start1 = g.addNode(uniq(g, `edge_start1_${tag}`)).init(new OperationNode.Builder("Sub", [dim1D, one1], {})).as(OperationNode);
    const start1T = g.addNode(uniq(g, `edge_start1T_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
    addEdge(g, start1, start1T, DataType.INT64, [1]);
    starts = start1T;
    ends = dim1D;
  }

  const sliceOp = g.addNode(uniq(g, `edge_slice_${tag}`)).init(new OperationNode.Builder("Slice", [cur, starts, ends, axes1], {})).as(OperationNode);
  const oneSlice = g.addNode(uniq(g, `edge_oneSlice_${tag}`)).init(new TensorNode.Builder(cur.literalType as DataType, Array(rank).fill(undefined), "intermediate")).as(TensorNode);
  addEdge(g, sliceOp, oneSlice, cur.literalType as DataType, oneSlice.shape);

  // Expand to desired size along axis
  const size1D = makeI64ShapeConst(g, `edge_size_${tag}`, [size]);
  const newShape = editShapeDim(g, shp, axis, size1D, `edge_shape_edit_${tag}`);
  const exp = g.addNode(uniq(g, `edge_expand_${tag}`)).init(new OperationNode.Builder("Expand", [oneSlice, newShape], {})).as(OperationNode);
  const slab = g.addNode(uniq(g, `edge_slab_${tag}`)).init(new TensorNode.Builder(cur.literalType as DataType, Array(rank).fill(undefined), "intermediate")).as(TensorNode);
  addEdge(g, exp, slab, cur.literalType as DataType, slab.shape);
  return slab;
}

function ensureReflectSlab(
  g: OnnxGraph.Class,
  cur: TensorNode.Class,
  axis: number,
  size: number,
  tag: string
): TensorNode.Class | undefined {
  if (size <= 0) return undefined;
  const rank = cur.shape.length;
  const shp = shapeOf(g, cur, `refl_shape_${tag}`);
  const axisIdx = makeI64ShapeConst(g, `refl_axis_${tag}`, [axis]);
  const gdim = g.addNode(uniq(g, `refl_gather_${tag}`)).init(new OperationNode.Builder("Gather", [shp, axisIdx], { axis: 0 })).as(OperationNode);
  const dim1D = g.addNode(uniq(g, `refl_dim1D_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  addEdge(g, gdim, dim1D, DataType.INT64, [1]);

  const one1 = makeI64ShapeConst(g, `refl_one_${tag}`, [1]);
  const twoSc = scalarI64(g, `refl_two_${tag}`, 2);
  const stepNeg1 = scalarI64(g, `refl_step_${tag}`, -1);

  let startSc: TensorNode.Class; let endSc: TensorNode.Class;
  if (tag.endsWith("L")) {
    // left: indices = [size, size-1, ..., 1]  -> Range(size, 0, -1)
    startSc = scalarI64(g, `refl_l_start_${tag}`, size);
    endSc = scalarI64(g, `refl_l_end_${tag}`, 0);
  } else {
    // right: indices = [S-2, S-3, ..., S-2-size+1]
    // start = dim - 2 ; endExclusive = start - size
    const start1 = g.addNode(uniq(g, `refl_r_s1_${tag}`)).init(new OperationNode.Builder("Sub", [dim1D, one1], {})).as(OperationNode); // dim-1 (1D)
    const start1T = g.addNode(uniq(g, `refl_r_s1T_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
    addEdge(g, start1, start1T, DataType.INT64, [1]);
    const start2 = g.addNode(uniq(g, `refl_r_s2_${tag}`)).init(new OperationNode.Builder("Sub", [start1T, one1], {})).as(OperationNode); // dim-2
    const start2T = g.addNode(uniq(g, `refl_r_s2T_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
    addEdge(g, start2, start2T, DataType.INT64, [1]);

    const size1Sc = scalarI64(g, `refl_r_size_${tag}`, size);
    const endOp = g.addNode(uniq(g, `refl_r_end_${tag}`)).init(new OperationNode.Builder("Sub", [start2T, size1Sc], {})).as(OperationNode);
    const endT = g.addNode(uniq(g, `refl_r_endT_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [], "intermediate")).as(TensorNode);
    addEdge(g, endOp, endT, DataType.INT64, []);

    startSc = start2T; // scalar in [1] actually, but Range allows scalars
    endSc = endT;      // scalar
  }

  const rangeOp = g.addNode(uniq(g, `refl_range_${tag}`)).init(new OperationNode.Builder("Range", [startSc, endSc, stepNeg1], {})).as(OperationNode);
  const idx = g.addNode(uniq(g, `refl_idx_${tag}`)).init(new TensorNode.Builder(DataType.INT64, [undefined as any], "intermediate")).as(TensorNode);
  addEdge(g, rangeOp, idx, DataType.INT64);

  const gatherOp = g.addNode(uniq(g, `refl_gather_data_${tag}`)).init(new OperationNode.Builder("Gather", [cur, idx], { axis })).as(OperationNode);
  const slab = g.addNode(uniq(g, `refl_slab_${tag}`)).init(new TensorNode.Builder(cur.literalType as DataType, Array(rank).fill(undefined), "intermediate")).as(TensorNode);
  addEdge(g, gatherOp, slab, cur.literalType as DataType, slab.shape);
  return slab;
}

/* ------------------------------ handler ------------------------------- */
export default function padHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "Pad") return false;

  const ins = op.getInputs?.() ?? [];
  const Xn = ins[0];
  if (!Xn?.is?.(TensorNode)) return false;

  const Xin = Xn.as(TensorNode);
  const rank = Xin.shape.length;

  // read pads (constant only)
  let pads: number[] | undefined = undefined;
  const padsNode = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  if (padsNode) pads = readPadsVectorFromTensorInput(g, padsNode);
  if (!pads) {
    const a = (op as any).getAttributes?.() ?? (op as any)["attributes"] ?? {};
    if (Array.isArray(a.pads)) pads = a.pads.map((x: any) => Number(x));
  }
  if (!pads || pads.length !== 2 * rank) return false;

  // mode
  const attr = (op as any).getAttributes?.() ?? (op as any)["attributes"] ?? {};
  const modeRaw = String(attr.mode ?? "constant").toLowerCase();
  const mode: "constant" | "edge" | "reflect" = (modeRaw === "edge" || modeRaw === "reflect") ? (modeRaw as any) : "constant";

  // pad value (only used in constant)
  let padValue = 0;
  if (ins[2]?.is?.(TensorNode)) {
    const s = readScalarFromTensorNode(ins[2].as(TensorNode));
    if (typeof s === "number" && Number.isFinite(s)) padValue = s;
  }

  // Output Y
  const outsNC = op.getOutgoers?.targets?.filterIs?.(TensorNode);
  const outs = toArrayLike<TensorNode.Class>(outsNC);
  if (outs.length !== 1) return false;
  const Y = outs[0];
  const dtype = Xin.literalType as DataType;

  // Prepare working tensor (may be cropped if negative pads)
  let cur = Xin;

  // Negative pads => crop via Slice, then zero-out those pads in the array
  const beg = pads.slice(0, rank);
  const end = pads.slice(rank);
  for (let ax = 0; ax < rank; ax++) {
    const negB = Math.max(0, -beg[ax]);
    const negE = Math.max(0, -end[ax]);
    if (negB === 0 && negE === 0) continue;

    const shp = shapeOf(g, cur, `pad_crop_shape_${op.id}_${ax}`);
    const axVec = makeI64ShapeConst(g, `pad_crop_axes_${op.id}_${ax}`, [ax]);
    const start1 = makeI64ShapeConst(g, `pad_crop_start_${op.id}_${ax}`, [negB]);

    // end = dim - negE (as [1])
    const gdim = g.addNode(uniq(g, `pad_crop_gdim_${op.id}_${ax}`)).init(new OperationNode.Builder("Gather", [shp, axVec], { axis: 0 })).as(OperationNode);
    const dim1D = g.addNode(uniq(g, `pad_crop_dim1D_${op.id}_${ax}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
    addEdge(g, gdim, dim1D, DataType.INT64, [1]);
    const negE1 = makeI64ShapeConst(g, `pad_crop_nege_${op.id}_${ax}`, [negE]);
    const endOp = g.addNode(uniq(g, `pad_crop_endop_${op.id}_${ax}`)).init(new OperationNode.Builder("Sub", [dim1D, negE1], {})).as(OperationNode);
    const end1 = g.addNode(uniq(g, `pad_crop_end_${op.id}_${ax}`)).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
    addEdge(g, endOp, end1, DataType.INT64, [1]);

    const sliceOp = g.addNode(uniq(g, `pad_crop_slice_${op.id}_${ax}`)).init(new OperationNode.Builder("Slice", [cur, start1, end1, axVec], {})).as(OperationNode);
    const mid = g.addNode(uniq(g, `pad_crop_mid_${op.id}_${ax}`)).init(new TensorNode.Builder(dtype, Array(rank).fill(undefined), "intermediate")).as(TensorNode);
    addEdge(g, sliceOp, mid, dtype, mid.shape);
    cur = mid;

    // adjust remaining pad values to non-negative
    beg[ax] = Math.max(0, beg[ax]);
    end[ax] = Math.max(0, end[ax]);
  }

  // Now apply non-negative pads axis by axis
  for (let ax = 0; ax < rank; ax++) {
    const pBeg = beg[ax];
    const pEnd = end[ax];
    if (pBeg === 0 && pEnd === 0) continue;

    let left: TensorNode.Class | undefined;
    let right: TensorNode.Class | undefined;

    if (mode === "constant") {
      if (pBeg > 0) left = ensurePadSlabConst(g, cur, ax, pBeg, dtype, padValue, `${op.id}_${ax}_L`);
      if (pEnd > 0) right = ensurePadSlabConst(g, cur, ax, pEnd, dtype, padValue, `${op.id}_${ax}_R`);
    } else if (mode === "edge") {
      left = ensureEdgeSlab(g, cur, ax, pBeg, `${op.id}_${ax}_L`);
      right = ensureEdgeSlab(g, cur, ax, pEnd, `${op.id}_${ax}_R`);
    } else { // reflect
      left = ensureReflectSlab(g, cur, ax, pBeg, `${op.id}_${ax}_L`);
      right = ensureReflectSlab(g, cur, ax, pEnd, `${op.id}_${ax}_R`);
    }

    const parts: TensorNode.Class[] = [];
    if (left) parts.push(left);
    parts.push(cur);
    if (right) parts.push(right!);

    const cc = g.addNode(uniq(g, `pad_concat_${op.id}_${ax}`)).init(new OperationNode.Builder("Concat", parts, { axis: ax })).as(OperationNode);

    if (ax === rank - 1) {
      addEdge(g, cc, Y, dtype, Y.shape);
      cur = Y;
    } else {
      const mid = g.addNode(uniq(g, `pad_mid_${op.id}_${ax}`)).init(new TensorNode.Builder(dtype, Array(rank).fill(undefined), "intermediate")).as(TensorNode);
      addEdge(g, cc, mid, dtype, mid.shape);
      cur = mid;
    }
  }

  // If nothing changed, route X -> Y via Identity
  if (cur === Xin) {
    const id = g.addNode(uniq(g, `pad_identity_${op.id}`)).init(new OperationNode.Builder("Identity", [Xin], {})).as(OperationNode);
    addEdge(g, id, Y, dtype, Y.shape);
  }

  // Remove original Pad op
  g.getNodeById(op.id).remove();

  // Cleanup constant inputs
  const padsTN = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  const valTN  = ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined;
  maybeRemoveOrphanConstant(g, padsTN);
  maybeRemoveOrphanConstant(g, valTN);

  return true;
}
