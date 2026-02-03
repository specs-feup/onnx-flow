import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { DataType } from "../../../OnnxTypes.js";
import { AnyTensorProto, decodeIntegerVectorFromTensorProto, toArrayLike, shapeOf, makeI64ShapeConst, editShapeDim, makeValueScalar1, uniq, addEdge, scalarI64, readScalarFromTensorNode, maybeRemoveOrphanConstant } from "../../../Utils.js";

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
      constOp.attributes["value"] ??
      constOp.getAttributes?.() ?? {}["value"];
    vec = decodeIntegerVectorFromTensorProto(valueAttr);
    if (vec && vec.length) return vec;
  }

  const name = tn.id;
  const anyG = g as any;
  const initLists: any[][] = [
    anyG?.rawModel?.graph?.initializer,
    anyG?.model?.graph?.initializer,
    anyG?.graph?.initializer,
  ].filter(Boolean);
  for (const list of initLists) {
    const found = Array.isArray(list) ? list.find((t) => t?.name === name) : undefined;
    if (found) {
      vec = decodeIntegerVectorFromTensorProto(found);
      if (vec && vec.length) return vec;
    }
  }
  return undefined;
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
  const sliceShape: (number | String | undefined)[] =
  Array.isArray(cur.shape) ? [...cur.shape] : new Array(rank).fill(undefined);
  if (rank > 0) {
    sliceShape[axis] = 1;
  }

  const oneSlice = g.addNode(uniq(g, `edge_oneSlice_${tag}`))
    .init(new TensorNode.Builder(cur.literalType as DataType, sliceShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, sliceOp, oneSlice, cur.literalType as DataType, sliceShape);
  addEdge(g, sliceOp, oneSlice, cur.literalType as DataType, oneSlice.shape);

  // Expand to desired size along axis
  const size1D = makeI64ShapeConst(g, `edge_size_${tag}`, [size]);
  const newShape = editShapeDim(g, shp, axis, size1D, `edge_shape_edit_${tag}`);
  const exp = g.addNode(uniq(g, `edge_expand_${tag}`)).init(new OperationNode.Builder("Expand", [oneSlice, newShape], {})).as(OperationNode);
  const slabShape: (number | String | undefined)[] =
    Array.isArray(cur.shape) ? [...cur.shape] : new Array(rank).fill(undefined);
  if (rank > 0) {
    slabShape[axis] = size;
  }

  const slab = g.addNode(uniq(g, `edge_slab_${tag}`))
    .init(new TensorNode.Builder(cur.literalType as DataType, slabShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, exp, slab, cur.literalType as DataType, slabShape);
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

  // If we know statically that the dimension is 0 or 1, reflect is invalid.
  const staticDim = cur.shape[axis];
  if (staticDim === 0 || staticDim === 1) {
    return undefined;
  }

  const shp = shapeOf(g, cur, `refl_shape_${tag}`);
  const axisIdx = makeI64ShapeConst(g, `refl_axis_${tag}`, [axis]);

  // dim1D: [dim]
  const gdim = g.addNode(uniq(g, `refl_gather_${tag}`))
    .init(new OperationNode.Builder("Gather", [shp, axisIdx], { axis: 0 }))
    .as(OperationNode);
  const dim1D = g.addNode(uniq(g, `refl_dim1D_${tag}`))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
    .as(TensorNode);
  addEdge(g, gdim, dim1D, DataType.INT64, [1]);

  // dimSc: scalar dim ([])
  const dimAxes = makeI64ShapeConst(g, `refl_dim_axes_${tag}`, [0]);
  const dimSq = g.addNode(uniq(g, `refl_dim_sq_${tag}`))
    .init(new OperationNode.Builder("Squeeze", [dim1D, dimAxes], {}))
    .as(OperationNode);
  const dimSc = g.addNode(uniq(g, `refl_dim_scalar_${tag}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  addEdge(g, dimSq, dimSc, DataType.INT64, []);

  // Compute dim-1 and clamp requested size to [0, dim-1]
  const oneSc = scalarI64(g, `refl_oneSc_${tag}`, 1);
  const zeroSc = scalarI64(g, `refl_zeroSc_${tag}`, 0);

  const dimMinus1Op = g.addNode(uniq(g, `refl_dim_minus1_${tag}`))
    .init(new OperationNode.Builder("Sub", [dimSc, oneSc], {}))
    .as(OperationNode);
  const dimMinus1 = g.addNode(uniq(g, `refl_dim_minus1T_${tag}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  addEdge(g, dimMinus1Op, dimMinus1, DataType.INT64, []);

  const sizeSc = scalarI64(g, `refl_size_${tag}`, size);
  const sizeClampOp = g.addNode(uniq(g, `refl_size_clamp_${tag}`))
    .init(new OperationNode.Builder("Min", [sizeSc, dimMinus1], {}))
    .as(OperationNode);
  const sizeClamped = g.addNode(uniq(g, `refl_size_clamped_${tag}`))
    .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
    .as(TensorNode);
  addEdge(g, sizeClampOp, sizeClamped, DataType.INT64, []);

  const stepNeg1 = scalarI64(g, `refl_step_${tag}`, -1);

  let startSc: TensorNode.Class;
  let endSc: TensorNode.Class;

  if (tag.endsWith("L")) {
    // LEFT: indices = [sizeClamped, sizeClamped-1, ..., 1]
    // Range(start=sizeClamped, end=0, step=-1)
    startSc = sizeClamped;
    endSc = zeroSc;
  } else {
    // RIGHT: start = dim-2, end = start - sizeClamped
    const twoSc = scalarI64(g, `refl_two_${tag}`, 2);
    const startOp = g.addNode(uniq(g, `refl_r_start_${tag}`))
      .init(new OperationNode.Builder("Sub", [dimSc, twoSc], {})) // dim-2
      .as(OperationNode);
    const startT = g.addNode(uniq(g, `refl_r_startT_${tag}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, startOp, startT, DataType.INT64, []);

    const endOp = g.addNode(uniq(g, `refl_r_end_${tag}`))
      .init(new OperationNode.Builder("Sub", [startT, sizeClamped], {}))
      .as(OperationNode);
    const endT = g.addNode(uniq(g, `refl_r_endT_${tag}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, endOp, endT, DataType.INT64, []);

    startSc = startT; // []
    endSc = endT;     // []
  }

  // Range(startSc, endSc, -1) -> 1D index vector.
  // Runtime length == clamped size; static shape uses original 'size' for now.
  const rangeOp = g.addNode(uniq(g, `refl_range_${tag}`))
    .init(new OperationNode.Builder("Range", [startSc, endSc, stepNeg1], {}))
    .as(OperationNode);

  const idxShape: (number | string | undefined)[] = [size];
  const idx = g.addNode(uniq(g, `refl_idx_${tag}`))
    .init(new TensorNode.Builder(DataType.INT64, idxShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, rangeOp, idx, DataType.INT64, idxShape);

  // Gather along 'axis' using these indices.
  const gatherOp = g.addNode(uniq(g, `refl_gather_data_${tag}`))
    .init(new OperationNode.Builder("Gather", [cur, idx], { axis }))
    .as(OperationNode);

  // slab has same rank as cur, but axis dim = original requested size
  const reflSlabShape: (number | String | undefined)[] =
    Array.isArray(cur.shape) ? [...cur.shape] : new Array(rank).fill(undefined);
  if (rank > 0) {
    reflSlabShape[axis] = size;
  }

  const slab = g.addNode(uniq(g, `refl_slab_${tag}`))
    .init(new TensorNode.Builder(cur.literalType as DataType, reflSlabShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, gatherOp, slab, cur.literalType as DataType, reflSlabShape);
  return slab;
}

/* ------------------------------ Handler ------------------------------- */
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
    const a = op.getAttributes?.() ?? op.attributes ?? {};
    if (Array.isArray(a.pads)) pads = a.pads.map((x) => Number(x));
  }
  if (!pads || pads.length !== 2 * rank) return false;

  // mode
  const attr = op.getAttributes?.() ?? op.attributes ?? {};
  const modeRaw = String(attr.mode ?? "constant").toLowerCase();
  const mode: "constant" | "edge" | "reflect" = (modeRaw === "edge" || modeRaw === "reflect") ? (modeRaw) : "constant";

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
