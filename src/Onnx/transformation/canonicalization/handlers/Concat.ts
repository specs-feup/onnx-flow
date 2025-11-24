import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { DataType } from "../../../OnnxTypes.js";
import { uniq, addEdge, toArrayLike, constI64, isNumeric, scalarI64, scalarZeroOfType } from "../../../Utils.js";

/* --------------------- opset13-friendly Squeeze/Unsqueeze -------------------- */
// In opset >= 13, Squeeze/Unsqueeze take axes as **2nd input**, not attribute.
function makeSqueeze(
  g: OnnxGraph.Class,
  x: TensorNode.Class,
  axes: number[],
  name: string
): { out: TensorNode.Class; op: OperationNode.Class } {
  const axesConst = constI64(g, `${name}_axes`, axes);
  const op = g
    .addNode(uniq(g, name))
    .init(new OperationNode.Builder("Squeeze", [x, axesConst], {}))
    .as(OperationNode);
  const out = g
    .addNode(uniq(g, `${name}_out`))
    .init(new TensorNode.Builder(DataType.INT64, axes.length === 1 ? [] : undefined, "intermediate"))
    .as(TensorNode);
  addEdge(g, op, out, DataType.INT64);
  return { out, op };
}

function makeUnsqueeze(
  g: OnnxGraph.Class,
  x: TensorNode.Class,
  axes: number[],
  outDtype: DataType,
  outShape: Array<number | string | undefined> | undefined,
  name: string
): { out: TensorNode.Class; op: OperationNode.Class } {
  const axesConst = constI64(g, `${name}_axes`, axes);
  const op = g
    .addNode(uniq(g, name))
    .init(new OperationNode.Builder("Unsqueeze", [x, axesConst], {}))
    .as(OperationNode);
  const out = g
    .addNode(uniq(g, `${name}_out`))
    .init(new TensorNode.Builder(outDtype, outShape, "intermediate"))
    .as(TensorNode);
  addEdge(g, op, out, outDtype, outShape);
  return { out, op };
}

/* ------------------------------ Handler ------------------------------- */
/**
 * Concat → ScatterElements block copies (numeric dtypes)
 *
 * Strategy:
 *   1) Build OutShape dynamically: same as inputs with axis dim = sum_i Shape(Xi)[axis]
 *   2) Initialize Y := Expand(zero(dtype), OutShape)
 *   3) For each input Xi, build per-element indices along `axis` using Range(offset, end, 1),
 *      rank-align via Unsqueeze (axes = all dims except `axis`) + Expand to Shape(Xi),
 *      then ScatterElements into Y on `axis`.
 *
 * Ops used: Shape, Gather, Add, Squeeze/Unsqueeze (as inputs), Concat (small vectors only),
 *           Range, Expand, ScatterElements.
 */
export default function concatHandler(
  g: OnnxGraph.Class,
  op: OperationNode.Class
): boolean {
  if (op.type !== "Concat") return false;

  const rawIns = op.getInputs?.() ?? [];
  if (rawIns.length < 2) return false;

  const inputs = rawIns
    .map((n) => (n?.is?.(TensorNode) ? n.as(TensorNode) : undefined))
    .filter(Boolean) as TensorNode.Class[];
  if (inputs.length < 2) return false;

  const outs = toArrayLike<TensorNode.Class>(
    op.getOutgoers?.targets?.filterIs?.(TensorNode)
  );
  if (outs.length !== 1) return false;
  const Y = outs[0];

  const a = op.getAttributes?.() ?? op.attributes ?? {};
  const axisAttr = Number(a.axis ?? 0);

  const rank = inputs[0].shape?.length;
  if (rank === undefined) return false;
  if (!inputs.every((t) => (t.shape?.length ?? -1) === rank)) return false;

  const dtype = inputs[0].literalType as DataType;
  if (!inputs.every((t) => t.literalType === dtype)) return false;
  if (!isNumeric(dtype)) return false; // numeric-only rewrite

  const axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
  if (axis < 0 || axis >= rank) return false;

  /* -------------------- Build OutShape (INT64 [rank]) -------------------- */
  const shape0Op = g
    .addNode(uniq(g, `Concat_Shape0_${op.id}`))
    .init(new OperationNode.Builder("Shape", [inputs[0]], {}))
    .as(OperationNode);
  const shape0 = g
    .addNode(uniq(g, `Concat_shape0_${op.id}`))
    .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
    .as(TensorNode);
  addEdge(g, shape0Op, shape0, DataType.INT64, [rank]);

  const dimPieces: TensorNode.Class[] = [];
  for (let d = 0; d < rank; d++) {
    if (d === axis) continue;
    const idx = constI64(g, `Concat_dim_${d}_idx_${op.id}`, [d]);
    const gop = g
      .addNode(uniq(g, `Concat_Gather_dim${d}_${op.id}`))
      .init(new OperationNode.Builder("Gather", [shape0, idx], { axis: 0 }))
      .as(OperationNode);
    const dimd = g
      .addNode(uniq(g, `Concat_dim${d}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
      .as(TensorNode);
    addEdge(g, gop, dimd, DataType.INT64, [1]);
    dimPieces.push(dimd);
  }

  // axis sizes (scalar per input)
  const sizeScalars: TensorNode.Class[] = [];
  for (let i = 0; i < inputs.length; i++) {
    const shapeIop = g
      .addNode(uniq(g, `Concat_Shape_in${i}_${op.id}`))
      .init(new OperationNode.Builder("Shape", [inputs[i]], {}))
      .as(OperationNode);
    const shapeI = g
      .addNode(uniq(g, `Concat_shape_in${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
      .as(TensorNode);
    addEdge(g, shapeIop, shapeI, DataType.INT64, [rank]);

    const axisIdx = constI64(g, `Concat_axis_idx_${i}_${op.id}`, [axis]);
    const gop = g
      .addNode(uniq(g, `Concat_Gather_axis_${i}_${op.id}`))
      .init(new OperationNode.Builder("Gather", [shapeI, axisIdx], { axis: 0 }))
      .as(OperationNode);
    const size1D = g
      .addNode(uniq(g, `Concat_size1D_${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
      .as(TensorNode);
    addEdge(g, gop, size1D, DataType.INT64, [1]);

    const { out: sizeSc } = makeSqueeze(g, size1D, [0], `Concat_Squeeze_sz_${i}_${op.id}`);
    sizeScalars.push(sizeSc);
  }

  // Sum axis sizes via Add chain (INT64 scalar)
  let sumAxis: TensorNode.Class = scalarI64(g, `Concat_sum_init_${op.id}`, 0);
  for (let i = 0; i < sizeScalars.length; i++) {
    const add = g
      .addNode(uniq(g, `Concat_sum_add_${i}_${op.id}`))
      .init(new OperationNode.Builder("Add", [sumAxis, sizeScalars[i]], {}))
      .as(OperationNode);
    const out = g
      .addNode(uniq(g, `Concat_sum_out_${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, add, out, DataType.INT64, []);
    sumAxis = out;
  }

  // Build OutShape by editing `shape0` at position `axis` using ScatterElements (no Concat)
  // indices = [axis]
  const axisIdxVec = constI64(g, `Concat_shape_axis_${op.id}`, [axis]);
  // updates = [sumAxis]
  const { out: sumAxis1D_forUpdate } = makeUnsqueeze(
    g,
    sumAxis,
    [0],
    DataType.INT64,
    [1],
    `Concat_unsq_sum_update_${op.id}`
  );
  const shapeEditOp = g
    .addNode(uniq(g, `Concat_edit_shape_${op.id}`))
    .init(new OperationNode.Builder("ScatterElements", [shape0, axisIdxVec, sumAxis1D_forUpdate], { axis: 0 }))
    .as(OperationNode);
  const outShape1D = g
    .addNode(uniq(g, `Concat_OutShape_${op.id}`))
    .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
    .as(TensorNode);
  addEdge(g, shapeEditOp, outShape1D, DataType.INT64, [rank]);

  /* --------------------- Initialize Y := Expand(0, shape) --------------------- */
  const zero = scalarZeroOfType(g, `Concat_zero_${op.id}`, dtype);
  const expandOp = g
    .addNode(uniq(g, `Concat_InitY_${op.id}`))
    .init(new OperationNode.Builder("Expand", [zero, outShape1D], {}))
    .as(OperationNode);
  let curY = g
    .addNode(uniq(g, `Concat_Y0_${op.id}`))
    .init(new TensorNode.Builder(dtype, Y.shape, "intermediate"))
    .as(TensorNode);
  addEdge(g, expandOp, curY, dtype, Y.shape);

  /* -------------- For each Xi: build indices and ScatterElements -------------- */
  let offsetSc: TensorNode.Class = scalarI64(g, `Concat_off_init_${op.id}`, 0); // INT64 scalar
  const oneSc = scalarI64(g, `Concat_one_${op.id}`, 1);

  for (let i = 0; i < inputs.length; i++) {
    const Xi = inputs[i];
    const sizeSc = sizeScalars[i];

    // end = offset + size
    const endOp = g
      .addNode(uniq(g, `Concat_off_end_op_${i}_${op.id}`))
      .init(new OperationNode.Builder("Add", [offsetSc, sizeSc], {}))
      .as(OperationNode);
    const endSc = g
      .addNode(uniq(g, `Concat_off_end_${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, endOp, endSc, DataType.INT64, []);

    // Range(offset, end, 1) → [Si] 
    const rangeOp = g.addNode(uniq(g, `Concat_range_${i}_${op.id}`))
      .init(new OperationNode.Builder("Range", [offsetSc, endSc, oneSc]))
      .as(OperationNode);

    // shape [size] along the concat axis
    const axisDim = Array.isArray(Xi.shape) ? Xi.shape[axis] : undefined;
    const rangeShape: (number | string | undefined)[] =
      [typeof axisDim === "number" ? axisDim : undefined];

    const range1D = g.addNode(uniq(g, `Concat_range1D_${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, rangeShape, "intermediate"))
      .as(TensorNode);
    addEdge(g, rangeOp, range1D, DataType.INT64, rangeShape);


    // Unsqueeze to rank r at all axes except `axis` so that the index dimension lands at `axis`
    const axesToUnsq: number[] = [];
    for (let d = 0; d < rank; d++) if (d !== axis) axesToUnsq.push(d);
    let idxRanked: TensorNode.Class;
    if (axesToUnsq.length === 0) {
      idxRanked = range1D;
    } else {
      const idxShape: (number | string | undefined)[] =
        Array.isArray(Xi.shape) ? [...Xi.shape] : new Array(rank).fill(undefined);

      // axesToUnsq are the non-concat dims → those should be 1 in the index tensor
      for (const d of axesToUnsq) {
        idxShape[d] = 1;
      }

      ({ out: idxRanked } = makeUnsqueeze(
        g,
        range1D,
        axesToUnsq,
        DataType.INT64,
        idxShape,
        `Concat_unsq_idx_${i}_${op.id}`
      ));
    }

    // Expand indices to match Shape(Xi)
    const shapeIop = g
      .addNode(uniq(g, `Concat_shape_for_idx_${i}_${op.id}`))
      .init(new OperationNode.Builder("Shape", [Xi], {}))
      .as(OperationNode);
    const shapeI = g
      .addNode(uniq(g, `Concat_shape_idx_${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
      .as(TensorNode);
    addEdge(g, shapeIop, shapeI, DataType.INT64, [rank]);

    const expIdxOp = g
      .addNode(uniq(g, `Concat_expand_idx_${i}_${op.id}`))
      .init(new OperationNode.Builder("Expand", [idxRanked, shapeI], {}))
      .as(OperationNode);
    const idxFull = g
      .addNode(uniq(g, `Concat_idx_full_${i}_${op.id}`))
      .init(new TensorNode.Builder(DataType.INT64, Xi.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, expIdxOp, idxFull, DataType.INT64, Xi.shape);

    // Y <- ScatterElements(Y, indices, Xi, axis)
    const scatterOp = g
      .addNode(uniq(g, `Concat_scatter_${i}_${op.id}`))
      .init(new OperationNode.Builder("ScatterElements", [curY, idxFull, Xi], { axis }))
      .as(OperationNode);
    const nextY = g
      .addNode(uniq(g, `Concat_Y${i + 1}_${op.id}`))
      .init(new TensorNode.Builder(dtype, Y.shape, "intermediate"))
      .as(TensorNode);
    addEdge(g, scatterOp, nextY, dtype, Y.shape);
    curY = nextY;

    // offset <- end
    offsetSc = endSc;
  }

  // Connect final Y to the original output via Identity to avoid multi-output Scatter
  const finalId = g
    .addNode(uniq(g, `Concat_Final_${op.id}`))
    .init(new OperationNode.Builder("Identity", [curY], {}))
    .as(OperationNode);
  addEdge(g, finalId, Y, dtype, Y.shape);

  // Remove original Concat op
  g.getNodeById(op.id).remove();

  return true;
}
