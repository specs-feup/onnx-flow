import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { scalarInt64 } from "../../Utilities.js";

// ---------- small local utils ----------
function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}

function i64scalar(g: OnnxGraph.Class, name: string, v: number): TensorNode.Class {
  // true 0-D INT64 constant
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(DataType.INT64, [], "constant", scalarInt64(v)))
    .as(TensorNode);
}

/** Read an INT64 (or INT32/UINT64) vector from a TensorNode's constantValue/initializer/value. */
function readConstIntegerVectorFromTensorNode(tn?: TensorNode.Class): number[] | undefined {
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

function toArrayLike<T = any>(nc: any): T[] {
  return nc?.toArray?.() ?? nc ?? [];
}

function maybeRemoveOrphanConstant(g: OnnxGraph.Class, tn?: TensorNode.Class) {
  if (!tn) return;

  // Only consider constants/initializers (don’t touch real inputs/intermediates)
  const isConstLike = (tn as any).type === "constant" || (tn as any).constantValue != null;
  if (!isConstLike) return;

  // If the tensor has any consumers, keep it
  const outOpsNC = tn.getOutgoers?.targets?.filterIs?.(OperationNode);
  const outOps = toArrayLike<OperationNode.Class>(outOpsNC);
  if (outOps.length > 0) return;

  // Optionally remove an upstream Constant op that ONLY fed this tensor
  const inOpsNC = tn.getIncomers?.sources?.filterIs?.(OperationNode);
  const inOps = toArrayLike<OperationNode.Class>(inOpsNC);
  for (const srcOp of inOps) {
    if (srcOp.type !== "Constant") continue;

    // If the Constant's outputs have no other consumers, remove it too
    const constOutsNC = srcOp.getOutgoers?.targets?.filterIs?.(TensorNode);
    const constOuts = toArrayLike<TensorNode.Class>(constOutsNC);
    let anyOtherConsumers = false;
    for (const outT of constOuts) {
      const consumersNC = outT.getOutgoers?.targets?.filterIs?.(OperationNode);
      const consumers = toArrayLike<OperationNode.Class>(consumersNC);
      // Ignore the tensor we're about to delete
      if (consumers.length > 0) { anyOtherConsumers = true; break; }
    }
    if (!anyOtherConsumers) {
      srcOp.remove();
    }
  }

  // Finally remove the tensor itself
  tn.remove();
}


// ---------- main handler ----------
export default function sliceHandler(g: OnnxGraph.Class, sl: OperationNode.Class): boolean {
  if (sl.type !== "Slice") return false;

  const ins = sl.getInputs?.() ?? [];
  const Xn = ins[0];
  if (!Xn?.is?.(TensorNode)) return false;

  const Xin = Xn.as(TensorNode);
  const inShape = Xin.shape.map(d => (typeof d === "number" ? d : 1));
  const rank = inShape.length;

  // 1) Read params (attributes first, then from Constant producers on input slots 1..4)
  const inputs = sl.getInputs?.() ?? [];
  const readVec = (idx: number) =>
    inputs[idx]?.is?.(TensorNode)
      ? readConstIntegerVectorFromTensorNode(inputs[idx].as(TensorNode))
      : undefined;

  let starts = readVec(1);
  let ends   = readVec(2);
  let axes   = readVec(3);
  let steps  = readVec(4);

  if (!starts || !ends) {
    // dynamic Slice → leave as-is (handler returns false so TransformChain will process normally)
    console.log("Unable to read attributes of the Slice", sl.id);
    return false;
  }

  if (!axes)  axes  = Array.from({ length: starts.length }, (_, i) => i);
  if (!steps) steps = new Array(axes.length).fill(1);

  // 2) Expand per-axis parameters; normalize; require positive steps
  const fullStarts = new Array(rank).fill(0);
  const fullEnds   = inShape.slice();
  const fullSteps  = new Array(rank).fill(1);

  for (let i = 0; i < axes.length; i++) {
    const ax  = axes[i];
    const dim = inShape[ax] > 0 ? inShape[ax] : 1;

    let s = Number(starts[i]);
    let e = Number(ends[i]);
    const stp = Number(steps[i]);

    if (!(stp > 0)) {
      // v1: only positive steps; give up and keep Slice
      return false;
    }
    if (s < 0) s = dim + s;
    if (e < 0) e = dim + e;
    s = Math.max(0, Math.min(s, dim));
    e = Math.max(0, Math.min(e, dim));

    fullStarts[ax] = s;
    fullEnds[ax]   = e;
    fullSteps[ax]  = stp;
  }

  // 3) Get the original Slice output tensor Y (we’ll write into it at the end via Identity)
  const outs = sl.getOutgoers.targets ?? [];
  if (outs.length !== 1 || !outs[0].is?.(TensorNode)) return false;
  const Y = outs[0].as(TensorNode);

  // Which axes actually change something?
  const changingAxes = [...new Set(axes)]
    .sort((a, b) => a - b)
    .filter(ax => {
      const s = fullStarts[ax], e = fullEnds[ax], stp = fullSteps[ax];
      return !(s === 0 && e === inShape[ax] && stp === 1);
    });

  let curT: TensorNode.Class = Xin;
  if (changingAxes.length === 0) {
    // True no-op → single Identity(X → Y)
    const id = g.addNode(uniq(g, `sl_id_${sl.id}`))
      .init(new OperationNode.Builder("Identity", [Xin], {}))
      .as(OperationNode);
    g.addEdge(id, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    g.getNodeById(sl.id).remove();
    return true;
  }

  // Build Range→Gather chain; last Gather writes straight into Y
  for (let i = 0; i < changingAxes.length; i++) {
    const ax = changingAxes[i];
    const s = fullStarts[ax], e = fullEnds[ax], stp = fullSteps[ax];
    const len = Math.max(0, Math.ceil((e - s) / stp));

    const cS = i64scalar(g, `sl_s_${sl.id}_${ax}`, s);
    const cE = i64scalar(g, `sl_e_${sl.id}_${ax}`, e);
    const cT = i64scalar(g, `sl_t_${sl.id}_${ax}`, stp);

    const range = g.addNode(uniq(g, `sl_range_${sl.id}_${ax}`))
      .init(new OperationNode.Builder("Range", [cS, cE, cT]))
      .as(OperationNode);
    const idx = g.addNode(uniq(g, `sl_idx_${sl.id}_${ax}`))
      .init(new TensorNode.Builder(DataType.INT64, [len], "intermediate"))
      .as(TensorNode);
    g.addEdge(range, idx).init(new OnnxEdge.Builder(idx.literalType, idx.shape)).as(OnnxEdge);

    const gather = g.addNode(uniq(g, `sl_gather_${sl.id}_${ax}`))
      .init(new OperationNode.Builder("Gather", [curT, idx], { axis: ax }))
      .as(OperationNode);

    const isLast = (i === changingAxes.length - 1);
    if (isLast) {
      // ✅ final producer → Y (no Identity, no shape mutation)
      g.addEdge(gather, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    } else {
      // intermediate hop
      const mid = g.addNode(uniq(g, `sl_mid_${sl.id}_${ax}`))
        .init(new TensorNode.Builder(curT.literalType, [], "intermediate"))
        .as(TensorNode);
      g.addEdge(gather, mid).init(new OnnxEdge.Builder(mid.literalType, mid.shape)).as(OnnxEdge);
      curT = mid;
    }
  }

  // Remove the original Slice node
  g.getNodeById(sl.id).remove();
  const paramTensor = (i: number) => {
    const v = ins[i];
    return v?.is?.(TensorNode) ? v.as(TensorNode) : undefined;
  };
  maybeRemoveOrphanConstant(g, paramTensor(1)); // starts
  maybeRemoveOrphanConstant(g, paramTensor(2)); // ends
  maybeRemoveOrphanConstant(g, paramTensor(3)); // axes
  maybeRemoveOrphanConstant(g, paramTensor(4)); // steps

  return true;
}
