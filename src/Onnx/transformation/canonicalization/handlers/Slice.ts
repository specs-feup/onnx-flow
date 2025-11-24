import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { readConstIntegerVectorFromTensorNode, uniq, maybeRemoveOrphanConstant, scalarI64 } from "../../../Utils.js";

// ---------- Handler ----------
export default function sliceHandler(g: OnnxGraph.Class, sl: OperationNode.Class): boolean {
  if (sl.type !== "Slice") return false;

  const ins = sl.getInputs?.() ?? [];
  if (ins.length < 2) return false;
  
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
    //console.log("Unable to read attributes of the Slice", sl.id);
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

    const cS = scalarI64(g, `sl_s_${sl.id}_${ax}`, s);
    const cE = scalarI64(g, `sl_e_${sl.id}_${ax}`, e);
    const cT = scalarI64(g, `sl_t_${sl.id}_${ax}`, stp);

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
      // final producer → Y (no Identity, no shape mutation)
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
