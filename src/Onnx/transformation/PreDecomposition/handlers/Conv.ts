import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { makeTensorProto } from "../../Utilities.js";

/* --------------------------------- utils --------------------------------- */
function uniq(g: OnnxGraph.Class, base: string): string {
  let i = 0, id = base;
  while (g.hasNode(id)) id = `${base}_${++i}`;
  return id;
}
function toArrayLike<T = any>(nc: any): T[] {
  return nc?.toArray?.() ?? nc ?? [];
}
function addEdge(
  g: OnnxGraph.Class,
  srcOp: OperationNode.Class,
  dstTensor: TensorNode.Class,
  dtype: DataType,
  shape?: Array<number | string | undefined>
) {
  g.addEdge(srcOp, dstTensor)
    .init(new OnnxEdge.Builder(dtype, shape ?? dstTensor.shape))
    .as(OnnxEdge);
}
function constI64(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(
      DataType.INT64,
      [vals.length],
      "constant",
      makeTensorProto(DataType.INT64, [vals.length], vals))
    ).as(TensorNode);
}
function constF32(g: OnnxGraph.Class, name: string, vals: number[]): TensorNode.Class {
  return g.addNode(uniq(g, name))
    .init(new TensorNode.Builder(
      DataType.FLOAT,
      [vals.length],
      "constant",
      makeTensorProto(DataType.FLOAT, [vals.length], vals))
    ).as(TensorNode);
}
function toNum(x: number | String | undefined): number | undefined {
  if (typeof x === "number") return x;
  if (typeof x === "string" && /^[0-9]+$/.test(x)) return Number(x);
  return undefined;
}
function toNumShape(s?: Array<number | String | undefined>): Array<number | undefined> | undefined {
  if (!s) return undefined;
  return s.map(toNum);
}
function normalizePads(padsAttr?: number[]): [number, number, number, number] {
  if (!padsAttr) return [0, 0, 0, 0];
  if (padsAttr.length === 4) return [padsAttr[0], padsAttr[1], padsAttr[2], padsAttr[3]];
  if (padsAttr.length === 8) return [padsAttr[2], padsAttr[3], padsAttr[6], padsAttr[7]];
  return [0, 0, 0, 0];
}

// Look up tensors by original ONNX name (initializer/constant/id)
function findTensorByOnnxName(g: OnnxGraph.Class, name?: string): TensorNode.Class | undefined {
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
function findConstantProducerAsTensor(g: OnnxGraph.Class, onnxName?: string): TensorNode.Class | undefined {
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

/* ============================== HANDLER ================================== */

export default function handleConv(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  console.log("CONV Handler");
  if (op.type !== "Conv") return false;

  const inNames: string[] =
    (op as any).inputs ??
    (op as any).input ??
    (op as any).getInputNames?.() ?? [];

  const outNames: string[] =
    (op as any).outputs ??
    (op as any).output ??
    (op as any).getOutputNames?.() ?? [];

  console.log(`--- Conv[${op.id}] ---`);
  console.log(`inputs: ${JSON.stringify(inNames)}`);
  console.log(`outputs: ${JSON.stringify(outNames)}`);

  // Resolve X/W/B by name (and Constants)
  let X = findTensorByOnnxName(g, inNames?.[0]);
  let W = findTensorByOnnxName(g, inNames?.[1]);
  let B = findTensorByOnnxName(g, inNames?.[2]);
  if (!W) W = findConstantProducerAsTensor(g, inNames?.[1]);
  if (!B && inNames?.[2]) B = findConstantProducerAsTensor(g, inNames?.[2]);

  console.log("name-based resolution:",
    "X:", X?.id ?? null,
    "W:", W?.id ?? null,
    "B:", B?.id ?? null
  );

  // Fallback to topo inputs if needed
  if (!X || !W) {
    const ins = op.getInputs?.() ?? [];
    X = X ?? (ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined);
    W = W ?? (ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined);
    B = B ?? (ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined);
  }
  if (!X || !W) return false;

  // Y
  const outsNC = op.getOutgoers?.targets?.filterIs?.(TensorNode);
  const outs = toArrayLike<TensorNode.Class>(outsNC);
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Attributes
  const a = (op as any).getAttributes?.() ?? (op as any).attributes ?? {};
  let strides: number[] = Array.isArray(a.strides) ? a.strides.map(Number) : [1, 1];
  let dilations: number[] = Array.isArray(a.dilations) ? a.dilations.map(Number) : [1, 1];
  if (strides.length === 1) strides = [strides[0], strides[0]];
  if (dilations.length === 1) dilations = [dilations[0], dilations[0]];
  const auto_pad = (a.auto_pad ?? "NOTSET") as string;
  const padsAttr: number[] | undefined = Array.isArray(a.pads) ? a.pads.map(Number) : (auto_pad === "VALID" ? [0,0,0,0] : undefined);
  const group = Number(a.group ?? 1);

  // Shapes & basic numbers
  const dtype = (X.literalType ?? DataType.FLOAT) as DataType;
  const [pt, pl, pb, pr] = normalizePads(padsAttr);
  const sH = strides[0] ?? 1, sW = strides[1] ?? 1;
  const dH = dilations[0] ?? 1, dW = dilations[1] ?? 1;

  const xShape = X.shape ?? [undefined, undefined, undefined, undefined];
  const wShape = W.shape ?? [undefined, undefined, undefined, undefined];

  const N = toNum(xShape[0]) ?? 1;
  const C = toNum(xShape[1]) ?? (toNum(wShape[1])! * group);
  const kH = toNum(wShape[2]) ?? 1;
  const kW = toNum(wShape[3]) ?? 1;
  const M  = toNum(wShape[0]) ?? 1;

  if (C % group !== 0 || M % group !== 0) return false;

  const Cg = Math.floor(C / group);
  const Mg = Math.floor(M / group);

  console.log("topology incomers:", (X.getIncomers?.sources?.toArray?.()?.length ?? 0) > 0 ? "ok" : "null");
  console.log("final tensors:",
    "X:", X.id, "shape:", JSON.stringify(X.shape),
    "W:", W.id, "shape:", JSON.stringify(W.shape),
    "B:", B ? B.id : "null", "Y:", Y.id, "shape:", JSON.stringify(Y.shape)
  );
  console.log("attrs:", {
    strides, dilations, padsAttr: padsAttr ?? [0,0,0,0], group, auto_pad: a.auto_pad ?? "NOTSET"
  });

  /* ---------------- Pad(X) ---------------- */
  const padsVec = constI64(g, "pads_pad", [0, 0, pt, pl, 0, 0, pb, pr]);
  const padVal = constF32(g, "pad_val", [0.0]);
  const padOp = g.addNode(uniq(g, "Pad_X"))
    .init(new OperationNode.Builder("Pad", [X, padsVec, padVal], {})).as(OperationNode);
  const Xpad = g.addNode(uniq(g, "Xpad"))
    .init(new TensorNode.Builder(dtype, [xShape[0], xShape[1], undefined, undefined], "intermediate")).as(TensorNode);
  addEdge(g, padOp, Xpad, dtype, toNumShape(Xpad.shape));
  console.log(`Pad created: Pad_X -> Xpad`);

  /* --------------- Shape/Gather on Xpad & W --------------- */
  const shapeX = g.addNode(uniq(g, "Shape_Xpad")).init(new OperationNode.Builder("Shape", [Xpad], {})).as(OperationNode);
  const tShapeX = g.addNode(uniq(g, "Shape_Xpad_T")).init(new TensorNode.Builder(DataType.INT64, [4], "intermediate")).as(TensorNode);
  addEdge(g, shapeX, tShapeX, DataType.INT64, [4]);

  const ax0 = constI64(g, "ax0", [0]);
  //const ax1 = constI64(g, "ax1", [1]);
  const ax2 = constI64(g, "ax2", [2]);
  const ax3 = constI64(g, "ax3", [3]);

  const gHpOp = g.addNode(uniq(g, "Gather_Hp")).init(new OperationNode.Builder("Gather", [tShapeX, ax2], { axis: 0 })).as(OperationNode);
  const gWpOp = g.addNode(uniq(g, "Gather_Wp")).init(new OperationNode.Builder("Gather", [tShapeX, ax3], { axis: 0 })).as(OperationNode);
  const Hp = g.addNode(uniq(g, "Hp")).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  const Wp = g.addNode(uniq(g, "Wp")).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  addEdge(g, gHpOp, Hp, DataType.INT64, [1]);
  addEdge(g, gWpOp, Wp, DataType.INT64, [1]);

  const shapeW = g.addNode(uniq(g, "Shape_W")).init(new OperationNode.Builder("Shape", [W], {})).as(OperationNode);
  const tShapeW = g.addNode(uniq(g, "Shape_W_T")).init(new TensorNode.Builder(DataType.INT64, [4], "intermediate")).as(TensorNode);
  addEdge(g, shapeW, tShapeW, DataType.INT64, [4]);

  const gMOp  = g.addNode(uniq(g, "Gather_M")).init(new OperationNode.Builder("Gather", [tShapeW, ax0], { axis: 0 })).as(OperationNode);
  const gkHOp = g.addNode(uniq(g, "Gather_kH")).init(new OperationNode.Builder("Gather", [tShapeW, ax2], { axis: 0 })).as(OperationNode);
  const gkWOp = g.addNode(uniq(g, "Gather_kW")).init(new OperationNode.Builder("Gather", [tShapeW, ax3], { axis: 0 })).as(OperationNode);
  const Mdim  = g.addNode(uniq(g, "Mdim")).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  const kHdim = g.addNode(uniq(g, "kHdim")).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  const kWdim = g.addNode(uniq(g, "kWdim")).init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  addEdge(g, gMOp,  Mdim,  DataType.INT64, [1]);
  addEdge(g, gkHOp, kHdim, DataType.INT64, [1]);
  addEdge(g, gkWOp, kWdim, DataType.INT64, [1]);

  /* --------------- Effective kernel & output sizes --------------- */
    /* --------------- Effective kernel (effK = (k-1)*d + 1) --------------- */
  const oneI = constI64(g, "oneI", [1]);
  const dHc  = constI64(g, "dHc", [dH]);
  const dWc  = constI64(g, "dWc", [dW]);

  // k - 1  (i64 -> i64 tensors)
  const kH_1_op = g.addNode(uniq(g, "kH_minus_1"))
    .init(new OperationNode.Builder("Sub", [kHdim, oneI], {})).as(OperationNode);
  const kW_1_op = g.addNode(uniq(g, "kW_minus_1"))
    .init(new OperationNode.Builder("Sub", [kWdim, oneI], {})).as(OperationNode);
  const kH_1 = g.addNode(uniq(g, "kH_1"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  const kW_1 = g.addNode(uniq(g, "kW_1"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  addEdge(g, kH_1_op, kH_1, DataType.INT64, [1]);
  addEdge(g, kW_1_op, kW_1, DataType.INT64, [1]);

  // Cast to float (materialize tensors)
  const kH_1f_op = g.addNode(uniq(g, "kH_1f"))
    .init(new OperationNode.Builder("Cast", [kH_1], { to: DataType.FLOAT })).as(OperationNode);
  const kW_1f_op = g.addNode(uniq(g, "kW_1f"))
    .init(new OperationNode.Builder("Cast", [kW_1], { to: DataType.FLOAT })).as(OperationNode);
  const dHf_op   = g.addNode(uniq(g, "dHf"))
    .init(new OperationNode.Builder("Cast", [dHc], { to: DataType.FLOAT })).as(OperationNode);
  const dWf_op   = g.addNode(uniq(g, "dWf"))
    .init(new OperationNode.Builder("Cast", [dWc], { to: DataType.FLOAT })).as(OperationNode);
  const kH_1f_T  = g.addNode(uniq(g, "kH_1f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  const kW_1f_T  = g.addNode(uniq(g, "kW_1f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  const dHf_T    = g.addNode(uniq(g, "dHf_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  const dWf_T    = g.addNode(uniq(g, "dWf_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  addEdge(g, kH_1f_op, kH_1f_T, DataType.FLOAT, [1]);
  addEdge(g, kW_1f_op, kW_1f_T, DataType.FLOAT, [1]);
  addEdge(g, dHf_op,   dHf_T,   DataType.FLOAT, [1]);
  addEdge(g, dWf_op,   dWf_T,   DataType.FLOAT, [1]);

  // (k-1)*d  (materialize tensors)
  const kHmd_f_op = g.addNode(uniq(g, "kHmd_f"))
    .init(new OperationNode.Builder("Mul", [kH_1f_T, dHf_T], {})).as(OperationNode);
  const kWmd_f_op = g.addNode(uniq(g, "kWmd_f"))
    .init(new OperationNode.Builder("Mul", [kW_1f_T, dWf_T], {})).as(OperationNode);
  const kHmd_f_T  = g.addNode(uniq(g, "kHmd_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  const kWmd_f_T  = g.addNode(uniq(g, "kWmd_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  addEdge(g, kHmd_f_op, kHmd_f_T, DataType.FLOAT, [1]);
  addEdge(g, kWmd_f_op, kWmd_f_T, DataType.FLOAT, [1]);

  // +(1.0) -> effKH_f/effKW_f (materialize) -> Cast back to i64 tensors effKH/effKW
  const oneF = constF32(g, "oneF", [1.0]);

  const effKH_f_op = g.addNode(uniq(g, "effKH_f"))
    .init(new OperationNode.Builder("Add", [kHmd_f_T, oneF], {})).as(OperationNode);
  const effKW_f_op = g.addNode(uniq(g, "effKW_f"))
    .init(new OperationNode.Builder("Add", [kWmd_f_T, oneF], {})).as(OperationNode);
  const effKH_f_T  = g.addNode(uniq(g, "effKH_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  const effKW_f_T  = g.addNode(uniq(g, "effKW_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  addEdge(g, effKH_f_op, effKH_f_T, DataType.FLOAT, [1]);
  addEdge(g, effKW_f_op, effKW_f_T, DataType.FLOAT, [1]);

  const effKH_c = g.addNode(uniq(g, "effKH_c"))
    .init(new OperationNode.Builder("Cast", [effKH_f_T], { to: DataType.INT64 })).as(OperationNode);
  const effKW_c = g.addNode(uniq(g, "effKW_c"))
    .init(new OperationNode.Builder("Cast", [effKW_f_T], { to: DataType.INT64 })).as(OperationNode);
  const effKH   = g.addNode(uniq(g, "effKH"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  const effKW   = g.addNode(uniq(g, "effKW"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  addEdge(g, effKH_c, effKH, DataType.INT64, [1]);
  addEdge(g, effKW_c, effKW, DataType.INT64, [1]);


  // ---------- output sizes: floor((Hp - effKH)/sH) + 1  (float div + floor) ----------
  const sHc = constI64(g, "sHc", [sH]);
  const sWc = constI64(g, "sWc", [sW]);

  const Hp_minus_eff_op = g.addNode(uniq(g, "Hp_minus_eff"))
    .init(new OperationNode.Builder("Sub", [Hp, effKH], {})).as(OperationNode);
  const Hp_minus_eff_T = g.addNode(uniq(g, "Hp_minus_eff_T"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  g.addEdge(Hp_minus_eff_op, Hp_minus_eff_T).init(new OnnxEdge.Builder(DataType.INT64, [1])).as(OnnxEdge);

  const Wp_minus_eff_op = g.addNode(uniq(g, "Wp_minus_eff"))
    .init(new OperationNode.Builder("Sub", [Wp, effKW], {})).as(OperationNode);
  const Wp_minus_eff_T = g.addNode(uniq(g, "Wp_minus_eff_T"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  g.addEdge(Wp_minus_eff_op, Wp_minus_eff_T).init(new OnnxEdge.Builder(DataType.INT64, [1])).as(OnnxEdge);

  const Hm_f_op = g.addNode(uniq(g, "Hm_f"))
    .init(new OperationNode.Builder("Cast", [Hp_minus_eff_T], { to: DataType.FLOAT })).as(OperationNode);
  const Hm_f_T = g.addNode(uniq(g, "Hm_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(Hm_f_op, Hm_f_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);

  const Wm_f_op = g.addNode(uniq(g, "Wm_f"))
    .init(new OperationNode.Builder("Cast", [Wp_minus_eff_T], { to: DataType.FLOAT })).as(OperationNode);
  const Wm_f_T = g.addNode(uniq(g, "Wm_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(Wm_f_op, Wm_f_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);

  const sHf_op = g.addNode(uniq(g, "sHf"))
    .init(new OperationNode.Builder("Cast", [sHc], { to: DataType.FLOAT })).as(OperationNode);
  const sHf_T = g.addNode(uniq(g, "sHf_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(sHf_op, sHf_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);

  const sWf_op = g.addNode(uniq(g, "sWf"))
    .init(new OperationNode.Builder("Cast", [sWc], { to: DataType.FLOAT })).as(OperationNode);
  const sWf_T = g.addNode(uniq(g, "sWf_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(sWf_op, sWf_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);

  const Hd_f_op = g.addNode(uniq(g, "Hd_f"))
    .init(new OperationNode.Builder("Div", [Hm_f_T, sHf_T], {})).as(OperationNode);
  const Hd_f_T = g.addNode(uniq(g, "Hd_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(Hd_f_op, Hd_f_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);

  const Wd_f_op = g.addNode(uniq(g, "Wd_f"))
    .init(new OperationNode.Builder("Div", [Wm_f_T, sWf_T], {})).as(OperationNode);
  const Wd_f_T = g.addNode(uniq(g, "Wd_f_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(Wd_f_op, Wd_f_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);

  const Hfl_op = g.addNode(uniq(g, "Hfl"))
    .init(new OperationNode.Builder("Floor", [Hd_f_T], {})).as(OperationNode);
  const Hfl_T = g.addNode(uniq(g, "Hfl_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(Hfl_op, Hfl_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);
  const Hfl_plus_one_op = g.addNode(uniq(g, "Hfl_plus_one"))
    .init(new OperationNode.Builder("Add", [Hfl_T, oneF], {})).as(OperationNode);
  const Hfl_plus_one = g.addNode(uniq(g,"Hfl_plus_one"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  addEdge(g, Hfl_plus_one_op, Hfl_plus_one, DataType.FLOAT, [1]);

  const Wfl_op = g.addNode(uniq(g, "Wfl"))
    .init(new OperationNode.Builder("Floor", [Wd_f_T], {})).as(OperationNode);
  const Wfl_T = g.addNode(uniq(g, "Wfl_T"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  g.addEdge(Wfl_op, Wfl_T).init(new OnnxEdge.Builder(DataType.FLOAT, [1])).as(OnnxEdge);
  const Wfl_plus_one_op = g.addNode(uniq(g, "Wfl_plus_one"))
    .init(new OperationNode.Builder("Add", [Wfl_T, oneF], {})).as(OperationNode);
  const Wfl_plus_one = g.addNode(uniq(g,"Wfl_plus_one"))
    .init(new TensorNode.Builder(DataType.FLOAT, [1], "intermediate")).as(TensorNode);
  addEdge(g, Wfl_plus_one_op, Wfl_plus_one, DataType.FLOAT, [1]);

  const Hout_i64_op = g.addNode(uniq(g, "Hout"))
    .init(new OperationNode.Builder("Cast", [Hfl_plus_one], { to: DataType.INT64 })).as(OperationNode);
  const Wout_i64_op = g.addNode(uniq(g, "Wout"))
    .init(new OperationNode.Builder("Cast", [Wfl_plus_one], { to: DataType.INT64 })).as(OperationNode);
  const Hout = g.addNode(uniq(g, "Hout_T"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  const Wout = g.addNode(uniq(g, "Wout_T"))
    .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
  g.addEdge(Hout_i64_op, Hout).init(new OnnxEdge.Builder(DataType.INT64, [1])).as(OnnxEdge);
  g.addEdge(Wout_i64_op, Wout).init(new OnnxEdge.Builder(DataType.INT64, [1])).as(OnnxEdge);

  /* --------------- Shared constants for Slice --------------- */
  const axes_nchw = constI64(g, "axes_nchw", [0, 1, 2, 3]);
  const steps     = constI64(g, "steps",     [1, 1, sH, sW]);

  /* --------------- Group slice of X --------------- */
  console.log(`grouping: group=${group} C=${C} M=${M} Cg=${Cg} Mg=${Mg}`);
  const groupAcc: TensorNode.Class[] = [];

  for (let gi = 0; gi < group; gi++) {
    const Nc   = constI64(g, `Nc_${gi}`,   [N]);
    const Cgc  = constI64(g, `Cgc_${gi}`,  [Cg]);
    const Hp1  = Hp;
    const Wp1  = Wp;

    const cstart = constI64(g, `cstart_g${gi}`, [0, gi * Cg, 0, 0]);
    const cendC  = constI64(g, `cendC_g${gi}`,  [gi * Cg + Cg]);
    const ends_ch = g.addNode(uniq(g, `ends_ch_full_${gi}`))
      .init(new OperationNode.Builder("Concat", [Nc, cendC, Hp1, Wp1], { axis: 0 })).as(OperationNode);
    const ends_ch_T = g.addNode(uniq(g, `ends_ch_fullT_${gi}`))
      .init(new TensorNode.Builder(DataType.INT64, [4], "intermediate")).as(TensorNode);
    addEdge(g, ends_ch, ends_ch_T, DataType.INT64, [4]);

    const sliceXg = g.addNode(uniq(g, `Slice_Xg_${gi}`))
      .init(new OperationNode.Builder("Slice", [Xpad, cstart, ends_ch_T, axes_nchw], {})).as(OperationNode);
    const Xg = g.addNode(uniq(g, `Xg_${gi}`))
      .init(new TensorNode.Builder(dtype, [xShape[0], Cg, undefined, undefined], "intermediate")).as(TensorNode);
    addEdge(g, sliceXg, Xg, dtype, toNumShape(Xg.shape));

    let accNHWC: TensorNode.Class | undefined;

    for (let kh = 0; kh < (kH ?? 1); kh++) {
      for (let kw = 0; kw < (kW ?? 1); kw++) {
        // ✅ FIXED: use Hout/Wout tensors here (not the ops)
        const HmulOp = g.addNode(uniq(g, `Hmul_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Mul", [Hout, sHc], {})).as(OperationNode);
        const WmulOp = g.addNode(uniq(g, `Wmul_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Mul", [Wout, sWc], {})).as(OperationNode);
        const HmulT = g.addNode(uniq(g, `HmulT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
        const WmulT = g.addNode(uniq(g, `WmulT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
        addEdge(g, HmulOp, HmulT, DataType.INT64, [1]);
        addEdge(g, WmulOp, WmulT, DataType.INT64, [1]);

        const Hoff = constI64(g, `Hoff_${gi}_${kh}_${kw}`, [kh * dH]);
        const Woff = constI64(g, `Woff_${gi}_${kh}_${kw}`, [kw * dW]);

        const HendOp = g.addNode(uniq(g, `Hend_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Add", [HmulT, Hoff], {})).as(OperationNode);
        const WendOp = g.addNode(uniq(g, `Wend_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Add", [WmulT, Woff], {})).as(OperationNode);
        const HendT = g.addNode(uniq(g, `HendT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
        const WendT = g.addNode(uniq(g, `WendT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
        addEdge(g, HendOp, HendT, DataType.INT64, [1]);
        addEdge(g, WendOp, WendT, DataType.INT64, [1]);

        const HendCOp = g.addNode(uniq(g, `HendC_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Min", [HendT, Hp], {})).as(OperationNode);
        const WendCOp = g.addNode(uniq(g, `WendC_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Min", [WendT, Wp], {})).as(OperationNode);
        const HendC = g.addNode(uniq(g, `HendCT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
        const WendC = g.addNode(uniq(g, `WendCT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate")).as(TensorNode);
        addEdge(g, HendCOp, HendC, DataType.INT64, [1]);
        addEdge(g, WendCOp, WendC, DataType.INT64, [1]);

        const starts = constI64(g, `st_${gi}_${kh}_${kw}`, [0, 0, kh * dH, kw * dW]);
        const endsOp = g.addNode(uniq(g, `ends_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Concat", [Nc, Cgc, HendC, WendC], { axis: 0 })).as(OperationNode);
        const endsT = g.addNode(uniq(g, `endsT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [4], "intermediate")).as(TensorNode);
        addEdge(g, endsOp, endsT, DataType.INT64, [4]);

        const XsOp = g.addNode(uniq(g, `Slice_Xs_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Slice", [Xg, starts, endsT, axes_nchw, steps], {})).as(OperationNode);
        const Xs = g.addNode(uniq(g, `Xs_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(dtype, [xShape[0], Cg, undefined, undefined], "intermediate")).as(TensorNode);
        addEdge(g, XsOp, Xs, dtype, toNumShape(Xs.shape));

        // per-tap weight slice
        const wst_tap = constI64(g, `wst_tap_${gi}_${kh}_${kw}`, [gi * Mg, 0, kh, kw]);
        const wend_M  = constI64(g, `wend_M_${gi}`, [gi * Mg + Mg]);
        const wend_C  = constI64(g, `wend_C_${gi}`, [Cg]);
        const wenTail = constI64(g, `wen_tail_${gi}_${kh}_${kw}`, [kh + 1, kw + 1]);
        const wenFull = g.addNode(uniq(g, `wen_full_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Concat", [wend_M, wend_C, wenTail], { axis: 0 })).as(OperationNode);
        const wenFullT = g.addNode(uniq(g, `wen_fullT_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(DataType.INT64, [4], "intermediate")).as(TensorNode);
        addEdge(g, wenFull, wenFullT, DataType.INT64, [4]);

        const sliceWtap = g.addNode(uniq(g, `Slice_Wtap_${gi}_${kh}_${kw}`))
          .init(new OperationNode.Builder("Slice", [W, wst_tap, wenFullT], {})).as(OperationNode);
        const Wtap = g.addNode(uniq(g, `Wtap_${gi}_${kh}_${kw}`))
          .init(new TensorNode.Builder(dtype, [Mg, Cg, 1, 1], "intermediate")).as(TensorNode);
        addEdge(g, sliceWtap, Wtap, dtype, toNumShape(Wtap.shape));

        // Unsqueeze X and Wtap (axes as inputs, opset ≥ 13)
        const axes1 = constI64(g, `axes1_${gi}_${kh}_${kw}`, [1]);
        const axes0 = constI64(g, `axes0_${gi}_${kh}_${kw}`, [0]);
        const uqXop = g.addNode(uniq(g, `Unsq_X_${gi}_${kh}_${kw}`)).init(new OperationNode.Builder("Unsqueeze", [Xs, axes1], {})).as(OperationNode);
        const uqWop = g.addNode(uniq(g, `Unsq_W_${gi}_${kh}_${kw}`)).init(new OperationNode.Builder("Unsqueeze", [Wtap, axes0], {})).as(OperationNode);
        const Xb = g.addNode(uniq(g, `Xb_${gi}_${kh}_${kw}`)).init(new TensorNode.Builder(dtype, [xShape[0], 1, Cg, undefined, undefined], "intermediate")).as(TensorNode);
        const Wb = g.addNode(uniq(g, `Wb_${gi}_${kh}_${kw}`)).init(new TensorNode.Builder(dtype, [1, Mg, Cg, 1, 1], "intermediate")).as(TensorNode);
        addEdge(g, uqXop, Xb, dtype, toNumShape(Xb.shape));
        addEdge(g, uqWop, Wb, dtype, toNumShape(Wb.shape));

        const mul = g.addNode(uniq(g, `Mul_${gi}_${kh}_${kw}`)).init(new OperationNode.Builder("Mul", [Xb, Wb], {})).as(OperationNode);
        const prod = g.addNode(uniq(g, `prod_${gi}_${kh}_${kw}`)).init(new TensorNode.Builder(dtype, [xShape[0], Mg, Cg, undefined, undefined], "intermediate")).as(TensorNode);
        addEdge(g, mul, prod, dtype, toNumShape(prod.shape));

        const axesC = constI64(g, `axesC_${gi}_${kh}_${kw}`, [2]);
        const red = g.addNode(uniq(g, `ReduceSum_${gi}_${kh}_${kw}`)).init(new OperationNode.Builder("ReduceSum", [prod, axesC], { keepdims: 0 })).as(OperationNode);
        const nmhw = g.addNode(uniq(g, `nmhw_${gi}_${kh}_${kw}`)).init(new TensorNode.Builder(dtype, [xShape[0], Mg, undefined, undefined], "intermediate")).as(TensorNode);
        addEdge(g, red, nmhw, dtype, toNumShape(nmhw.shape));

        const toNHWC = g.addNode(uniq(g, `T_to_NHWC_${gi}_${kh}_${kw}`)).init(new OperationNode.Builder("Transpose", [nmhw], { perm: [0, 2, 3, 1] })).as(OperationNode);
        const ypart = g.addNode(uniq(g, `Ypart_${gi}_${kh}_${kw}`)).init(new TensorNode.Builder(dtype, [xShape[0], undefined, undefined, Mg], "intermediate")).as(TensorNode);
        addEdge(g, toNHWC, ypart, dtype, toNumShape(ypart.shape));

        if (!accNHWC) accNHWC = ypart;
        else {
          const add = g.addNode(uniq(g, `AccAdd_${gi}_${kh}_${kw}`)).init(new OperationNode.Builder("Add", [accNHWC, ypart], {})).as(OperationNode);
          const accT = g.addNode(uniq(g, `AccT_${gi}_${kh}_${kw}`)).init(new TensorNode.Builder(dtype, [xShape[0], undefined, undefined, Mg], "intermediate")).as(TensorNode);
          addEdge(g, add, accT, dtype, toNumShape(accT.shape));
          accNHWC = accT;
        }
      }
    }

    if (!accNHWC) return false;
    groupAcc.push(accNHWC);
  }

  // Concat groups (NHWC, dim=3)
  let nhwc: TensorNode.Class;
  if (groupAcc.length === 1) nhwc = groupAcc[0]!;
  else {
    const cat = g.addNode(uniq(g, "Concat_groups")).init(new OperationNode.Builder("Concat", groupAcc, { axis: 3 })).as(OperationNode);
    nhwc = g.addNode(uniq(g, "nhwc_all")).init(new TensorNode.Builder(dtype, [xShape[0], undefined, undefined, M], "intermediate")).as(TensorNode);
    addEdge(g, cat, nhwc, dtype, toNumShape(nhwc.shape));
  }

  // Bias: reshape and add with correct broadcasting
  let nhwc_b = nhwc;
  if (B) {
    // Detect scalar-like bias: rank-0 OR rank-1 with length==1
    const bRank = B.shape?.length ?? 0;  // rank-0 if shape is undefined/[]
    const bLen  = typeof B.shape?.[0] === "number" ? (B.shape![0] as number) : undefined;
    const isScalarLike = (bRank === 0) || (bRank === 1 && bLen === 1);

    const one = constI64(g, "oneA", [1]);
    const oneB = constI64(g, "oneB", [1]);
    const oneC = constI64(g, "oneC", [1]);

    // Target shape for reshape(B): [1,1,1,1] if scalar-like, else [1,1,1,M]
    const lastDim = isScalarLike
      ? constI64(g, "oneD", [1])
      : (g.getNodeById("Mdim")?.as?.(TensorNode) ?? constI64(g, "Mdim_fallback", [M]));

    const shapeB4op = g.addNode(uniq(g, "shapeB4"))
      .init(new OperationNode.Builder("Concat", [one, oneB, oneC, lastDim], { axis: 0 }))
      .as(OperationNode);
    const shapeB4 = g.addNode(uniq(g, "shapeB4T"))
      .init(new TensorNode.Builder(DataType.INT64, [4], "intermediate"))
      .as(TensorNode);
    addEdge(g, shapeB4op, shapeB4, DataType.INT64, [4]);

    const reshapeB = g.addNode(uniq(g, "ReshapeB"))
      .init(new OperationNode.Builder("Reshape", [B, shapeB4], {}))
      .as(OperationNode);
    const B4 = g.addNode(uniq(g, "B4"))
      .init(new TensorNode.Builder(dtype, [1, 1, 1, isScalarLike ? 1 : M], "intermediate"))
      .as(TensorNode);
    addEdge(g, reshapeB, B4, dtype, toNumShape(B4.shape));

    const addB = g.addNode(uniq(g, "AddBias"))
      .init(new OperationNode.Builder("Add", [nhwc, B4], {}))
      .as(OperationNode);
    const nhwcBias = g.addNode(uniq(g, "nhwc_bias"))
      .init(new TensorNode.Builder(dtype, [xShape[0], undefined, undefined, M], "intermediate"))
      .as(TensorNode);
    addEdge(g, addB, nhwcBias, dtype, toNumShape(nhwcBias.shape));
    nhwc_b = nhwcBias;
  }

  Y.setShape([xShape[0], M, undefined, undefined]);
  // NHWC -> NCHW
  const toNCHW = g.addNode(uniq(g, `T_to_NCHW_${op.id}`)).init(new OperationNode.Builder("Transpose", [nhwc_b], { perm: [0, 3, 1, 2] })).as(OperationNode);
  addEdge(g, toNCHW, Y, Y.literalType ?? dtype, toNumShape(Y.shape));

  // Remove Conv
  g.getNodeById(op.id)?.remove();
  console.log(`Rewired ${op.id} → ${Y.id}; removing Conv node`);
  return true;
}
