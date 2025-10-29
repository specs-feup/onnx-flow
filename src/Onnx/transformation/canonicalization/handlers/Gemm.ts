import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import { toArrayLike, uniq, addEdge, scalarOfType } from "../../../Utils.js";

/* ------------------------------ Handler ------------------------------- */
export default function gemmHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
  if (op.type !== "Gemm") return false;

  // Inputs in topo order
  const ins = op.getInputs?.() ?? [];
  if (ins.length < 2) return false;

  const A = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
  const B = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
  const C = ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined;
  if (!A || !B) return false;

  // Single output tensor Y
  const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
  if (outs.length !== 1) return false;
  const Y = outs[0];

  // Attributes (defaults: alpha=1.0, beta=1.0, transA=0, transB=0)
  const a = (op as any).getAttributes?.() ?? (op as any).attributes ?? {};
  const alpha = Number(a.alpha ?? 1.0);
  const beta  = Number(a.beta  ?? 1.0);
  const transA = Number(a.transA ?? 0) === 1 ? 1 : 0;
  const transB = Number(a.transB ?? 0) === 1 ? 1 : 0;

  // DType selections
  const dtypeLeft  = (A.literalType ?? DataType.FLOAT) as DataType;
  const dtypeRight = (C?.literalType ?? dtypeLeft) as DataType;

  /* ---------- optional Transpose on A/B ---------- */
  let A_in: TensorNode.Class = A;
  let B_in: TensorNode.Class = B;

  if (transA) {
    const tA = g.addNode(uniq(g, `Gemm_TA_${op.id}`))
      .init(new OperationNode.Builder("Transpose", [A], { perm: [1, 0] }))
      .as(OperationNode);
    const A_T = g.addNode(uniq(g, `Gemm_A_T_${op.id}`))
      .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, tA, A_T, dtypeLeft);
    A_in = A_T;
  }

  if (transB) {
    const tB = g.addNode(uniq(g, `Gemm_TB_${op.id}`))
      .init(new OperationNode.Builder("Transpose", [B], { perm: [1, 0] }))
      .as(OperationNode);
    const B_T = g.addNode(uniq(g, `Gemm_B_T_${op.id}`))
      .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, tB, B_T, dtypeLeft);
    B_in = B_T;
  }

  /* ------------------ MatMul(A', B') ------------------ */
  const mm = g.addNode(uniq(g, `Gemm_MM_${op.id}`))
    .init(new OperationNode.Builder("MatMul", [A_in, B_in], {}))
    .as(OperationNode);
  const MM = g.addNode(uniq(g, `Gemm_MM_T_${op.id}`))
    .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
    .as(TensorNode);
  addEdge(g, mm, MM, dtypeLeft);

  /* -------------------- scale by alpha -------------------- */
  let left: TensorNode.Class = MM;
  if (alpha !== 1.0) {
    const aC = scalarOfType(g, `Gemm_alpha_${op.id}`, alpha, dtypeLeft);
    const mulA = g.addNode(uniq(g, `Gemm_MulAlpha_${op.id}`))
      .init(new OperationNode.Builder("Mul", [MM, aC], {}))
      .as(OperationNode);
    const ScMM = g.addNode(uniq(g, `Gemm_ScaledMM_${op.id}`))
      .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
      .as(TensorNode);
    addEdge(g, mulA, ScMM, dtypeLeft);
    left = ScMM;
  }

  /* -------------------- optional + beta*C -------------------- */
  let producedToY = false;

  if (C && beta !== 0.0) {
    let cTerm: TensorNode.Class = C;
    if (beta !== 1.0) {
      const bC = scalarOfType(g, `Gemm_beta_${op.id}`, beta, dtypeRight);
      const mulB = g.addNode(uniq(g, `Gemm_MulBeta_${op.id}`))
        .init(new OperationNode.Builder("Mul", [C, bC], {}))
        .as(OperationNode);
      const ScC = g.addNode(uniq(g, `Gemm_ScaledC_${op.id}`))
        .init(new TensorNode.Builder(dtypeRight, [], "intermediate"))
        .as(TensorNode);
      addEdge(g, mulB, ScC, dtypeRight);
      cTerm = ScC;
    }

    // Y = left + cTerm
    const add = g.addNode(uniq(g, `Gemm_Add_${op.id}`))
      .init(new OperationNode.Builder("Add", [left, cTerm], {}))
      .as(OperationNode);
    g.addEdge(add, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    producedToY = true;
  }

  if (!producedToY) {
    // No C-branch → wire 'left' directly to Y
    const srcOp = toArrayLike<OperationNode.Class>(left.getIncomers?.sources?.filterIs?.(OperationNode))[0];
    if (srcOp) {
      g.addEdge(srcOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    } else {
      // fallback: Identity
      const id = g.addNode(uniq(g, `Gemm_Id_${op.id}`))
        .init(new OperationNode.Builder("Identity", [left], {}))
        .as(OperationNode);
      g.addEdge(id, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    }
  }

  // Remove original Gemm op
  g.getNodeById(op.id).remove();

  return true;
}
