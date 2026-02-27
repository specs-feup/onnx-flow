import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import type { ConcreteValueNode, DataType } from "../../../OnnxTypes.js";
import {
    toArrayLike,
    uniq,
    addEdge,
    scalarOfType,
    tryAsConcreteValueNode,
    getFloatAttr,
    getIntAttr,
} from "../../../Utils.js";

/* ------------------------------ Handler ------------------------------- */
export default function gemmHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
    if (op.type !== "Gemm") return false;

    // Inputs in topo order
    const ins = op.getInputs() ?? [];
    if (ins.length < 2) {
        throw new Error(`[GemmHandler] Node ${op.id} missing required inputs (A, B).`);
    }

    const A = tryAsConcreteValueNode(ins[0]);
    const B = tryAsConcreteValueNode(ins[1]);
    const C = tryAsConcreteValueNode(ins[2]);

    if (!A || !B) {
        throw new Error(`[GemmHandler] Node ${op.id} has invalid A or B inputs.`);
    }

    // Single output tensor Y
    const outs = op.getOutputs();
    if (outs.length !== 1) return false;
    const Y = outs[0];

    // Attributes (defaults: alpha=1.0, beta=1.0, transA=0, transB=0)
    const alpha = getFloatAttr(op, "alpha", 1.0);
    const beta = getFloatAttr(op, "beta", 1.0);
    const transA = getIntAttr(op, "transA", 0) === 1 ? 1 : 0;
    const transB = getIntAttr(op, "transB", 0) === 1 ? 1 : 0;

    // DType selections
    const dtypeLeft = A.literalType as DataType;
    const dtypeRight = (C?.literalType ?? dtypeLeft) as DataType;

    /* ---------- optional Transpose on A/B ---------- */
    let A_in: ConcreteValueNode = A;
    let B_in: ConcreteValueNode = B;

    if (transA) {
        const tA = g
            .addNode(uniq(g, `Gemm_TA_${op.id}`))
            .init(new OperationNode.Builder("Transpose", [A], { perm: [1, 0] }))
            .as(OperationNode);
        const A_T = g
            .addNode(uniq(g, `Gemm_A_T_${op.id}`))
            .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
            .as(TensorNode);
        addEdge(g, tA, A_T, dtypeLeft);
        A_in = A_T;
    }

    if (transB) {
        const tB = g
            .addNode(uniq(g, `Gemm_TB_${op.id}`))
            .init(new OperationNode.Builder("Transpose", [B], { perm: [1, 0] }))
            .as(OperationNode);
        const B_T = g
            .addNode(uniq(g, `Gemm_B_T_${op.id}`))
            .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
            .as(TensorNode);
        addEdge(g, tB, B_T, dtypeLeft);
        B_in = B_T;
    }

    /* ------------------ MatMul(A', B') ------------------ */
    const mm = g
        .addNode(uniq(g, `Gemm_MM_${op.id}`))
        .init(new OperationNode.Builder("MatMul", [A_in, B_in], {}))
        .as(OperationNode);
    const MM = g
        .addNode(uniq(g, `Gemm_MM_T_${op.id}`))
        .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
        .as(TensorNode);
    addEdge(g, mm, MM, dtypeLeft);

    /* -------------------- scale by alpha -------------------- */
    let left: TensorNode.Class = MM;
    if (alpha !== 1.0) {
        const aC = scalarOfType(g, `Gemm_alpha_${op.id}`, alpha, dtypeLeft);
        const mulA = g
            .addNode(uniq(g, `Gemm_MulAlpha_${op.id}`))
            .init(new OperationNode.Builder("Mul", [MM, aC], {}))
            .as(OperationNode);
        const ScMM = g
            .addNode(uniq(g, `Gemm_ScaledMM_${op.id}`))
            .init(new TensorNode.Builder(dtypeLeft, [], "intermediate"))
            .as(TensorNode);
        addEdge(g, mulA, ScMM, dtypeLeft);
        left = ScMM;
    }

    /* -------------------- optional + beta*C -------------------- */
    let producedToY = false;

    if (C && beta !== 0.0) {
        let cTerm: ConcreteValueNode = C;
        if (beta !== 1.0) {
            const bC = scalarOfType(g, `Gemm_beta_${op.id}`, beta, dtypeRight);
            const mulB = g
                .addNode(uniq(g, `Gemm_MulBeta_${op.id}`))
                .init(new OperationNode.Builder("Mul", [C, bC], {}))
                .as(OperationNode);
            const ScC = g
                .addNode(uniq(g, `Gemm_ScaledC_${op.id}`))
                .init(new TensorNode.Builder(dtypeRight, [], "intermediate"))
                .as(TensorNode);
            addEdge(g, mulB, ScC, dtypeRight);
            cTerm = ScC;
        }

        // Y = left + cTerm
        const add = g
            .addNode(uniq(g, `Gemm_Add_${op.id}`))
            .init(new OperationNode.Builder("Add", [left, cTerm], {}))
            .as(OperationNode);
        g.addEdge(add, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
        producedToY = true;
    }

    if (!producedToY) {
        // No C-branch → wire 'left' directly to Y
        const srcOp = toArrayLike<OperationNode.Class>(
            left.getIncomers.sources.filterIs(OperationNode),
        )[0];
        g.addEdge(srcOp, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    }

    g.getNodeById(op.id)?.remove();

    return true;
}
