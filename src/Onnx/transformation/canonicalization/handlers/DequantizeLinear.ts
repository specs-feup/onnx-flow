import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { toArrayLike, uniq, addEdge, scalarOfType, constI64 } from "../../../Utils.js";

/* ------------------------------ Handler ------------------------------- */
/**
 * DequantizeLinear(x, scale[, zero_point], axis)
 *
 * Robust lowering (broadcast-safe for later loop linearizers):
 *   1) Cast X,S,Z to a common float type (floatT)
 *   2) **Expand S and Z to Shape(X)** — no Unsqueeze; this pre-resolves broadcasting
 *   3) y = (Xf - Zx) * Sx
 *
 * By expanding to Shape(X) explicitly, downstream elementwise-lowering passes that
 * flatten/gather per-element do **not** need to handle per-axis broadcasting math.
 */
export default function dequantizeLinearHandler(
    g: OnnxGraph.Class,
    op: OperationNode.Class,
): boolean {
    if (op.type !== "DequantizeLinear") return false;

    const ins = op.getInputs?.() ?? [];
    if (ins.length < 2) return false;

    const X = ins[0]?.is?.(TensorNode) ? ins[0].as(TensorNode) : undefined;
    const S = ins[1]?.is?.(TensorNode) ? ins[1].as(TensorNode) : undefined;
    const Z = ins[2]?.is?.(TensorNode) ? ins[2].as(TensorNode) : undefined;
    if (!X || !S) return false;

    // Single output tensor Y
    const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
    if (outs.length !== 1) return false;
    const Y = outs[0];

    // Attributes (axis for per-channel). Default 0 per ONNX spec.
    const a = op.getAttributes?.() ?? op.attributes ?? {};
    const axisAttr = Number(a.axis ?? 0);

    // Choose computation float dtype: prefer Y's float type, else FLOAT
    const floatSet = new Set([
        DataType.FLOAT,
        DataType.FLOAT16,
        DataType.BFLOAT16,
        DataType.DOUBLE,
    ]);
    const yT = (Y.literalType ?? DataType.FLOAT) as DataType;
    const floatT: DataType = floatSet.has(yT) ? yT : DataType.FLOAT;

    const xShape = X.shape ?? [];
    const rank = xShape.length;

    Y.setLiteralType(floatT);
    if (!Array.isArray(Y.shape) || Y.shape.length !== xShape.length) {
        Y.setShape(xShape.slice());
    }

    /* ---------------- Cast inputs to floatT ---------------- */
    const castX = g
        .addNode(uniq(g, `DQL_CastX_${op.id}`))
        .init(new OperationNode.Builder("Cast", [X], { to: floatT }))
        .as(OperationNode);
    const Xf = g
        .addNode(uniq(g, `DQL_Xf_${op.id}`))
        .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
        .as(TensorNode);
    addEdge(g, castX, Xf, floatT, X.shape);

    const castS = g
        .addNode(uniq(g, `DQL_CastS_${op.id}`))
        .init(new OperationNode.Builder("Cast", [S], { to: floatT }))
        .as(OperationNode);
    const Sf = g
        .addNode(uniq(g, `DQL_Sf_${op.id}`))
        .init(new TensorNode.Builder(floatT, S.shape, "intermediate"))
        .as(TensorNode);
    addEdge(g, castS, Sf, floatT, S.shape);

    let Zf: TensorNode.Class;
    if (Z) {
        const castZ = g
            .addNode(uniq(g, `DQL_CastZ_${op.id}`))
            .init(new OperationNode.Builder("Cast", [Z], { to: floatT }))
            .as(OperationNode);
        Zf = g
            .addNode(uniq(g, `DQL_Zf_${op.id}`))
            .init(new TensorNode.Builder(floatT, Z.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, castZ, Zf, floatT, Z.shape);
    } else {
        Zf = scalarOfType(g, `DQL_Zzero_${op.id}`, 0, floatT);
    }

    /* ---- Align S and Z per ONNX axis semantics, then Expand to Shape(X) ---- */
    const shapeXop = g
        .addNode(uniq(g, `DQL_ShapeX_${op.id}`))
        .init(new OperationNode.Builder("Shape", [Xf], {}))
        .as(OperationNode);
    const shapeX = g
        .addNode(uniq(g, `DQL_shapeX_${op.id}`))
        .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
        .as(TensorNode);
    addEdge(g, shapeXop, shapeX, DataType.INT64, [rank]);

    const sRank = S.shape?.length ?? 0;

    // Per-tensor: true scalar, or a rank-1 tensor of length 1
    const singleLen =
        sRank === 1 && typeof S.shape?.[0] === "number" && (S.shape![0] as number) === 1;

    const perTensor = sRank === 0 || singleLen;
    const perAxis = !perTensor && sRank === 1;

    let Sx: TensorNode.Class = Sf;
    let Zx: TensorNode.Class = Zf;

    if (perAxis) {
        const axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
        if (axis < 0 || axis >= rank) return false;

        // Optional static length check if known
        const xAxisDim = typeof xShape?.[axis] === "number" ? (xShape![axis] as number) : undefined;

        // Build axes tensor: unsqueeze on every dim except 'axis'
        const axesVals = [];
        for (let i = 0; i < rank; i++) {
            if (i !== axis) axesVals.push(i);
        }
        const axes = constI64(g, `DQL_axes_${op.id}`, axesVals);

        // Shape for S after Unsqueeze: [1, ..., |S|, ..., 1]
        const sRankedShape: (number | string | undefined)[] = Array(rank).fill(1);

        // axis dim = length of S, or X's axis dim as a fallback
        const sLen =
            Array.isArray(S.shape) && typeof S.shape[0] === "number"
                ? S.shape[0]
                : xShape && typeof xShape[axis] === "number"
                  ? xShape[axis]
                  : undefined;

        if (sLen !== undefined && xAxisDim !== undefined && sLen !== xAxisDim) return false;

        if (rank > 0) {
            sRankedShape[axis] = sLen;
        }

        const uSop = g
            .addNode(uniq(g, `DQL_unsqS_${op.id}`))
            .init(new OperationNode.Builder("Unsqueeze", [Sf, axes]))
            .as(OperationNode);
        const Sranked = g
            .addNode(uniq(g, `DQL_Srank_${op.id}`))
            .init(new TensorNode.Builder(floatT, sRankedShape, "intermediate"))
            .as(TensorNode);
        addEdge(g, uSop, Sranked, floatT, sRankedShape);

        const uZop = g
            .addNode(uniq(g, `DQL_unsqZ_${op.id}`))
            .init(new OperationNode.Builder("Unsqueeze", [Zf, axes]))
            .as(OperationNode);

        // Same broadcast shape as Sranked
        const zRankedShape = [...sRankedShape];

        const Zranked = g
            .addNode(uniq(g, `DQL_Zrank_${op.id}`))
            .init(new TensorNode.Builder(floatT, zRankedShape, "intermediate"))
            .as(TensorNode);
        addEdge(g, uZop, Zranked, floatT, zRankedShape);

        // Finally Expand to exactly Shape(X)
        const expSop = g
            .addNode(uniq(g, `DQL_ExpandS_${op.id}`))
            .init(new OperationNode.Builder("Expand", [Sranked, shapeX], {}))
            .as(OperationNode);
        Sx = g
            .addNode(uniq(g, `DQL_Sx_${op.id}`))
            .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, expSop, Sx, floatT, X.shape);

        const expZop = g
            .addNode(uniq(g, `DQL_ExpandZ_${op.id}`))
            .init(new OperationNode.Builder("Expand", [Zranked, shapeX], {}))
            .as(OperationNode);
        Zx = g
            .addNode(uniq(g, `DQL_Zx_${op.id}`))
            .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, expZop, Zx, floatT, X.shape);
    } else {
        // per-tensor: directly Expand scalar S and (scalar or 1D) Z
        const expSop = g
            .addNode(uniq(g, `DQL_ExpandS_${op.id}`))
            .init(new OperationNode.Builder("Expand", [Sf, shapeX], {}))
            .as(OperationNode);
        Sx = g
            .addNode(uniq(g, `DQL_Sx_${op.id}`))
            .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, expSop, Sx, floatT, X.shape);

        const expZop = g
            .addNode(uniq(g, `DQL_ExpandZ_${op.id}`))
            .init(new OperationNode.Builder("Expand", [Zf, shapeX], {}))
            .as(OperationNode);
        Zx = g
            .addNode(uniq(g, `DQL_Zx_${op.id}`))
            .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, expZop, Zx, floatT, X.shape);
    }

    /* ---------------- y = (Xf - Zx) * Sx ---------------- */
    const sub = g
        .addNode(uniq(g, `DQL_Sub_${op.id}`))
        .init(new OperationNode.Builder("Sub", [Xf, Zx], {}))
        .as(OperationNode);
    const D = g
        .addNode(uniq(g, `DQL_D_${op.id}`))
        .init(new TensorNode.Builder(floatT, X.shape, "intermediate"))
        .as(TensorNode);
    addEdge(g, sub, D, floatT, X.shape);

    const mul = g
        .addNode(uniq(g, `DQL_Mul_${op.id}`))
        .init(new OperationNode.Builder("Mul", [D, Sx], {}))
        .as(OperationNode);

    g.addEdge(mul, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);

    g.getNodeById(op.id).remove();

    return true;
}
