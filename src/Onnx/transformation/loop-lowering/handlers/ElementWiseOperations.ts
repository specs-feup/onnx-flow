import OnnxEdge from "@specs-feup/onnx-flow/Onnx/OnnxEdge";
import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { uniq, toStaticShape, Shape, getLargestRankShape } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopCtx, resolveFusedInput, squeezeIfLen1, broadcastShapes } from "../BuildLoop.js";

/* ============================== HANDLER ================================== */

export default function handleElementWiseOperation(
    op: OperationNode.Class,
    g: OnnxGraph.Class,
    ctx: LoopCtx,
): TensorNode.Class {
    const inputs = op.getInputs()!.map((inp) => resolveFusedInput(g, inp, ctx, op));

    // Turn [1] -> [] (scalar) when allowed; leave other shapes alone
    const effInputs = inputs.map((inp, i) =>
        squeezeIfLen1(g, inp, ctx.axes, `${op.id}_in${i}_scalar`),
    );

    const node = g
        .addNode(uniq(g, `${op.type}_${op.id}`))
        .init(new OperationNode.Builder(op.type, effInputs))
        .as(OperationNode);

    const allScalars = effInputs.every((t) => t.shape.length === 0);
    let outShape: (number | string)[];
    if (allScalars) {
        outShape = [];
    } else {
        // Try true broadcast; if any unknown, fall back to "largest rank" (safe)
        try {
            const shapes = effInputs.map((t) => toStaticShape(t.shape as Shape));
            const bshape = broadcastShapes(shapes);
            outShape = bshape;
        } catch {
            outShape = getLargestRankShape(effInputs);
        }
    }

    const out = g
        .addNode(uniq(g, `${op.id}_out`))
        .init(new TensorNode.Builder(inputs[0].literalType, outShape, "intermediate"))
        .as(TensorNode);

    g.addEdge(node, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

    // Gate ONLY when we’re in a coalesced + fused chain
    if (ctx.coalesce && ctx.gateByK && ctx.kIdx && ctx.kM1) {
        const eqNode = g
            .addNode(uniq(g, `eq_k_last_${op.id}`))
            .init(new OperationNode.Builder("Equal", [ctx.kIdx, ctx.kM1]))
            .as(OperationNode);
        const eqOut = g
            .addNode(uniq(g, `eq_k_last_${op.id}_out`))
            .init(new TensorNode.Builder(DataType.BOOL, [], "intermediate"))
            .as(TensorNode);
        g.addEdge(eqNode, eqOut)
            .init(new OnnxEdge.Builder(eqOut.literalType, eqOut.shape))
            .as(OnnxEdge);

        const passthrough = ctx.running ?? inputs[0];

        // Where(eq, applied_out, passthrough_left_input)
        const whereNode = g
            .addNode(uniq(g, `gate_${op.type}_${op.id}`))
            .init(new OperationNode.Builder("Where", [eqOut, out, passthrough]))
            .as(OperationNode);
        const gated = g
            .addNode(uniq(g, `gated_${op.id}`))
            .init(new TensorNode.Builder(passthrough.literalType, outShape, "intermediate"))
            .as(TensorNode);
        g.addEdge(whereNode, gated)
            .init(new OnnxEdge.Builder(gated.literalType, gated.shape))
            .as(OnnxEdge);

        return gated;
    }

    return out;
}
