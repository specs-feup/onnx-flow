import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    readConstIntegerVectorFromTensorNode,
    uniq,
    maybeRemoveOrphanConstant,
    scalarI64,
} from "../../../Utils.js";
import ConstantNode from "@specs-feup/onnx-flow/Onnx/ConstantNode";

// ---------- Handler ----------
export default function sliceHandler(g: OnnxGraph.Class, sl: OperationNode.Class): boolean {
    if (sl.type !== "Slice") return false;

    // 1. Inputs (Standard Opset 10+: data, starts, ends, axes?, steps?)
    const ins = sl.getInputs?.() ?? [];

    // Strict Input Check (Phase 5.2): Slice MUST have at least 3 inputs (data, starts, ends).
    if (ins.length < 3) {
        throw new Error(
            `[SliceHandler] Node ${sl.id} is missing required inputs. Expected 3 (data, starts, ends), got ${ins.length}. Adapter failure?`,
        );
    }

    const Xn = ins[0];
    if (!Xn?.is?.(TensorNode) && !Xn?.is?.(ConstantNode)) {
        throw new Error(`[SliceHandler] Node ${sl.id} input[0] (data) is invalid.`);
    }

    const Xin = Xn.is(TensorNode) ? Xn.as(TensorNode) : Xn.as(ConstantNode);
    const inShape = Xin.shape.map((d) => (typeof d === "number" ? d : 1));
    const rank = inShape.length;

    // 2. Read Parameters (Strictly from Inputs)
    const readVec = (idx: number) => {
        const t = ins[idx];
        // If the node should exist (idx 1 or 2) but is undefined, it's a structural error
        if (!t && (idx === 1 || idx === 2)) {
            throw new Error(`[SliceHandler] Node ${sl.id} missing required input at index ${idx}.`);
        }

        if (t?.is?.(ConstantNode)) {
            return readConstIntegerVectorFromTensorNode(t.as(ConstantNode));
        }
        return t?.is?.(TensorNode)
            ? readConstIntegerVectorFromTensorNode(t.as(TensorNode))
            : undefined;
    };

    const starts = readVec(1);
    const ends = readVec(2);
    let axes = readVec(3);
    let steps = readVec(4);

    // If starts/ends are dynamic (not constant), we cannot compile this static slice logic.
    // We return false here (optimization failure), but we DO NOT throw, because valid ONNX allows dynamic slice.
    if (!starts || !ends) {
        return false;
    }

    // Defaults
    if (!axes) axes = Array.from({ length: starts.length }, (_, i) => i);
    if (!steps) steps = new Array(axes.length).fill(1);

    // 3. Normalize to full rank vectors
    // We build arrays of size [rank] where indices NOT in 'axes' correspond to full slices (0, dim, 1).
    const fullStarts = new Array(rank).fill(0);
    const fullEnds = inShape.slice();
    const fullSteps = new Array(rank).fill(1);

    for (let i = 0; i < axes.length; i++) {
        const ax = axes[i];
        if (ax < 0 || ax >= rank) continue; // Should not happen in valid ONNX

        const dim = inShape[ax]; // Dimension size
        const dimVal = dim > 0 ? dim : 2147483647; // Handle unknown dim safely if needed

        let s = Number(starts[i]);
        let e = Number(ends[i]);
        const stp = Number(steps[i]);

        if (stp === 0) return false; // Invalid step

        // Normalize negatives
        if (s < 0) s += dimVal;
        if (e < 0) e += dimVal;

        // Clamp
        if (stp > 0) {
            s = Math.max(0, Math.min(s, dimVal));
            e = Math.max(0, Math.min(e, dimVal));
        } else {
            s = Math.min(dimVal - 1, Math.max(s, 0));
            e = Math.min(dimVal - 1, Math.max(e, -1));
        }

        fullStarts[ax] = s;
        fullEnds[ax] = e;
        fullSteps[ax] = stp;
    }

    // 4. Output
    const outs = sl.getOutgoers.targets ?? [];
    if (outs.length !== 1 || !outs[0].is?.(TensorNode)) return false;
    const Y = outs[0].as(TensorNode);

    // 5. Determine affected axes
    const changingAxes: number[] = [];
    for (let ax = 0; ax < rank; ax++) {
        const s = fullStarts[ax];
        const e = fullEnds[ax];
        const stp = fullSteps[ax];
        const dim = inShape[ax];

        // Is this a no-op slice on this axis? (Start=0, End=Dim, Step=1)
        const isNoOp = s === 0 && e === dim && stp === 1;
        if (!isNoOp) {
            changingAxes.push(ax);
        }
    }

    // 6. Rewrite
    if (changingAxes.length === 0) {
        // Identity Rewrite
        const id = g
            .addNode(uniq(g, `Slice_Id_${sl.id}`))
            .init(new OperationNode.Builder("Identity", [Xin], {}))
            .as(OperationNode);
        g.addEdge(id, Y).init(new OnnxEdge.Builder(Y.literalType, Y.shape)).as(OnnxEdge);
    } else {
        // Chain of Range + Gather
        let curT: TensorNode.Class | ConstantNode.Class = Xin;

        for (let i = 0; i < changingAxes.length; i++) {
            const ax = changingAxes[i];
            const s = fullStarts[ax];
            const e = fullEnds[ax];
            const stp = fullSteps[ax];

            const cS = scalarI64(g, `Slice_S_${sl.id}_${ax}`, s);
            const cE = scalarI64(g, `Slice_E_${sl.id}_${ax}`, e);
            const cStep = scalarI64(g, `Slice_Step_${sl.id}_${ax}`, stp);

            const range = g
                .addNode(uniq(g, `Slice_Range_${sl.id}_${ax}`))
                .init(new OperationNode.Builder("Range", [cS, cE, cStep], {}))
                .as(OperationNode);

            // Length calculation
            const len = Math.max(0, Math.ceil((e - s) / stp));
            const idx = g
                .addNode(uniq(g, `Slice_Idx_${sl.id}_${ax}`))
                .init(new TensorNode.Builder(DataType.INT64, [len], "intermediate"))
                .as(TensorNode);
            g.addEdge(range, idx)
                .init(new OnnxEdge.Builder(DataType.INT64, [len]))
                .as(OnnxEdge);

            const gather = g
                .addNode(uniq(g, `Slice_Gather_${sl.id}_${ax}`))
                .init(new OperationNode.Builder("Gather", [curT, idx], { axis: ax }))
                .as(OperationNode);

            const isLast = i === changingAxes.length - 1;
            if (isLast) {
                g.addEdge(gather, Y)
                    .init(new OnnxEdge.Builder(Y.literalType, Y.shape))
                    .as(OnnxEdge);
            } else {
                const mid = g
                    .addNode(uniq(g, `Slice_Mid_${sl.id}_${ax}`))
                    .init(new TensorNode.Builder(curT.literalType, [], "intermediate")) // Shape inferred later or ignored
                    .as(TensorNode);
                g.addEdge(gather, mid)
                    .init(new OnnxEdge.Builder(mid.literalType, mid.shape))
                    .as(OnnxEdge);
                curT = mid;
            }
        }
    }

    // 7. Clean up
    g.getNodeById(sl.id).remove();

    // Cleanup orphaned constant inputs
    const getInput = (i: number) =>
        ins[i]?.is?.(TensorNode)
            ? ins[i].as(TensorNode)
            : ins[i]?.is?.(ConstantNode)
              ? ins[i].as(ConstantNode)
              : undefined;
    maybeRemoveOrphanConstant(g, getInput(1)); // starts
    maybeRemoveOrphanConstant(g, getInput(2)); // ends
    maybeRemoveOrphanConstant(g, getInput(3)); // axes
    maybeRemoveOrphanConstant(g, getInput(4)); // steps

    return true;
}
