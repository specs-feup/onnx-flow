import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import type { AttributeValue, ConcreteValueNode } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import { addEdge, scalarOfType, tensorOnesConst, toArrayLike, uniq } from "../../../Utils.js";

export default function averagePoolHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
    if (op.type !== "AveragePool") return false;

    // 1. Validate Inputs
    const ins = op.getInputs() ?? [];
    if (ins.length !== 1) {
        throw new Error(
            `[AveragePoolHandler] Node ${op.id} must have exactly 1 input (X). Got ${ins.length}.`,
        );
    }

    const X = ins[0]?.tryAs(TensorNode);

    if (!X) {
        throw new Error("Expected first input to be a valid TensorNode.");
        // OR return early: return;
    }

    // 2. Validate Outputs
    const outs = toArrayLike<TensorNode.Class>(op.getOutgoers.targets.filterIs(TensorNode));
    if (outs.length !== 1) return false;
    const Y = outs[0];

    // 3. Shape Analysis
    const xShape = X.shape;
    const rank = xShape.length;
    if (rank !== 4) return false; // Currently only supporting 2D NCHW (Rank 4)

    const C = xShape[1];
    if (typeof C !== "number") return false; // Need concrete channel count

    const dtype = X.literalType as DataType;

    // 4. Parse Attributes (Strictly)
    const attrs = op.getAttributes();

    // Kernel Shape (Required)
    const kernelShape = attrs["kernel_shape"] as number[] | undefined;
    if (!kernelShape || kernelShape.length !== 2) return false;
    const [kH, kW] = kernelShape.map(Number);

    // Strides (Optional, default 1)
    const strides = "strides" in attrs ? (attrs["strides"] as number[]) : [1, 1];
    const [sH, sW] = strides.map(Number);

    // Pads (Optional, default 0)
    // Note: AutoPad logic handling is simplified here for clarity
    const pads = "pads" in attrs ? (attrs["pads"] as number[]) : [0, 0, 0, 0];
    const [pT, pL, pB, pR] = pads.length === 4 ? pads.map(Number) : [0, 0, 0, 0];

    const autoPad = "auto_pad" in attrs ? (attrs["auto_pad"] as string) : "NOTSET";
    const countIncludePad = Number(attrs["count_include_pad"] ?? 0);
    const ceilMode = Number(attrs["ceil_mode"] ?? 0);

    // 5. Optimization Heuristic: Tiled Global Pool
    // If this looks like a global pool split into tiles, leave it for the loop-lowering pass.
    if (
        autoPad === "NOTSET" &&
        ceilMode === 0 &&
        pads.every((p) => p === 0) &&
        kH === sH &&
        kW === sW
    ) {
        return false;
    }

    // 6. Rewrite to Conv
    // Strategy: Sum = Conv(X, Ones); Count = Conv(OnesLikeX, Ones) or Const; Y = Sum / Count

    // A. Create Weight Tensor (Ones) -> [C, 1, kH, kW]
    const Wones = tensorOnesConst(g, `AvgPool_W_${op.id}`, dtype, [C, 1, kH, kW]);

    // B. Conv Attributes
    const convAttrs: Record<string, AttributeValue> = {
        group: C,
        strides: [sH, sW],
    };
    if (autoPad !== "NOTSET") {
        convAttrs["auto_pad"] = autoPad;
    } else {
        convAttrs["pads"] = [pT, pL, pB, pR];
    }

    // C. Compute Sum
    const convSum = g
        .addNode(uniq(g, `AvgPool_Sum_${op.id}`))
        .init(new OperationNode.Builder("Conv", [X, Wones], convAttrs))
        .as(OperationNode);

    const sumOut = g
        .addNode(uniq(g, `AvgPool_SumT_${op.id}`))
        .init(new TensorNode.Builder(dtype, Y.shape, "intermediate"))
        .as(TensorNode);
    addEdge(g, convSum, sumOut, dtype, Y.shape);

    // D. Compute Divisor
    let divisor: ConcreteValueNode;

    if (countIncludePad === 1 || autoPad === "VALID") {
        // Simple case: Divide by kernel area
        const area = kH * kW;
        divisor = scalarOfType(g, `AvgPool_Divisor_${op.id}`, area, dtype);
    } else {
        // Complex case: We must count valid pixels (excluding padding)
        // 1. Create a mask of Ones with shape X
        const shapeOp = g
            .addNode(uniq(g, `AvgPool_Shape_${op.id}`))
            .init(new OperationNode.Builder("Shape", [X], {}))
            .as(OperationNode);
        const shapeT = g
            .addNode(uniq(g, `AvgPool_ShapeT_${op.id}`))
            .init(new TensorNode.Builder(DataType.INT64, [rank], "intermediate"))
            .as(TensorNode);
        addEdge(g, shapeOp, shapeT, DataType.INT64, [rank]);

        const oneSc = scalarOfType(g, `AvgPool_OneSc_${op.id}`, 1, dtype);
        const expand = g
            .addNode(uniq(g, `AvgPool_Expand_${op.id}`))
            .init(new OperationNode.Builder("Expand", [oneSc, shapeT], {}))
            .as(OperationNode);
        const mask = g
            .addNode(uniq(g, `AvgPool_Mask_${op.id}`))
            .init(new TensorNode.Builder(dtype, X.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, expand, mask, dtype, X.shape);

        // 2. Convolve Mask with OnesKernel (counts valid overlaps)
        const convCount = g
            .addNode(uniq(g, `AvgPool_Count_${op.id}`))
            .init(new OperationNode.Builder("Conv", [mask, Wones], convAttrs))
            .as(OperationNode);

        divisor = g
            .addNode(uniq(g, `AvgPool_CountT_${op.id}`))
            .init(new TensorNode.Builder(dtype, Y.shape, "intermediate"))
            .as(TensorNode);
        addEdge(g, convCount, divisor, dtype, Y.shape);
    }

    // E. Final Divide
    const divOp = g
        .addNode(uniq(g, `AvgPool_Div_${op.id}`))
        .init(new OperationNode.Builder("Div", [sumOut, divisor], {}))
        .as(OperationNode);

    // Replace old edge
    addEdge(g, divOp, Y, dtype, Y.shape);

    // Remove original node
    g.getNodeById(op.id)?.remove();

    return true;
}
