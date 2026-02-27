import type OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import type OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import type TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import {
    toStaticShape,
    makeTensorConst,
    scalarInt64,
    computeStrides,
    getAttr,
    toScalar,
} from "@specs-feup/onnx-flow/Onnx/Utils";
import type { LoopCtx } from "../BuildLoop.js";
import {
    resolveFusedInput,
    decodeMixedRadix,
    buildLinearIndex,
    unsqueezeIdx,
    ensureFlatInput,
    gatherFrom,
} from "../BuildLoop.js";
import type { ConcreteValueNode, KnownShape } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";

/* ============================== HANDLER ================================== */

export default function handleTranspose(
    op: OperationNode.Class,
    g: OnnxGraph.Class,
    ctx: LoopCtx,
): TensorNode.Class {
    const xIn = op.getInputs()![0];
    const X = resolveFusedInput(g, xIn, ctx, op, /*flatten*/ false, /*returnGather*/ false);

    const inShapeNum = toStaticShape(X.shape as KnownShape);
    const rank = inShapeNum.length;

    // Read perm safely, default to reverse if missing or wrong length
    let perm = getAttr(op, "perm");
    if (!Array.isArray(perm) || perm.length !== rank) {
        perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
    }

    // Precompute inverse perm (inversePerm[k] = output axis where input axis k lands)
    const inversePerm: number[] = new Array(rank);
    const validPerm = perm as number[];
    for (let outAxis = 0; outAxis < rank; outAxis++) {
        inversePerm[validPerm[outAxis]] = outAxis;
    }

    // Compute output shape (allow unknowns)
    const outShape = validPerm.map((p) => inShapeNum[p]);
    //ctx.outShape = outShape;

    // Mixed–radix decode in output space (unknown → 1 to keep arithmetic valid)
    const decodeDims = outShape.map((d) => (d > 0 ? d : 1));
    const oDigits = decodeMixedRadix(g, ctx.iter, decodeDims, `tp_${op.id}`);

    // Map back to input digits, honoring broadcast (input dim == 1 → use 0)
    const iDigits: ConcreteValueNode[] = [];
    for (let k = 0; k < rank; k++) {
        const inDim = inShapeNum[k] > 0 ? inShapeNum[k] : 1;
        if (inDim === 1) {
            const z = makeTensorConst(g, `tp_zero_${op.id}_${k}`, scalarInt64(0));
            iDigits.push(z);
        } else {
            const outPos = inversePerm[k]; // where input axis k appears after transpose
            iDigits.push(oDigits[outPos]);
        }
    }

    // Linearize and gather one element
    const strides = computeStrides(inShapeNum.map((d) => (d > 0 ? d : 1)));
    const lin = buildLinearIndex(g, iDigits, strides, `tp_lin_${op.id}`);
    const linU = unsqueezeIdx(g, lin, ctx.axes, `tp_linU_${op.id}`);

    const flat = ensureFlatInput(g, X);
    const [, gathered] = gatherFrom(g, flat, `tp_g_${op.id}`, linU, 0); // [1]

    // Force to scalar [] to ensure consistent rank 0 for downstream ops
    return toScalar(g, gathered, `tp_scalar_${op.id}`);
}
