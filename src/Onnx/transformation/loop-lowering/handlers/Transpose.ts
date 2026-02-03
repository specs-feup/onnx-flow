import OnnxGraph from "@specs-feup/onnx-flow/Onnx/OnnxGraph";
import { DataType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";
import OperationNode from "@specs-feup/onnx-flow/Onnx/OperationNode";
import TensorNode from "@specs-feup/onnx-flow/Onnx/TensorNode";
import { toStaticShape, makeTensorConst, scalarInt64, computeStrides } from "@specs-feup/onnx-flow/Onnx/Utils";
import { LoopCtx, resolveFusedInput, decodeMixedRadix, buildLinearIndex, unsqueezeIdx, ensureFlatInput, gatherFrom, squeezeIfLen1 } from "../BuildLoop.js";

/* ============================== HANDLER ================================== */

export default function handleTranspose(
  op: OperationNode.Class,
  g: OnnxGraph.Class,
  ctx: LoopCtx
): TensorNode.Class {
    function getAttr<T = any>(op: OperationNode.Class, key: string, dflt?: T): T | undefined {
    // Try common access patterns in your IR
    const anyOp: any = op as any;
    if (typeof anyOp.getAttributes === 'function') {
      const obj = anyOp.getAttributes();
      if (obj && key in obj) return obj[key];
    }
    if (typeof anyOp.getAttribute === 'function') {
      const v = anyOp.getAttribute(key);
      if (v !== undefined) return v;
    }
    if (anyOp.attributes && key in anyOp.attributes) {
      return anyOp.attributes[key];
    }
    return dflt;
  }

  const xIn = op.getInputs()![0];
  const X = resolveFusedInput(g, xIn, ctx, op, /*flatten*/ false, /*returnGather*/ false);

  const inShapeNum = toStaticShape(X.shape as (number | String)[]);
  const rank = inShapeNum.length;

  // Read perm safely, default to reverse if missing or wrong length
  let perm = getAttr<number[]>(op, "perm");
  if (!Array.isArray(perm) || perm.length !== rank) {
    perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
  }

  // Precompute inverse perm (inversePerm[k] = output axis where input axis k lands)
  const inversePerm: number[] = new Array(rank);
  for (let outAxis = 0; outAxis < rank; outAxis++) {
    inversePerm[perm[outAxis]] = outAxis;
  }

  // Compute output shape (allow unknowns)
  const outShape = perm.map(p => inShapeNum[p]);
  //ctx.outShape = outShape;

  // Mixed–radix decode in output space (unknown → 1 to keep arithmetic valid)
  const decodeDims = outShape.map(d => (d > 0 ? d : 1));
  const oDigits = decodeMixedRadix(g, ctx.iter, decodeDims, `tp_${op.id}`);

  // Map back to input digits, honoring broadcast (input dim == 1 → use 0)
  const iDigits: TensorNode.Class[] = [];
  for (let k = 0; k < rank; k++) {
    const inDim = inShapeNum[k] > 0 ? inShapeNum[k] : 1;
    if (inDim === 1) {
      const z = makeTensorConst(g, `tp_zero_${op.id}_${k}`, DataType.INT64, "constant", scalarInt64(0));
      iDigits.push(z);
    } else {
      const outPos = inversePerm[k]; // where input axis k appears after transpose
      // Guard against malformed perms just in case
      if (outPos === undefined) {
        const z = makeTensorConst(g, `tp_safezero_${op.id}_${k}`, DataType.INT64, "constant", scalarInt64(0));
        iDigits.push(z);
      } else {
        iDigits.push(oDigits[outPos]);
      }
    }
  }

  // Linearize and gather one element
  const strides = computeStrides(inShapeNum.map(d => (d > 0 ? d : 1)));
  const lin = buildLinearIndex(g, iDigits, strides, `tp_lin_${op.id}`);
  const linU = unsqueezeIdx(g, lin, ctx.axes, `tp_linU_${op.id}`);

  const flat = ensureFlatInput(g, X);
  const [_, gathered] = gatherFrom(g, flat, `tp_g_${op.id}`, linU, 0); // [1]
  return squeezeIfLen1(g, gathered, ctx.axes, `tp_sq_${op.id}`); // []
}