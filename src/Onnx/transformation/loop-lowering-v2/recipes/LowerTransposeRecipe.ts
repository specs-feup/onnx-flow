import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { toStaticShape, makeTensorConst, scalarInt64, computeStrides, getAttr, toScalar } from "../../../Utils.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput } from "../RecipeUtils.js";
import { decodeMixedRadix, buildLinearIndex, unsqueezeIdx, gatherFrom, ensureFlatInput } from "../../loop-lowering/BuildLoop.js";

export class LowerTransposeRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        // Reject Transpose if it's missing inputs
        if (op.type !== "Transpose" || !op.getInputs() || op.getInputs()!.length === 0) return false;
        return true;
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape
    ): ValueNode {
        const xIn = op.getInputs()![0];
        
        // Flattened outer tensor (no gather yet!)
        const X = resolveRecipeInput(body, xIn, valueMap, iter, axes, outShape, false, false);

        const inShapeNum = toStaticShape(X.shape as KnownShape);
        const rank = inShapeNum.length;

        let perm = getAttr(op, "perm");
        if (!Array.isArray(perm) || perm.length !== rank) {
            perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        }

        const inversePerm: number[] = new Array(rank);
        const validPerm = perm as number[];
        for (let outAxis = 0; outAxis < rank; outAxis++) {
            inversePerm[validPerm[outAxis]] = outAxis;
        }

        const transposeOutShape = validPerm.map((p) => inShapeNum[p]);
        const decodeDims = transposeOutShape.map((d) => (d > 0 ? d : 1));
        
        // Decode iter into output coordinates
        const oDigits = decodeMixedRadix(body, iter, decodeDims, `tp_${op.id}`);

        // Map back to input coordinates
        const iDigits: ConcreteValueNode[] = [];
        for (let k = 0; k < rank; k++) {
            const inDim = inShapeNum[k] > 0 ? inShapeNum[k] : 1;
            if (inDim === 1) {
                iDigits.push(makeTensorConst(body, `tp_zero_${op.id}_${k}`, scalarInt64(0)));
            } else {
                const outPos = inversePerm[k]; 
                iDigits.push(oDigits[outPos]);
            }
        }

        // Linearize and gather
        const strides = computeStrides(inShapeNum.map((d) => (d > 0 ? d : 1)));
        const lin = buildLinearIndex(body, iDigits, strides, `tp_lin_${op.id}`);
        const linU = unsqueezeIdx(body, lin, axes, `tp_linU_${op.id}`);

        const flat = ensureFlatInput(body, X);
        const [, gathered] = gatherFrom(body, flat, `tp_g_${op.id}`, linU, 0);

        return toScalar(body, gathered, `tp_scalar_${op.id}`);
    }
}