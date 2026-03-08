import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { toStaticShape, scalarInt64, computeStrides, getAttr } from "../../../Utils.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import {
    resolveRecipeInput,
    decodeMixedRadix,
    buildLinearIndex,
    ensureFlatInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";

export class LowerTransposeRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        if (op.type !== "Transpose") return false;
        const inputs = op.getInputs();
        return !!inputs && inputs.length > 0;
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): ValueNode {
        const builder = new GraphBuilder(body, `tp_${op.id}`);
        const xIn = op.getInputs()![0];

        // Resolve input as a captured tensor without gathering yet
        const X = resolveRecipeInput(builder, xIn, valueMap, iter, axes, outShape, false, false);

        const inShapeNum = toStaticShape(X.shape as KnownShape);
        const rank = inShapeNum.length;

        // Determine permutation
        let perm = getAttr(op, "perm");
        if (!Array.isArray(perm) || perm.length !== rank) {
            perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        }

        // Compute inverse permutation to map output axes back to input axes
        const inversePerm: number[] = new Array(rank);
        const validPerm = perm as number[];
        for (let outAxis = 0; outAxis < rank; outAxis++) {
            inversePerm[validPerm[outAxis]] = outAxis;
        }

        const transposeOutShape = validPerm.map((p) => inShapeNum[p]);
        const decodeDims = transposeOutShape.map((d) => (d > 0 ? d : 1));

        // 1. Decode global iter into output-space coordinates
        const oDigits = decodeMixedRadix(builder, iter, decodeDims, `decode`);

        // 2. Map coordinates back to input-space
        const iDigits: ConcreteValueNode[] = [];
        for (let k = 0; k < rank; k++) {
            const inDim = inShapeNum[k] > 0 ? inShapeNum[k] : 1;
            if (inDim === 1) {
                iDigits.push(builder.createConstant(`zero_${k}`, scalarInt64(0)));
            } else {
                const outPos = inversePerm[k];
                iDigits.push(oDigits[outPos]);
            }
        }

        // 3. Linearize input coordinates and gather the scalar
        const strides = computeStrides(inShapeNum.map((d) => (d > 0 ? d : 1)));
        const lin = buildLinearIndex(builder, iDigits, strides, `lin`);
        const [linU] = builder.createOp("Unsqueeze", [lin, axes]);

        const flat = ensureFlatInput(builder, X);
        const [gathered] = builder.createOp("Gather", [flat, linU], { axis: 0 });

        // Ensure the result is a pure scalar []
        return squeezeIfLen1(builder, gathered, axes, `scalar`);
    }
}
