import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { toStaticShape, scalarInt64, getAttr, int64Vec, asStaticDims } from "../../../Utils.js";
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

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: number[] | ConcreteValueNode;
    } {
        const inputs = op.getInputs()!;
        const rank = inputs[0].shape.length;

        // 1. Static Case
        if (!inputs[0].shape.includes(-1) && rank > 0) {
            const inShapeStatic = asStaticDims(inputs[0].shape);
            const totalIters = inShapeStatic.reduce((a, b) => a * b, 1);
            return { totalIters, carryShape: [totalIters] };
        }

        // 2. Dynamic Case
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `tp_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        const [shapeNode] = builder.createOp("Shape", [inputs[0]]);
        const [totalIters] = builder.createOp("ReduceProd", [shapeNode, axes0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0]);

        let perm = getAttr(op, "perm");
        if (!Array.isArray(perm) || perm.length !== rank) {
            perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        }
        const permConst = builder.createConstant(`perm_${op.id}`, int64Vec(perm as number[]));
        const [targetShape] = builder.createOp("Gather", [shapeNode, permConst], { axis: 0 });

        return { totalIters, carryShape, targetShape };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        _carryNode: ConcreteValueNode,
        targetShapeNode: ValueNode,
    ): ValueNode {
        const builder = new GraphBuilder(body, `tp_${op.id}`);
        const xIn = op.getInputs()![0];

        // Resolve input as a captured tensor without gathering yet
        const X = resolveRecipeInput(
            builder,
            xIn,
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
        );

        const inShapeNum = toStaticShape(X.shape as KnownShape);
        const rank = inShapeNum.length;

        // Determine permutation
        let perm = getAttr(op, "perm");
        if (!Array.isArray(perm) || perm.length !== rank) {
            perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        }

        const inversePerm: number[] = new Array(rank);
        const validPerm = perm as number[];
        for (let outAxis = 0; outAxis < rank; outAxis++) {
            inversePerm[validPerm[outAxis]] = outAxis;
        }

        // --- NEW DYNAMIC SUPPORT ---
        let decodeDims: (number | ValueNode)[] = [];
        let inDims: (number | ValueNode)[] = [];

        if (inShapeNum.includes(-1)) {
            // Dynamic shape: generate Shape node and extract individual dimensions
            const [shapeNode] = builder.createOp("Shape", [X]);
            for (let i = 0; i < rank; i++) {
                const iConst = builder.createConstant(`dim_idx_${i}`, int64Vec([i]));
                const [dimValRaw] = builder.createOp("Gather", [shapeNode, iConst], { axis: 0 });
                const [dimVal] = builder.createOp("Squeeze", [dimValRaw, axes]);
                inDims.push(dimVal);
            }
            decodeDims = validPerm.map((p) => inDims[p]);
        } else {
            // Static fallback
            const transposeOutShape = validPerm.map((p) => inShapeNum[p]);
            decodeDims = transposeOutShape.map((d) => (d > 0 ? d : 1));
            inDims = inShapeNum.map((d) => (d > 0 ? d : 1));
        }
        // ---------------------------

        // 1. Decode global iter into output-space coordinates
        const oDigits = decodeMixedRadix(builder, iter, decodeDims, `decode`);

        // 2. Map coordinates back to input-space
        const iDigits: ConcreteValueNode[] = [];
        for (let k = 0; k < rank; k++) {
            const outPos = inversePerm[k];
            iDigits.push(oDigits[outPos]);
        }

        // 3. Linearize input coordinates and gather the scalar
        // We calculate strides dynamically by accumulating (CumProd-like approach) backwards
        const strides: (number | ValueNode)[] = new Array(rank).fill(1);
        let currentStride: ValueNode | number = 1;

        for (let i = rank - 1; i >= 0; i--) {
            strides[i] = currentStride;
            if (i > 0) {
                if (typeof currentStride === "number" && typeof inDims[i] === "number") {
                    currentStride = currentStride * (inDims[i] as number);
                } else {
                    const cNode =
                        typeof currentStride === "number"
                            ? builder.createConstant(`cs_${i}`, scalarInt64(currentStride))
                            : currentStride;
                    const dNode =
                        typeof inDims[i] === "number"
                            ? builder.createConstant(`cd_${i}`, scalarInt64(inDims[i] as number))
                            : (inDims[i] as ValueNode);
                    [currentStride] = builder.createOp("Mul", [cNode, dNode]);
                }
            }
        }

        const lin = buildLinearIndex(builder, iDigits, strides, `lin`);
        const [linU] = builder.createOp("Unsqueeze", [lin, axes]);

        const flat = ensureFlatInput(builder, X);
        const [gathered] = builder.createOp("Gather", [flat, linU], { axis: 0 });

        return squeezeIfLen1(builder, gathered, axes, `scalar`);
    }
}
