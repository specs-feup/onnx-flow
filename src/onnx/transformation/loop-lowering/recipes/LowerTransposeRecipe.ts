import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import {
    toStaticShape,
    scalarInt64,
    getAttr,
    int64Vec,
    asStaticDims,
} from "../../../utils/index.js";
import type { LoopCarryContext, LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import {
    resolveRecipeInput,
    decodeMixedRadix,
    buildLinearIndex,
    ensureFlatInput,
    gatherScalar,
    buildAguPointer,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerTransposeRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerTranspose";
    public readonly targetOp = "Transpose";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["data-movement", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Size",
        "Reshape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "And",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
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
        loopCtx?: LoopCarryContext,
    ): ValueNode {
        const builder = new GraphBuilder(body, `tp_${op.id}`);
        const xIn = op.getInputs()![0];

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
            loopCtx,
        );

        const inShapeNum = toStaticShape(X.shape as KnownShape);
        const rank = inShapeNum.length;

        let perm = getAttr(op, "perm");
        if (!Array.isArray(perm) || perm.length !== rank) {
            perm = Array.from({ length: rank }, (_, i) => rank - 1 - i);
        }
        const validPerm = perm as number[];

        let decodeDims: (number | ValueNode)[] = [];
        let inDims: (number | ValueNode)[] = [];

        if (inShapeNum.includes(-1)) {
            const [shapeNode] = builder.createOp("Shape", [X]);
            for (let i = 0; i < rank; i++) {
                const iConst = builder.createConstant(`dim_idx_${i}`, scalarInt64(i));
                inDims.push(gatherScalar(builder, shapeNode, iConst));
            }
            decodeDims = validPerm.map((p) => inDims[p]);
        } else {
            const transposeOutShape = validPerm.map((p) => inShapeNum[p]);
            decodeDims = transposeOutShape.map((d) => (d > 0 ? d : 1));
            inDims = inShapeNum.map((d) => (d > 0 ? d : 1));
        }

        // Compute input strides
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

        let lin: ConcreteValueNode;
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            // Permute input strides to match output iteration order
            const mappedStrides = validPerm.map((p) => strides[p]);
            lin = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                decodeDims,
                mappedStrides,
                `tp_${op.id}`,
            );
        } else {
            const inversePerm: number[] = new Array(rank);
            for (let outAxis = 0; outAxis < rank; outAxis++) {
                inversePerm[validPerm[outAxis]] = outAxis;
            }
            const oDigits = decodeMixedRadix(builder, iter, decodeDims, `decode`);
            const iDigits: ConcreteValueNode[] = [];
            for (let k = 0; k < rank; k++) {
                iDigits.push(oDigits[inversePerm[k]]);
            }
            lin = buildLinearIndex(builder, iDigits, strides, `lin`);
        }

        const flat = ensureFlatInput(builder, X);
        return gatherScalar(builder, flat, lin);
    }
}
