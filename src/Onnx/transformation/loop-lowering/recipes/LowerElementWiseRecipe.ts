import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    DataType,
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
} from "../../../OnnxTypes.js";
import type { LoopCarryContext, LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import {
    broadcastMixedShapes,
    extractMixedDims,
    packShapeVector,
    productMixed,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { asStaticDims, int64Vec, toStaticShape } from "../../../Utils.js";

export class LowerElementWiseRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerElementWise";
    public readonly targetOp = OpCategory.ElementWise;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["elementwise", "fuseable"] as const;
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
    ];

    match(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.ElementWise) return false;

        const inputs = op.getInputs() ?? [];
        if (
            inputs.length > 0 &&
            inputs.every((inp) => (inp.shape as unknown[] | undefined)?.length === 0)
        ) {
            return false;
        }

        return true;
    }

    getLoopBounds(
        op: OperationNode.Class,
        outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: number[] | ConcreteValueNode;
    } {
        const inputs = op.getInputs()!;
        const staticOut = asStaticDims(outShape);

        // 1. Static Case
        if (staticOut.length > 0 && !outShape.includes(-1)) {
            const totalIters = staticOut.reduce((a, b) => a * b, 1);
            return { totalIters, carryShape: staticOut };
        }

        // 2. Dynamic Case: O(rank) INT64 shape broadcasting (zero dummy tensor allocations)
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `ew_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        if (inputs.length === 0) return { totalIters: 1, carryShape: [1] };

        let targetShapeNode: ConcreteValueNode;
        let totalIters: ConcreteValueNode;
        if (inputs.length === 1) {
            [targetShapeNode] = builder.createOp("Shape", [inputs[0]]);
            const dims = extractMixedDims(
                builder,
                inputs[0],
                toStaticShape(inputs[0].shape),
                `ewb_0_${op.id}`,
            );
            totalIters = productMixed(builder, dims, `ewb_prod_${op.id}`);
        } else {
            const perInputDims = inputs.map((inp, i) =>
                extractMixedDims(builder, inp, toStaticShape(inp.shape), `ewb_${i}_${op.id}`),
            );
            const outDims = broadcastMixedShapes(builder, perInputDims, `ewb_bc_${op.id}`);
            if (outDims.length === 0) {
                return {
                    totalIters: 1,
                    carryShape: [1],
                    targetShape: int64Vec([]) as unknown as ConcreteValueNode,
                };
            }
            targetShapeNode = packShapeVector(builder, outDims, `ewb_tsh_${op.id}`);
            totalIters = productMixed(builder, outDims, `ewb_prod_${op.id}`);
        }

        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);

        return { totalIters, carryShape, targetShape: targetShapeNode };
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
        const builder = new GraphBuilder(body, `ew_${op.id}`);

        // 1. Resolve scalar inputs (safely handling optional/undefined inputs)
        const inputs = op
            .getInputs()!
            .map((inp) =>
                resolveRecipeInput(
                    builder,
                    inp,
                    valueMap,
                    iter,
                    axes,
                    outShape,
                    true,
                    true,
                    targetShapeNode,
                    loopCtx,
                ),
            );

        // Turn [1] -> [] (pure scalar) to match element-wise expectations in the loop body
        const effInputs = inputs.map((inp, i) =>
            squeezeIfLen1(builder, inp, axes, `${op.id}_in${i}_scalar`),
        );

        // 2. Perform the operation on the scalars
        // Pass op.attributes to preserve required attributes (like 'to' for Cast, 'alpha' for LeakyRelu)
        const [out] = builder.createOp(op.type, effInputs, op.attributes);

        return out!;
    }
}
