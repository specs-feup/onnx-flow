import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import { resolveRecipeInput } from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { int64Vec, scalarInt64, readScalarFromTensorNode } from "../../../Utils.js";
import ConstantNode from "../../../ConstantNode.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerRangeRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerRange";
    public readonly targetOp = "Range";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Size",
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
        return op.type === "Range";
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

        // --- STATIC EVALUATION PATH (Fixes Slice Canonicalization) ---
        // If the inputs are constants, we calculate the bounds statically.
        // This avoids dynamic shape ValueNodes that cause ORT to crash with strict Loop carry shape mismatches.
        if (
            inputs[0].is(ConstantNode) &&
            inputs[1].is(ConstantNode) &&
            inputs[2].is(ConstantNode)
        ) {
            const start = readScalarFromTensorNode(inputs[0]) ?? 0;
            const limit = readScalarFromTensorNode(inputs[1]) ?? 0;
            const delta = readScalarFromTensorNode(inputs[2]) ?? 1;

            if (delta === 0) {
                throw new Error(
                    `[LowerRangeRecipe] Range operation '${op.id}' has a delta of 0, which is mathematically invalid and causes infinite bounds.`,
                );
            }

            const tripCount = Math.max(0, Math.ceil((limit - start) / delta));

            return { totalIters: tripCount, carryShape: [tripCount] };
        }

        // --- DYNAMIC FALLBACK PATH (For purely dynamic Range ops) ---
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `bounds_${op.id}`);

        // 1. Calculate (limit - start)
        const sub = builder.createOp("Sub", [inputs[1], inputs[0]])[0];

        // 2. Cast to FLOAT so ONNX Ceil doesn't crash
        const subFloat = builder.createOp("Cast", [sub], { to: DataType.FLOAT })[0];
        const deltaFloat = builder.createOp("Cast", [inputs[2]], { to: DataType.FLOAT })[0];

        // 3. Div & Ceil & Cast back to INT64
        const div = builder.createOp("Div", [subFloat, deltaFloat])[0];
        const ceil = builder.createOp("Ceil", [div])[0];
        const cast = builder.createOp("Cast", [ceil], { to: DataType.INT64 })[0];

        // 4. Force strictly to a scalar (safeguards against inputs that are 1D arrays of size 1)
        const scalarShape = builder.createConstant(`scalar_shape_${op.id}`, int64Vec([]));
        const castScalar = builder.createOp("Reshape", [cast, scalarShape])[0];

        // 5. trip_count = max(0, ...)
        const zeroConst = builder.createConstant(`zero_${op.id}`, scalarInt64(0));
        const tripCount = builder.createOp("Max", [castScalar, zeroConst])[0];

        // 6. For Range, the carry is 1D with length = tripCount
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));
        const tripCount1D = builder.createOp("Unsqueeze", [tripCount, axes0])[0];

        // Return the ValueNode `tripCount1D` directly so GraphBuilder
        // triggers the dynamic array allocation path using Expand(zeros, tripCount1D).
        return { totalIters: tripCount, carryShape: tripCount1D, targetShape: tripCount1D };
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
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `range_${op.id}`);
        const inputs = op.getInputs()!;
        const dtype = inputs[0].literalType || DataType.FLOAT;

        let delta = resolveRecipeInput(
            builder,
            inputs[2],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            true,
            targetShapeNode,
            loopCtx,
        );
        const scalarShape = builder.createConstant(`scalar_shape_range_${op.id}`, int64Vec([]));
        delta = builder.createOp("Reshape", [delta, scalarShape], {}, [
            { type: dtype, shape: [] },
        ])[0];

        // --- POINTER / INDUCTION MODE ---
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const outerScalarShape = loopCtx.outerBuilder.createConstant(
                `outer_scalar_shape_${op.id}`,
                int64Vec([]),
            );
            const [initStart] = loopCtx.outerBuilder.createOp(
                "Reshape",
                [inputs[0], outerScalarShape],
                {},
                [{ type: dtype, shape: [] }],
            );
            const currVal = loopCtx.addCarry(`range_val_${op.id}`, initStart, dtype, []);
            const [nextVal] = builder.createOp("Add", [currVal, delta], {}, [
                { type: dtype, shape: [] },
            ]);
            loopCtx.setNextState(currVal, nextVal);
            return currVal;
        }

        // --- RADIX FALLBACK MODE ---
        let start = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            true,
            targetShapeNode,
            loopCtx,
        );
        start = builder.createOp("Reshape", [start, scalarShape])[0];

        const [iterCast] = builder.createOp("Cast", [iter], { to: dtype });
        const [iterStep] = builder.createOp("Mul", [iterCast, delta]);
        const [currentVal] = builder.createOp("Add", [start, iterStep]);
        const [currentValScalar] = builder.createOp("Reshape", [currentVal, scalarShape], {}, [
            { type: dtype, shape: [] },
        ]);

        return currentValScalar;
    }
}
