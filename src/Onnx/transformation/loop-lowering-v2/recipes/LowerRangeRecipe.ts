import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import type { LoopLoweringRecipe, RecipeApplyResult } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput } from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { int64Vec, readScalarFromTensorNode } from "@specs-feup/onnx-flow/Onnx/Utils";
import ConstantNode from "@specs-feup/onnx-flow/Onnx/ConstantNode";

export class LowerRangeRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        return op.type === "Range";
    }

    getLoopBounds(op: OperationNode.Class, outShape: KnownShape) {
        const inputs = op.getInputs()!;
        // If we can't read them statically, build the math
        const builder = new GraphBuilder(op.graph as any, `bounds_${op.id}`);
        
        // trip_count = max(0, ceil((limit - start) / delta))
        const sub = builder.createOp("Sub", [inputs[1], inputs[0]])[0];
        const div = builder.createOp("Div", [sub, inputs[2]])[0];
        const ceil = builder.createOp("Ceil", [div])[0];
        const cast = builder.createOp("Cast", [ceil], { to: DataType.INT64 })[0];
        
        // For Range, the carry is 1D with length = tripCount
        const axes0 = builder.createConstant("axes0", int64Vec([0]));
        const tripCount1D = builder.createOp("Unsqueeze", [cast, axes0])[0];

        return { totalIters: cast, carryShape: tripCount1D };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `range_${op.id}`);
        const inputs = op.getInputs()!;

        // 1. Resolve start and delta as scalars (auto-broadcast/gather)
        const start = resolveRecipeInput(
            builder,
            inputs[0],
            valueMap,
            iter,
            axes,
            outShape,
            true,
            true,
        );
        const delta = resolveRecipeInput(
            builder,
            inputs[2],
            valueMap,
            iter,
            axes,
            outShape,
            true,
            true,
        );

        const dtype = start.literalType || DataType.FLOAT;

        // 2. Cast current iteration index to the target data type
        const [iterCast] = builder.createOp("Cast", [iter], { to: dtype });

        // 3. Compute current value: start + (iter * delta)
        const [iterStep] = builder.createOp("Mul", [iterCast, delta]);
        const [currentVal] = builder.createOp("Add", [start, iterStep]);

        return currentVal;
    }
}
