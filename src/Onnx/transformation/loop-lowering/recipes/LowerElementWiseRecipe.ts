import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput, squeezeIfLen1 } from "../RecipeUtils.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { asStaticDims, int64Vec, UNKOWN_SHAPE } from "@specs-feup/onnx-flow/Onnx/Utils";

export class LowerElementWiseRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.ElementWise) return false;

        const inputs = op.getInputs() ?? [];
        if (
            inputs.length > 0 &&
            inputs.every((inp) => inp.shape !== undefined && inp.shape!.length === 0)
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

        // 2. Dynamic Case (compute broadcasted shape via dummy math)
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `ew_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        let currentDummy: ValueNode | undefined = undefined;

        for (let i = 0; i < inputs.length; i++) {
            const inp = inputs[i];

            const [shapeNode] = builder.createOp("Shape", [inp]);
            const expectedCoS = [{ type: DataType.FLOAT, shape: UNKOWN_SHAPE }];
            const [dummy] = builder.createOp("ConstantOfShape", [shapeNode], {}, expectedCoS);

            if (!currentDummy) {
                currentDummy = dummy;
            } else {
                const expectedAdd = [{ type: DataType.FLOAT, shape: UNKOWN_SHAPE }];
                [currentDummy] = builder.createOp("Add", [currentDummy, dummy], {}, expectedAdd);
            }
        }

        if (!currentDummy) return { totalIters: 1, carryShape: [1] };

        const [targetShapeNode] = builder.createOp("Shape", [currentDummy]);
        const [totalIters] = builder.createOp("ReduceProd", [targetShapeNode, axes0], {
            keepdims: 0,
        });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0]);

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
    ): ValueNode {
        const builder = new GraphBuilder(body, `ew_${op.id}`);

        // 1. Resolve scalar inputs (safely handling optional/undefined inputs)
        const inputs = op
            .getInputs()!
            .map((inp) =>
                inp !== undefined
                    ? resolveRecipeInput(
                          builder,
                          inp,
                          valueMap,
                          iter,
                          axes,
                          outShape,
                          true,
                          true,
                          targetShapeNode,
                      )
                    : undefined,
            );

        // Turn [1] -> [] (pure scalar) to match element-wise expectations in the loop body
        const effInputs = inputs.map((inp, i) =>
            inp ? squeezeIfLen1(builder, inp, axes, `${op.id}_in${i}_scalar`) : undefined,
        );

        // 2. Perform the operation on the scalars
        // Pass op.attributes to preserve required attributes (like 'to' for Cast, 'alpha' for LeakyRelu)
        const [out] = builder.createOp(
            op.type,
            effInputs.filter((inp) => inp !== undefined),
            op.attributes,
        );

        return out!;
    }
}
