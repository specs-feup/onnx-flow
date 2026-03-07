import type OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { uniq } from "../../../Utils.js";
import type { LoopLoweringRecipe } from "../LoopLoweringRecipe.js";
import { resolveRecipeInput } from "../RecipeUtils.js";
import OnnxEdge from "../../../OnnxEdge.js";
import { squeezeIfLen1 } from "../../loop-lowering/BuildLoop.js";
import { OpRegistry } from "@specs-feup/onnx-flow/Onnx/Schema/OpRegistry";
import { OpCategory } from "@specs-feup/onnx-flow/Onnx/Schema/OpSchema";

export class LowerElementWiseRecipe implements LoopLoweringRecipe {
    canApply(op: OperationNode.Class): boolean {
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.ElementWise) return false;
        console.log("HEEEEYYY");
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
        
        // 1. Get scalar inputs (either from valueMap or by gathering from outer tensors)
        const inputs = op.getInputs()!.map((inp) => 
            resolveRecipeInput(body, inp, valueMap, iter, axes, outShape)
        );

        // Turn [1] -> [] (pure scalar) when allowed
        const effInputs = inputs.map((inp, i) =>
            squeezeIfLen1(body, inp, axes, `${op.id}_in${i}_scalar`)
        );

        // 2. Perform the operation on the scalars
        const node = body
            .addNode(uniq(body, `${op.type}_${op.id}`))
            .init(new OperationNode.Builder(op.type, effInputs))
            .as(OperationNode);

        const out = body
            .addNode(uniq(body, `${op.id}_out`))
            .init(new TensorNode.Builder(inputs[0].literalType, [], "intermediate")) // output is ALWAYS a scalar []
            .as(TensorNode);

        body.addEdge(node, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);

        return out;
    }
}