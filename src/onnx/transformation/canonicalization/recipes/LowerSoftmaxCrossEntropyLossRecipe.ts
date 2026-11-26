import type OperationNode from "../../../OperationNode.js";
import type { GraphBuilder } from "../../../GraphBuilder.js";
import type { DecompositionRecipe } from "../../Recipe.js";
import type { AttributeMap, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import type { DataType } from "../../../OnnxTypes.js";
import { getAttr, getStringAttr } from "../../../utils/index.js";

export class LowerSoftmaxCrossEntropyLossRecipe implements DecompositionRecipe {
    public readonly name = "LowerSoftmaxCrossEntropyLoss";
    public readonly targetOp = "SoftmaxCrossEntropyLoss";
    public readonly exposesControlFlow = false;
    public readonly exposesDataAccess = false;
    public readonly phase = "canonicalization" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["loss"];
    public readonly producedOps = ["LogSoftmax", "NegativeLogLikelihoodLoss"];

    match(op: OperationNode.Class): boolean {
        return op.type === "SoftmaxCrossEntropyLoss" && (op.getInputs()?.length ?? 0) >= 2;
    }

    apply(op: OperationNode.Class, builder: GraphBuilder): void {
        const ins = op.getInputs() as ConcreteValueNode[];
        const scores = ins[0];
        const labels = ins[1];
        const weights = ins.length >= 3 ? ins[2] : undefined;

        const outputs = op.getOutputs();
        const lossOut = outputs[0];
        const logProbOut = outputs.length >= 2 ? outputs[1] : undefined;

        const dtype = scores.literalType as DataType;
        const [logProb] = builder.createOp("LogSoftmax", [scores], { axis: 1 }, [
            { type: dtype, shape: scores.shape as KnownShape },
        ]);

        const nllAttrs: AttributeMap = {
            reduction: getStringAttr(op, "reduction", "mean"),
        };
        const rawIgnore = getAttr(op, "ignore_index");
        if (typeof rawIgnore === "number") {
            nllAttrs["ignore_index"] = rawIgnore;
        }

        const nllInputs = weights ? [logProb, labels, weights] : [logProb, labels];
        const [loss] = builder.createOp("NegativeLogLikelihoodLoss", nllInputs, nllAttrs, [
            { type: dtype, shape: lossOut.shape as KnownShape },
        ]);

        builder.replaceAllUsesWith(lossOut, loss);
        if (logProbOut) {
            builder.replaceAllUsesWith(logProbOut, logProb);
        }
        builder.removeNode(op);
    }
}
