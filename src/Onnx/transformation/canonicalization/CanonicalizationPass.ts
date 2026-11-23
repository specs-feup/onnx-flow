import type OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";
import type { DecompositionRecipe } from "../Recipe.js";
import { topologicalSortOperationNodes } from "../../Utils.js";

import { LowerSoftmaxRecipe } from "./recipes/LowerSoftmaxRecipe.js";
import { LowerSliceRecipe } from "./recipes/LowerSliceRecipe.js";
import { LowerGemmRecipe } from "./recipes/LowerGemmRecipe.js";
import { LowerQuantizeLinearRecipe } from "./recipes/LowerQuantizeLinearRecipe.js";
import { LowerDequantizeLinearRecipe } from "./recipes/LowerDequantizeLinearRecipe.js";
import { LowerPadRecipe } from "./recipes/LowerPadRecipe.js";
import { LowerExpandRecipe } from "./recipes/LowerExpandRecipe.js";
import { LowerConcatRecipe } from "./recipes/LowerConcatRecipe.js";
import { LowerClipRecipe } from "./recipes/LowerClipRecipe.js";
import { LowerAveragePoolRecipe } from "./recipes/LowerAveragePoolRecipe.js";
import { LowerReluRecipe } from "./recipes/LowerReluRecipe.js";
import { LowerSubRecipe } from "./recipes/LowerSubRecipe.js";
import { LowerGreaterOrEqualRecipe } from "./recipes/LowerGreaterOrEqualRecipe.js";
import { LowerLessRecipe } from "./recipes/LowerLessRecipe.js";
import { LowerLessOrEqualRecipe } from "./recipes/LowerLessOrEqualRecipe.js";
import { LowerMinRecipe } from "./recipes/LowerMinRecipe.js";
import { LowerMaxRecipe } from "./recipes/LowerMaxRecipe.js";
import { LowerAbsRecipe } from "./recipes/LowerAbsRecipe.js";
import { LowerSignRecipe } from "./recipes/LowerSignRecipe.js";
import { LowerLeakyReluRecipe } from "./recipes/LowerLeakyReluRecipe.js";
import { LowerHardSigmoidRecipe } from "./recipes/LowerHardSigmoidRecipe.js";
import { LowerSoftplusRecipe } from "./recipes/LowerSoftplusRecipe.js";
import { LowerMishRecipe } from "./recipes/LowerMishRecipe.js";
import { LowerEluRecipe } from "./recipes/LowerEluRecipe.js";
import { LowerCeluRecipe } from "./recipes/LowerCeluRecipe.js";
import { LowerSeluRecipe } from "./recipes/LowerSeluRecipe.js";
import { LowerOrRecipe } from "./recipes/LowerOrRecipe.js";
import { LowerDivRecipe } from "./recipes/LowerDivRecipe.js";
import { LowerModRecipe } from "./recipes/LowerModRecipe.js";
import { LowerPowRecipe } from "./recipes/LowerPowRecipe.js";
import { LowerNotRecipe } from "./recipes/LowerNotRecipe.js";
import { LowerSigmoidRecipe } from "./recipes/LowerSigmoidRecipe.js";
import { LowerSoftsignRecipe } from "./recipes/LowerSoftsignRecipe.js";
import { LowerTanRecipe } from "./recipes/LowerTanRecipe.js";
import { LowerSinhRecipe } from "./recipes/LowerSinhRecipe.js";
import { LowerCoshRecipe } from "./recipes/LowerCoshRecipe.js";
import { LowerTanhSigmoidRecipe } from "./recipes/LowerTanhSigmoidRecipe.js";
import { LowerAsinhRecipe } from "./recipes/LowerAsinhRecipe.js";
import { LowerAcoshRecipe } from "./recipes/LowerAcoshRecipe.js";
import { LowerAtanhRecipe } from "./recipes/LowerAtanhRecipe.js";
import { LowerCeilRecipe } from "./recipes/LowerCeilRecipe.js";
import { LowerRoundAddRecipe } from "./recipes/LowerRoundAddRecipe.js";
import { LowerIsNanRecipe } from "./recipes/LowerIsNanRecipe.js";
import { LowerIsInfRecipe } from "./recipes/LowerIsInfRecipe.js";
import { LowerAsinRecipe } from "./recipes/LowerAsinRecipe.js";
import { LowerAcosRecipe } from "./recipes/LowerAcosRecipe.js";
import { LowerAtanRecipe } from "./recipes/LowerAtanRecipe.js";
import { LowerErfRecipe } from "./recipes/LowerErfRecipe.js";
import { LowerBatchNormalizationRecipe } from "./recipes/LowerBatchNormalizationRecipe.js";
import { LowerLayerNormalizationRecipe } from "./recipes/LowerLayerNormalizationRecipe.js";
import { LowerInstanceNormalizationRecipe } from "./recipes/LowerInstanceNormalizationRecipe.js";
import { LowerGroupNormalizationRecipe } from "./recipes/LowerGroupNormalizationRecipe.js";
import { LowerFloorRecipe } from "./recipes/LowerFloorRecipe.js";
import { LowerNegRecipe } from "./recipes/LowerNegRecipe.js";
import { LowerCosPCRecipe } from "./recipes/LowerCosPCRecipe.js";
import { LowerLogPCRecipe } from "./recipes/LowerLogPCRecipe.js";
import { LowerSqrtPCRecipe } from "./recipes/LowerSqrtPCRecipe.js";
import { LowerSinPCRecipe } from "./recipes/LowerSinPCRecipe.js";
import { LowerReciprocalRecipe } from "./recipes/LowerReciprocalRecipe.js";
import { LowerExpRecipe } from "./recipes/LowerExpRecipe.js";
import { LowerCastLikeRecipe } from "./recipes/LowerCastLikeRecipe.js";
import { LowerMeanRecipe } from "./recipes/LowerMeanRecipe.js";
import { LowerSumRecipe } from "./recipes/LowerSumRecipe.js";
import { LowerBinarizerRecipe } from "./recipes/LowerBinarizerRecipe.js";
import { LowerDynamicQuantizeLinearRecipe } from "./recipes/LowerDynamicQuantizeLinearRecipe.js";
import { LowerGeluRecipe } from "./recipes/LowerGeluRecipe.js";
import { LowerHardSwishRecipe } from "./recipes/LowerHardSwishRecipe.js";
import { LowerPReluRecipe } from "./recipes/LowerPReluRecipe.js";
import { LowerShrinkRecipe } from "./recipes/LowerShrinkRecipe.js";
import { LowerThresholdedReluRecipe } from "./recipes/LowerThresholdedReluRecipe.js";
import { LowerConvIntegerRecipe } from "./recipes/LowerConvIntegerRecipe.js";
import { LowerEyeLikeRecipe } from "./recipes/LowerEyeLikeRecipe.js";
import { LowerFlattenRecipe } from "./recipes/LowerFlattenRecipe.js";
import { LowerGlobalAveragePoolRecipe } from "./recipes/LowerGlobalAveragePoolRecipe.js";
import { LowerGlobalLpPoolRecipe } from "./recipes/LowerGlobalLpPoolRecipe.js";
import { LowerGlobalMaxPoolRecipe } from "./recipes/LowerGlobalMaxPoolRecipe.js";
import { LowerHardmaxRecipe } from "./recipes/LowerHardmaxRecipe.js";
import { LowerLogSoftmaxRecipe } from "./recipes/LowerLogSoftmaxRecipe.js";
import { LowerLpNormalizationRecipe } from "./recipes/LowerLpNormalizationRecipe.js";
import { LowerMatMulIntegerRecipe } from "./recipes/LowerMatMulIntegerRecipe.js";
import { LowerOneHotRecipe } from "./recipes/LowerOneHotRecipe.js";
import { LowerQLinearConvRecipe } from "./recipes/LowerQLinearConvRecipe.js";
import { LowerQLinearMatMulRecipe } from "./recipes/LowerQLinearMatMulRecipe.js";
import { LowerSplitRecipe } from "./recipes/LowerSplitRecipe.js";
import { LowerTileRecipe } from "./recipes/LowerTileRecipe.js";
import { LowerTriluRecipe } from "./recipes/LowerTriluRecipe.js";
import { LowerLpPoolRecipe } from "./recipes/LowerLpPoolRecipe.js";
import { LowerNegativeLogLikelihoodLossRecipe } from "./recipes/LowerNegativeLogLikelihoodLossRecipe.js";
import { LowerSoftmaxCrossEntropyLossRecipe } from "./recipes/LowerSoftmaxCrossEntropyLossRecipe.js";
import { LowerGRURecipe } from "./recipes/LowerGRURecipe.js";
import { LowerLSTMRecipe } from "./recipes/LowerLSTMRecipe.js";
import { LowerRNNRecipe } from "./recipes/LowerRNNRecipe.js";
import { LowerScanRecipe } from "./recipes/LowerScanRecipe.js";
import { LowerScatterRecipe } from "./recipes/LowerScatterRecipe.js";
import { LowerArgMaxRecipe, LowerArgMinRecipe } from "./recipes/LowerArgMinMaxRecipe.js";

export class CanonicalizationPass implements GraphPass {
    public readonly name = "Canonicalization";

    // Map ONNX OpTypes to their respective canonicalization recipes
    private registry = new Map<string, DecompositionRecipe>([
        ["Softmax", new LowerSoftmaxRecipe()],
        ["Slice", new LowerSliceRecipe()],
        ["Gemm", new LowerGemmRecipe()],
        ["QuantizeLinear", new LowerQuantizeLinearRecipe()],
        ["DequantizeLinear", new LowerDequantizeLinearRecipe()],
        ["Pad", new LowerPadRecipe()],
        ["Expand", new LowerExpandRecipe()],
        ["Concat", new LowerConcatRecipe()],
        ["Clip", new LowerClipRecipe()],
        ["AveragePool", new LowerAveragePoolRecipe()],

        // ElementWise Operations
        ["Relu", new LowerReluRecipe()],
        ["Sub", new LowerSubRecipe()],
        ["GreaterOrEqual", new LowerGreaterOrEqualRecipe()],
        ["Less", new LowerLessRecipe()],
        ["LessOrEqual", new LowerLessOrEqualRecipe()],
        ["Min", new LowerMinRecipe()],
        ["Max", new LowerMaxRecipe()],
        ["Abs", new LowerAbsRecipe()],
        ["Sign", new LowerSignRecipe()],
        ["LeakyRelu", new LowerLeakyReluRecipe()],
        ["HardSigmoid", new LowerHardSigmoidRecipe()],
        ["Softplus", new LowerSoftplusRecipe()],
        ["Mish", new LowerMishRecipe()],
        ["Elu", new LowerEluRecipe()],
        ["Celu", new LowerCeluRecipe()],
        ["Selu", new LowerSeluRecipe()],
        ["Or", new LowerOrRecipe()],
        ["Div", new LowerDivRecipe()],
        ["Mod", new LowerModRecipe()],
        ["Pow", new LowerPowRecipe()],
        ["Not", new LowerNotRecipe()],
        ["Sigmoid", new LowerSigmoidRecipe()],
        ["Softsign", new LowerSoftsignRecipe()],
        ["Tan", new LowerTanRecipe()],
        ["Sinh", new LowerSinhRecipe()],
        ["Cosh", new LowerCoshRecipe()],
        ["Tanh", new LowerTanhSigmoidRecipe()],
        //["Tanh", new LowerTanhDivRecipe()],
        ["Asinh", new LowerAsinhRecipe()],
        ["Acosh", new LowerAcoshRecipe()],
        ["Atanh", new LowerAtanhRecipe()],
        ["Ceil", new LowerCeilRecipe()],
        ["Round", new LowerRoundAddRecipe()],
        //["Round", new LowerRoundStepRecipe()],
        ["IsNaN", new LowerIsNanRecipe()],
        ["IsInf", new LowerIsInfRecipe()],
        //["Sqrt", new LowerSqrtRecipe()],
        //["Log", new LowerLogRecipe()],
        //["Sin", new LowerSinRecipe()],
        //["Cos", new LowerCosRecipe()],
        ["Sqrt", new LowerSqrtPCRecipe()],
        ["Log", new LowerLogPCRecipe()],
        ["Sin", new LowerSinPCRecipe()],
        ["Cos", new LowerCosPCRecipe()],
        ["Asin", new LowerAsinRecipe()],
        ["Acos", new LowerAcosRecipe()],
        ["Atan", new LowerAtanRecipe()],
        ["Erf", new LowerErfRecipe()],
        ["Exp", new LowerExpRecipe()],
        ["Floor", new LowerFloorRecipe()],
        ["Neg", new LowerNegRecipe()],
        ["Reciprocal", new LowerReciprocalRecipe()],
        ["CastLike", new LowerCastLikeRecipe()],
        ["Mean", new LowerMeanRecipe()],
        ["Sum", new LowerSumRecipe()],
        ["Binarizer", new LowerBinarizerRecipe()],
        ["DynamicQuantizeLinear", new LowerDynamicQuantizeLinearRecipe()],
        ["Gelu", new LowerGeluRecipe()],
        ["HardSwish", new LowerHardSwishRecipe()],
        ["PRelu", new LowerPReluRecipe()],
        ["Shrink", new LowerShrinkRecipe()],
        ["ThresholdedRelu", new LowerThresholdedReluRecipe()],
        ["ConvInteger", new LowerConvIntegerRecipe()],
        ["EyeLike", new LowerEyeLikeRecipe()],
        ["Flatten", new LowerFlattenRecipe()],
        ["GlobalAveragePool", new LowerGlobalAveragePoolRecipe()],
        ["GlobalLpPool", new LowerGlobalLpPoolRecipe()],
        ["GlobalMaxPool", new LowerGlobalMaxPoolRecipe()],
        ["Hardmax", new LowerHardmaxRecipe()],
        ["LogSoftmax", new LowerLogSoftmaxRecipe()],
        ["LpNormalization", new LowerLpNormalizationRecipe()],
        ["MatMulInteger", new LowerMatMulIntegerRecipe()],
        ["OneHot", new LowerOneHotRecipe()],
        ["QLinearConv", new LowerQLinearConvRecipe()],
        ["QLinearMatMul", new LowerQLinearMatMulRecipe()],
        ["Split", new LowerSplitRecipe()],
        ["Tile", new LowerTileRecipe()],
        ["Trilu", new LowerTriluRecipe()],
        ["LpPool", new LowerLpPoolRecipe()],
        ["NegativeLogLikelihoodLoss", new LowerNegativeLogLikelihoodLossRecipe()],
        ["SoftmaxCrossEntropyLoss", new LowerSoftmaxCrossEntropyLossRecipe()],
        ["GRU", new LowerGRURecipe()],
        ["LSTM", new LowerLSTMRecipe()],
        ["RNN", new LowerRNNRecipe()],
        ["Scan", new LowerScanRecipe()],
        ["Scatter", new LowerScatterRecipe()],
        ["ArgMax", new LowerArgMaxRecipe()],
        ["ArgMin", new LowerArgMinRecipe()],

        //Normalization Operations
        ["BatchNormalization", new LowerBatchNormalizationRecipe()],
        ["LayerNormalization", new LowerLayerNormalizationRecipe()],
        ["InstanceNormalization", new LowerInstanceNormalizationRecipe()],
        ["GroupNormalization", new LowerGroupNormalizationRecipe()],
    ]);

    public run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        let globalChanged = false;
        let localChanged = true;

        while (localChanged) {
            localChanged = false;

            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                const recipe = this.registry.get(op.type);

                if (recipe) {
                    // Ask the recipe for an opportunity
                    const opportunity = recipe.match(op)
                        ? new TransformationOpportunity(
                              recipe.name,
                              op.id,
                              `Apply ${recipe.name} to ${op.type}`,
                              (builder) => recipe.apply(op, builder),
                          )
                        : null;

                    if (opportunity) {
                        //const builder = new GraphBuilder(graph, `lowering_${op.id}`);
                        const builder = new TrackedGraphBuilder(
                            graph,
                            opportunity.id,
                            opportunity.description,
                            "canonicalization",
                        );

                        // Execute the opportunity
                        opportunity.apply(builder);

                        localChanged = true;
                        globalChanged = true;

                        const patch = builder.commitPatch();
                        historyManager.pushPatch(patch);
                    }
                }
            }
        }

        return globalChanged;
    }
}
