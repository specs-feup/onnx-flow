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
import { LowerDetRecipe } from "./recipes/LowerDetRecipe.js";
import { LowerEinsumRecipe } from "./recipes/LowerEinsumRecipe.js";
import { LowerTopKRecipe } from "./recipes/LowerTopKRecipe.js";
import { LowerRNNLoopRecipe } from "./recipes/LowerRNNLoopRecipe.js";
import { LowerGRULoopRecipe } from "./recipes/LowerGRULoopRecipe.js";
import { LowerLSTMLoopRecipe } from "./recipes/LowerLSTMLoopRecipe.js";
import { LowerSplitDynamicRecipe } from "./recipes/LowerSplitDynamicRecipe.js";
import {
    LowerVariadicMaxRecipe,
    LowerVariadicMinRecipe,
} from "./recipes/LowerVariadicMinMaxRecipe.js";
import {
    LowerArgMaxLoopRecipe,
    LowerArgMinLoopRecipe,
} from "./recipes/LowerArgMinMaxLoopRecipe.js";
import { LowerTopKLoopRecipe } from "./recipes/LowerTopKLoopRecipe.js";
import { LowerMaxPoolWithIndicesRecipe } from "./recipes/LowerMaxPoolWithIndicesRecipe.js";
import { LowerLpNormalizationGeneralRecipe } from "./recipes/LowerLpNormalizationGeneralRecipe.js";
import { LowerQuantizeLinearExtendedRecipe } from "./recipes/LowerQuantizeLinearExtendedRecipe.js";
import { LowerDequantizeLinearBlockedRecipe } from "./recipes/LowerDequantizeLinearBlockedRecipe.js";
import { LowerHardmaxDynamicRecipe } from "./recipes/LowerHardmaxDynamicRecipe.js";
import { LowerTriluDynamicRecipe } from "./recipes/LowerTriluDynamicRecipe.js";
import { LowerEyeLikeDynamicRecipe } from "./recipes/LowerEyeLikeDynamicRecipe.js";
import { LowerDetDynamicRecipe } from "./recipes/LowerDetDynamicRecipe.js";

export class CanonicalizationPass implements GraphPass {
    public readonly name = "Canonicalization";

    // Map ONNX OpTypes to ordered lists of candidate canonicalization recipes (first match wins)
    private registry = new Map<string, DecompositionRecipe[]>();

    constructor(customRecipes?: DecompositionRecipe[]) {
        const defaultRecipes: DecompositionRecipe[] = [
            new LowerSoftmaxRecipe(),
            new LowerSliceRecipe(),
            new LowerGemmRecipe(),
            new LowerQuantizeLinearRecipe(),
            new LowerDequantizeLinearRecipe(),
            new LowerPadRecipe(),
            new LowerExpandRecipe(),
            new LowerConcatRecipe(),
            new LowerClipRecipe(),
            new LowerAveragePoolRecipe(),
            new LowerRNNLoopRecipe(),
            new LowerGRULoopRecipe(),
            new LowerLSTMLoopRecipe(),
            new LowerSplitDynamicRecipe(),
            new LowerVariadicMinRecipe(),
            new LowerVariadicMaxRecipe(),
            new LowerArgMaxLoopRecipe(),
            new LowerArgMinLoopRecipe(),
            new LowerTopKLoopRecipe(),
            new LowerMaxPoolWithIndicesRecipe(),
            new LowerLpNormalizationGeneralRecipe(),
            new LowerQuantizeLinearExtendedRecipe(),
            new LowerDequantizeLinearBlockedRecipe(),
            new LowerHardmaxDynamicRecipe(),
            new LowerTriluDynamicRecipe(),
            new LowerEyeLikeDynamicRecipe(),
            new LowerDetDynamicRecipe(),

            // ElementWise Operations
            new LowerReluRecipe(),
            new LowerSubRecipe(),
            new LowerGreaterOrEqualRecipe(),
            new LowerLessRecipe(),
            new LowerLessOrEqualRecipe(),
            new LowerMinRecipe(),
            new LowerMaxRecipe(),
            new LowerAbsRecipe(),
            new LowerSignRecipe(),
            new LowerLeakyReluRecipe(),
            new LowerHardSigmoidRecipe(),
            new LowerSoftplusRecipe(),
            new LowerMishRecipe(),
            new LowerEluRecipe(),
            new LowerCeluRecipe(),
            new LowerSeluRecipe(),
            new LowerOrRecipe(),
            new LowerDivRecipe(),
            new LowerModRecipe(),
            new LowerPowRecipe(),
            new LowerNotRecipe(),
            new LowerSigmoidRecipe(),
            new LowerSoftsignRecipe(),
            new LowerTanRecipe(),
            new LowerSinhRecipe(),
            new LowerCoshRecipe(),
            new LowerTanhSigmoidRecipe(),
            new LowerAsinhRecipe(),
            new LowerAcoshRecipe(),
            new LowerAtanhRecipe(),
            new LowerCeilRecipe(),
            new LowerRoundAddRecipe(),
            new LowerIsNanRecipe(),
            new LowerIsInfRecipe(),
            new LowerSqrtPCRecipe(),
            new LowerLogPCRecipe(),
            new LowerSinPCRecipe(),
            new LowerCosPCRecipe(),
            new LowerAsinRecipe(),
            new LowerAcosRecipe(),
            new LowerAtanRecipe(),
            new LowerErfRecipe(),
            new LowerExpRecipe(),
            new LowerFloorRecipe(),
            new LowerNegRecipe(),
            new LowerReciprocalRecipe(),
            new LowerCastLikeRecipe(),
            new LowerMeanRecipe(),
            new LowerSumRecipe(),
            new LowerBinarizerRecipe(),
            new LowerDynamicQuantizeLinearRecipe(),
            new LowerGeluRecipe(),
            new LowerHardSwishRecipe(),
            new LowerPReluRecipe(),
            new LowerShrinkRecipe(),
            new LowerThresholdedReluRecipe(),
            new LowerConvIntegerRecipe(),
            new LowerEyeLikeRecipe(),
            new LowerFlattenRecipe(),
            new LowerGlobalAveragePoolRecipe(),
            new LowerGlobalLpPoolRecipe(),
            new LowerGlobalMaxPoolRecipe(),
            new LowerHardmaxRecipe(),
            new LowerLogSoftmaxRecipe(),
            new LowerLpNormalizationRecipe(),
            new LowerMatMulIntegerRecipe(),
            new LowerOneHotRecipe(),
            new LowerQLinearConvRecipe(),
            new LowerQLinearMatMulRecipe(),
            new LowerSplitRecipe(),
            new LowerTileRecipe(),
            new LowerTriluRecipe(),
            new LowerLpPoolRecipe(),
            new LowerNegativeLogLikelihoodLossRecipe(),
            new LowerSoftmaxCrossEntropyLossRecipe(),
            new LowerGRURecipe(),
            new LowerLSTMRecipe(),
            new LowerRNNRecipe(),
            new LowerScanRecipe(),
            new LowerScatterRecipe(),
            new LowerArgMaxRecipe(),
            new LowerArgMinRecipe(),
            new LowerTopKRecipe(),
            new LowerEinsumRecipe(),
            new LowerDetRecipe(),

            // Normalization Operations
            new LowerBatchNormalizationRecipe(),
            new LowerLayerNormalizationRecipe(),
            new LowerInstanceNormalizationRecipe(),
            new LowerGroupNormalizationRecipe(),
        ];

        for (const recipe of customRecipes ?? defaultRecipes) {
            this.register(recipe);
        }
    }

    public register(recipe: DecompositionRecipe): void {
        const key = String(recipe.targetOp);
        const existing = this.registry.get(key);
        if (existing !== undefined) {
            existing.push(recipe);
            existing.sort((a, b) => b.priority - a.priority);
        } else {
            this.registry.set(key, [recipe]);
        }
    }

    public getRecipesFor(opType: string): readonly DecompositionRecipe[] {
        return this.registry.get(opType) ?? [];
    }

    public run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        let globalChanged = false;
        let localChanged = true;

        while (localChanged) {
            localChanged = false;

            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                const candidates = this.registry.get(op.type);
                if (candidates === undefined) continue;

                const recipe = candidates.find((candidate) => candidate.match(op));
                if (recipe === undefined) continue;

                const opportunity = new TransformationOpportunity(
                    recipe.name,
                    op.id,
                    `Apply ${recipe.name} to ${op.type}`,
                    (builder) => recipe.apply(op, builder),
                );

                const builder = new TrackedGraphBuilder(
                    graph,
                    opportunity.id,
                    opportunity.description,
                    "canonicalization",
                );

                opportunity.apply(builder);

                localChanged = true;
                globalChanged = true;

                const patch = builder.commitPatch();
                historyManager.pushPatch(patch);
            }
        }

        return globalChanged;
    }
}
