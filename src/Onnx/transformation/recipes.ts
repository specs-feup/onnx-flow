// Loop Lowering Recipes
import { LowerMatMulRecipe } from "./loop-lowering/recipes/LowerMatMulRecipe.js";
import { LowerCoalescedMatMulRecipe } from "./loop-lowering/recipes/LowerCoalescedMatMulRecipe.js";
import { LowerConvRecipe } from "./loop-lowering/recipes/LowerConvRecipe.js";
import { LowerElementWiseRecipe } from "./loop-lowering/recipes/LowerElementWiseRecipe.js";
import { LowerRangeRecipe } from "./loop-lowering/recipes/LowerRangeRecipe.js";
import { LowerReductionRecipe } from "./loop-lowering/recipes/LowerReductionRecipe.js";
import { LowerTransposeRecipe } from "./loop-lowering/recipes/LowerTransposeRecipe.js";
import { LowerConvTransposeRecipe } from "./loop-lowering/recipes/LowerConvTransposeRecipe.js";
import { LowerCumSumRecipe } from "./loop-lowering/recipes/LowerCumSumRecipe.js";
import { LowerGatherElementsRecipe } from "./loop-lowering/recipes/LowerGatherElementsRecipe.js";
import { LowerGatherNDRecipe } from "./loop-lowering/recipes/LowerGatherNDRecipe.js";
import { LowerScatterNDRecipe } from "./loop-lowering/recipes/LowerScatterNDRecipe.js";
import { LowerMaxPoolRecipe } from "./loop-lowering/recipes/LowerMaxPoolRecipe.js";
import { LowerResizeRecipe } from "./loop-lowering/recipes/LowerResizeRecipe.js";
import { LowerReverseSequenceRecipe } from "./loop-lowering/recipes/LowerReverseSequenceRecipe.js";

// Canonicalization Recipes
import { LowerSoftmaxRecipe } from "./canonicalization/recipes/LowerSoftmaxRecipe.js";
import { LowerGemmRecipe } from "./canonicalization/recipes/LowerGemmRecipe.js";
import { LowerSliceRecipe } from "./canonicalization/recipes/LowerSliceRecipe.js";
import { LowerReluRecipe } from "./canonicalization/recipes/LowerReluRecipe.js";
import { LowerSubRecipe } from "./canonicalization/recipes/LowerSubRecipe.js";
import { LowerPadRecipe } from "./canonicalization/recipes/LowerPadRecipe.js";
import { LowerExpandRecipe } from "./canonicalization/recipes/LowerExpandRecipe.js";
import { LowerConcatRecipe } from "./canonicalization/recipes/LowerConcatRecipe.js";
import { LowerClipRecipe } from "./canonicalization/recipes/LowerClipRecipe.js";
import { LowerAveragePoolRecipe } from "./canonicalization/recipes/LowerAveragePoolRecipe.js";
import { LowerQuantizeLinearRecipe } from "./canonicalization/recipes/LowerQuantizeLinearRecipe.js";
import { LowerDequantizeLinearRecipe } from "./canonicalization/recipes/LowerDequantizeLinearRecipe.js";
import { LowerGroupNormalizationRecipe } from "./canonicalization/recipes/LowerGroupNormalizationRecipe.js";
import { LowerAbsRecipe } from "./canonicalization/recipes/LowerAbsRecipe.js";
import { LowerInstanceNormalizationRecipe } from "./canonicalization/recipes/LowerInstanceNormalizationRecipe.js";
import { LowerLayerNormalizationRecipe } from "./canonicalization/recipes/LowerLayerNormalizationRecipe.js";
import { LowerBatchNormalizationRecipe } from "./canonicalization/recipes/LowerBatchNormalizationRecipe.js";
import { LowerSignRecipe } from "./canonicalization/recipes/LowerSignRecipe.js";
import { LowerErfRecipe } from "./canonicalization/recipes/LowerErfRecipe.js";
import { LowerAcoshRecipe } from "./canonicalization/recipes/LowerAcoshRecipe.js";
import { LowerAcosRecipe } from "./canonicalization/recipes/LowerAcosRecipe.js";
import { LowerAsinhRecipe } from "./canonicalization/recipes/LowerAsinhRecipe.js";
import { LowerAsinRecipe } from "./canonicalization/recipes/LowerAsinRecipe.js";
import { LowerAtanhRecipe } from "./canonicalization/recipes/LowerAtanhRecipe.js";
import { LowerAtanRecipe } from "./canonicalization/recipes/LowerAtanRecipe.js";
import { LowerCeilRecipe } from "./canonicalization/recipes/LowerCeilRecipe.js";
import { LowerCeluRecipe } from "./canonicalization/recipes/LowerCeluRecipe.js";
import { LowerCoshRecipe } from "./canonicalization/recipes/LowerCoshRecipe.js";
import { LowerCosRecipe } from "./canonicalization/recipes/LowerCosRecipe.js";
import { LowerDivRecipe } from "./canonicalization/recipes/LowerDivRecipe.js";
import { LowerEluRecipe } from "./canonicalization/recipes/LowerEluRecipe.js";
import { LowerHardSigmoidRecipe } from "./canonicalization/recipes/LowerHardSigmoidRecipe.js";
import { LowerIsInfRecipe } from "./canonicalization/recipes/LowerIsInfRecipe.js";
import { LowerIsNanRecipe } from "./canonicalization/recipes/LowerIsNanRecipe.js";
import { LowerLeakyReluRecipe } from "./canonicalization/recipes/LowerLeakyReluRecipe.js";
import { LowerLogRecipe } from "./canonicalization/recipes/LowerLogRecipe.js";
import { LowerMishRecipe } from "./canonicalization/recipes/LowerMishRecipe.js";
import { LowerModRecipe } from "./canonicalization/recipes/LowerModRecipe.js";
import { LowerNotRecipe } from "./canonicalization/recipes/LowerNotRecipe.js";
import { LowerPowRecipe } from "./canonicalization/recipes/LowerPowRecipe.js";
import { LowerRoundAddRecipe } from "./canonicalization/recipes/LowerRoundAddRecipe.js";
import { LowerRoundStepRecipe } from "./canonicalization/recipes/LowerRoundStepRecipe.js";
import { LowerSeluRecipe } from "./canonicalization/recipes/LowerSeluRecipe.js";
import { LowerSigmoidRecipe } from "./canonicalization/recipes/LowerSigmoidRecipe.js";
import { LowerSinhRecipe } from "./canonicalization/recipes/LowerSinhRecipe.js";
import { LowerSinRecipe } from "./canonicalization/recipes/LowerSinRecipe.js";
import { LowerSoftplusRecipe } from "./canonicalization/recipes/LowerSoftplusRecipe.js";
import { LowerSoftsignRecipe } from "./canonicalization/recipes/LowerSoftsignRecipe.js";
import { LowerSqrtRecipe } from "./canonicalization/recipes/LowerSqrtRecipe.js";
import { LowerTanhDivRecipe } from "./canonicalization/recipes/LowerTanhDivRecipe.js";
import { LowerTanhSigmoidRecipe } from "./canonicalization/recipes/LowerTanhSigmoidRecipe.js";
import { LowerTanRecipe } from "./canonicalization/recipes/LowerTanRecipe.js";
import { LowerOrRecipe } from "./canonicalization/recipes/LowerOrRecipe.js";
import { LowerGreaterOrEqualRecipe } from "./canonicalization/recipes/LowerGreaterOrEqualRecipe.js";
import { LowerLessRecipe } from "./canonicalization/recipes/LowerLessRecipe.js";
import { LowerLessOrEqualRecipe } from "./canonicalization/recipes/LowerLessOrEqualRecipe.js";
import { LowerMinRecipe } from "./canonicalization/recipes/LowerMinRecipe.js";
import { LowerMaxRecipe } from "./canonicalization/recipes/LowerMaxRecipe.js";
import { LowerFloorRecipe } from "./canonicalization/recipes/LowerFloorRecipe.js";
import { LowerNegRecipe } from "./canonicalization/recipes/LowerNegRecipe.js";
import { LowerReciprocalRecipe } from "./canonicalization/recipes/LowerReciprocalRecipe.js";
import { LowerExpRecipe } from "./canonicalization/recipes/LowerExpRecipe.js";
import { LowerCosPCRecipe } from "./canonicalization/recipes/LowerCosPCRecipe.js";
import { LowerSinPCRecipe } from "./canonicalization/recipes/LowerSinPCRecipe.js";
import { LowerLogPCRecipe } from "./canonicalization/recipes/LowerLogPCRecipe.js";
import { LowerSqrtPCRecipe } from "./canonicalization/recipes/LowerSqrtPCRecipe.js";
import { LowerCastLikeRecipe } from "./canonicalization/recipes/LowerCastLikeRecipe.js";
import { LowerMeanRecipe } from "./canonicalization/recipes/LowerMeanRecipe.js";
import { LowerSumRecipe } from "./canonicalization/recipes/LowerSumRecipe.js";
import { LowerBinarizerRecipe } from "./canonicalization/recipes/LowerBinarizerRecipe.js";
import { LowerDynamicQuantizeLinearRecipe } from "./canonicalization/recipes/LowerDynamicQuantizeLinearRecipe.js";
import { LowerGeluRecipe } from "./canonicalization/recipes/LowerGeluRecipe.js";
import { LowerHardSwishRecipe } from "./canonicalization/recipes/LowerHardSwishRecipe.js";
import { LowerPReluRecipe } from "./canonicalization/recipes/LowerPReluRecipe.js";
import { LowerShrinkRecipe } from "./canonicalization/recipes/LowerShrinkRecipe.js";
import { LowerThresholdedReluRecipe } from "./canonicalization/recipes/LowerThresholdedReluRecipe.js";
import { LowerConvIntegerRecipe } from "./canonicalization/recipes/LowerConvIntegerRecipe.js";
import { LowerEyeLikeRecipe } from "./canonicalization/recipes/LowerEyeLikeRecipe.js";
import { LowerFlattenRecipe } from "./canonicalization/recipes/LowerFlattenRecipe.js";
import { LowerGlobalAveragePoolRecipe } from "./canonicalization/recipes/LowerGlobalAveragePoolRecipe.js";
import { LowerGlobalLpPoolRecipe } from "./canonicalization/recipes/LowerGlobalLpPoolRecipe.js";
import { LowerGlobalMaxPoolRecipe } from "./canonicalization/recipes/LowerGlobalMaxPoolRecipe.js";
import { LowerHardmaxRecipe } from "./canonicalization/recipes/LowerHardmaxRecipe.js";
import { LowerLogSoftmaxRecipe } from "./canonicalization/recipes/LowerLogSoftmaxRecipe.js";
import { LowerLpNormalizationRecipe } from "./canonicalization/recipes/LowerLpNormalizationRecipe.js";
import { LowerMatMulIntegerRecipe } from "./canonicalization/recipes/LowerMatMulIntegerRecipe.js";
import { LowerOneHotRecipe } from "./canonicalization/recipes/LowerOneHotRecipe.js";
import { LowerQLinearConvRecipe } from "./canonicalization/recipes/LowerQLinearConvRecipe.js";
import { LowerQLinearMatMulRecipe } from "./canonicalization/recipes/LowerQLinearMatMulRecipe.js";
import { LowerSplitRecipe } from "./canonicalization/recipes/LowerSplitRecipe.js";
import { LowerTileRecipe } from "./canonicalization/recipes/LowerTileRecipe.js";
import { LowerTriluRecipe } from "./canonicalization/recipes/LowerTriluRecipe.js";
import { LowerLpPoolRecipe } from "./canonicalization/recipes/LowerLpPoolRecipe.js";
import { LowerNegativeLogLikelihoodLossRecipe } from "./canonicalization/recipes/LowerNegativeLogLikelihoodLossRecipe.js";
import { LowerSoftmaxCrossEntropyLossRecipe } from "./canonicalization/recipes/LowerSoftmaxCrossEntropyLossRecipe.js";
import { LowerScanRecipe } from "./canonicalization/recipes/LowerScanRecipe.js";
import { LowerRNNRecipe } from "./canonicalization/recipes/LowerRNNRecipe.js";
import { LowerGRURecipe } from "./canonicalization/recipes/LowerGRURecipe.js";
import { LowerLSTMRecipe } from "./canonicalization/recipes/LowerLSTMRecipe.js";
import { LowerScatterRecipe } from "./canonicalization/recipes/LowerScatterRecipe.js";
import {
    LowerArgMaxRecipe,
    LowerArgMinRecipe,
} from "./canonicalization/recipes/LowerArgMinMaxRecipe.js";

// Export classes for manual instantiation
export {
    // Canonicalization Recipes
    LowerSoftmaxRecipe,
    LowerGemmRecipe,
    LowerSliceRecipe,
    LowerReluRecipe,
    LowerSubRecipe,
    LowerPadRecipe,
    LowerExpandRecipe,
    LowerConcatRecipe,
    LowerClipRecipe,
    LowerAveragePoolRecipe,
    LowerQuantizeLinearRecipe,
    LowerDequantizeLinearRecipe,
    LowerGroupNormalizationRecipe,
    LowerBatchNormalizationRecipe,
    LowerLayerNormalizationRecipe,
    LowerInstanceNormalizationRecipe,
    LowerAbsRecipe,
    LowerSignRecipe,
    LowerDivRecipe,
    LowerModRecipe,
    LowerPowRecipe,
    LowerNotRecipe,
    LowerSigmoidRecipe,
    LowerSoftsignRecipe,
    LowerTanRecipe,
    LowerSinhRecipe,
    LowerCoshRecipe,
    LowerTanhSigmoidRecipe,
    LowerTanhDivRecipe,
    LowerAsinhRecipe,
    LowerAcoshRecipe,
    LowerAtanhRecipe,
    LowerCeilRecipe,
    LowerRoundAddRecipe,
    LowerRoundStepRecipe,
    LowerIsNanRecipe,
    LowerIsInfRecipe,
    LowerSqrtRecipe,
    LowerLogRecipe,
    LowerSinRecipe,
    LowerCosRecipe,
    LowerAsinRecipe,
    LowerAcosRecipe,
    LowerAtanRecipe,
    LowerErfRecipe,
    LowerLeakyReluRecipe,
    LowerHardSigmoidRecipe,
    LowerSoftplusRecipe,
    LowerMishRecipe,
    LowerEluRecipe,
    LowerCeluRecipe,
    LowerSeluRecipe,
    LowerOrRecipe,
    LowerGreaterOrEqualRecipe,
    LowerLessRecipe,
    LowerLessOrEqualRecipe,
    LowerMinRecipe,
    LowerMaxRecipe,
    LowerFloorRecipe,
    LowerNegRecipe,
    LowerReciprocalRecipe,
    LowerExpRecipe,
    LowerCosPCRecipe,
    LowerSinPCRecipe,
    LowerLogPCRecipe,
    LowerSqrtPCRecipe,
    LowerCastLikeRecipe,
    LowerMeanRecipe,
    LowerSumRecipe,
    LowerBinarizerRecipe,
    LowerDynamicQuantizeLinearRecipe,
    LowerGeluRecipe,
    LowerHardSwishRecipe,
    LowerPReluRecipe,
    LowerShrinkRecipe,
    LowerThresholdedReluRecipe,
    LowerConvIntegerRecipe,
    LowerEyeLikeRecipe,
    LowerFlattenRecipe,
    LowerGlobalAveragePoolRecipe,
    LowerGlobalLpPoolRecipe,
    LowerGlobalMaxPoolRecipe,
    LowerHardmaxRecipe,
    LowerLogSoftmaxRecipe,
    LowerLpNormalizationRecipe,
    LowerMatMulIntegerRecipe,
    LowerOneHotRecipe,
    LowerQLinearConvRecipe,
    LowerQLinearMatMulRecipe,
    LowerSplitRecipe,
    LowerTileRecipe,
    LowerTriluRecipe,
    LowerLpPoolRecipe,
    LowerNegativeLogLikelihoodLossRecipe,
    LowerSoftmaxCrossEntropyLossRecipe,
    LowerGRURecipe,
    LowerLSTMRecipe,
    LowerRNNRecipe,
    LowerScanRecipe,
    LowerScatterRecipe,
    LowerArgMaxRecipe,
    LowerArgMinRecipe,

    // Loop Lowering Recipes
    LowerMatMulRecipe,
    LowerCoalescedMatMulRecipe,
    LowerConvRecipe,
    LowerElementWiseRecipe,
    LowerRangeRecipe,
    LowerReductionRecipe,
    LowerTransposeRecipe,
    LowerConvTransposeRecipe,
    LowerCumSumRecipe,
    LowerGatherElementsRecipe,
    LowerGatherNDRecipe,
    LowerScatterNDRecipe,
    LowerMaxPoolRecipe,
    LowerResizeRecipe,
    LowerReverseSequenceRecipe,
};

// Export an organized catalog object for easy IDE discovery/autocomplete
export const Recipes = {
    canonicalization: {
        Softmax: LowerSoftmaxRecipe,
        Gemm: LowerGemmRecipe,
        Slice: LowerSliceRecipe,
        Relu: LowerReluRecipe,
        Sub: LowerSubRecipe,
        Pad: LowerPadRecipe,
        Expand: LowerExpandRecipe,
        Concat: LowerConcatRecipe,
        Clip: LowerClipRecipe,
        AveragePool: LowerAveragePoolRecipe,
        QuantizeLinear: LowerQuantizeLinearRecipe,
        DequantizeLinear: LowerDequantizeLinearRecipe,
        GroupNormalization: LowerGroupNormalizationRecipe,
        BatchNormalization: LowerBatchNormalizationRecipe,
        LayerNormalization: LowerLayerNormalizationRecipe,
        InstanceNormalization: LowerInstanceNormalizationRecipe,
        Abs: LowerAbsRecipe,
        Sign: LowerSignRecipe,
        Div: LowerDivRecipe,
        Mod: LowerModRecipe,
        Pow: LowerPowRecipe,
        Not: LowerNotRecipe,
        Sigmoid: LowerSigmoidRecipe,
        Softsign: LowerSoftsignRecipe,
        Tan: LowerTanRecipe,
        Sinh: LowerSinhRecipe,
        Cosh: LowerCoshRecipe,
        TanhSigmoid: LowerTanhSigmoidRecipe,
        TanhDiv: LowerTanhDivRecipe,
        Asinh: LowerAsinhRecipe,
        Acosh: LowerAcoshRecipe,
        Atanh: LowerAtanhRecipe,
        Ceil: LowerCeilRecipe,
        RoundAdd: LowerRoundAddRecipe,
        RoundStep: LowerRoundStepRecipe,
        IsNan: LowerIsNanRecipe,
        IsInf: LowerIsInfRecipe,
        Sqrt: LowerSqrtRecipe,
        Log: LowerLogRecipe,
        Sin: LowerSinRecipe,
        Cos: LowerCosRecipe,
        Asin: LowerAsinRecipe,
        Acos: LowerAcosRecipe,
        Atan: LowerAtanRecipe,
        Erf: LowerErfRecipe,
        LeakyRelu: LowerLeakyReluRecipe,
        HardSigmoid: LowerHardSigmoidRecipe,
        Softplus: LowerSoftplusRecipe,
        Mish: LowerMishRecipe,
        Elu: LowerEluRecipe,
        Celu: LowerCeluRecipe,
        Selu: LowerSeluRecipe,
        Or: LowerOrRecipe,
        GreaterOrEqual: LowerGreaterOrEqualRecipe,
        Less: LowerLessRecipe,
        LessOrEqual: LowerLessOrEqualRecipe,
        Min: LowerMinRecipe,
        Max: LowerMaxRecipe,
        Exp: LowerExpRecipe,
        Floor: LowerFloorRecipe,
        Reciprocal: LowerReciprocalRecipe,
        Neg: LowerNegRecipe,
        SinPC: LowerSinPCRecipe,
        CosPC: LowerCosPCRecipe,
        LogPC: LowerLogPCRecipe,
        SqrtPC: LowerSqrtPCRecipe,
        CastLike: LowerCastLikeRecipe,
        Mean: LowerMeanRecipe,
        Sum: LowerSumRecipe,
        Binarizer: LowerBinarizerRecipe,
        DynamicQuantizeLinear: LowerDynamicQuantizeLinearRecipe,
        Gelu: LowerGeluRecipe,
        HardSwish: LowerHardSwishRecipe,
        PRelu: LowerPReluRecipe,
        Shrink: LowerShrinkRecipe,
        ThresholdedRelu: LowerThresholdedReluRecipe,
        ConvInteger: LowerConvIntegerRecipe,
        EyeLike: LowerEyeLikeRecipe,
        Flatten: LowerFlattenRecipe,
        GlobalAveragePool: LowerGlobalAveragePoolRecipe,
        GlobalLpPool: LowerGlobalLpPoolRecipe,
        GlobalMaxPool: LowerGlobalMaxPoolRecipe,
        Hardmax: LowerHardmaxRecipe,
        LogSoftmax: LowerLogSoftmaxRecipe,
        LpNormalization: LowerLpNormalizationRecipe,
        MatMulInteger: LowerMatMulIntegerRecipe,
        OneHot: LowerOneHotRecipe,
        QLinearConv: LowerQLinearConvRecipe,
        QLinearMatMul: LowerQLinearMatMulRecipe,
        Split: LowerSplitRecipe,
        Tile: LowerTileRecipe,
        Trilu: LowerTriluRecipe,
        LpPool: LowerLpPoolRecipe,
        NegativeLogLikelihoodLoss: LowerNegativeLogLikelihoodLossRecipe,
        SoftmaxCrossEntropyLoss: LowerSoftmaxCrossEntropyLossRecipe,
        GRU: LowerGRURecipe,
        LSTM: LowerLSTMRecipe,
        RNN: LowerRNNRecipe,
        Scan: LowerScanRecipe,
        Scatter: LowerScatterRecipe,
        ArgMax: LowerArgMaxRecipe,
        ArgMin: LowerArgMinRecipe,
    },
    loopLowering: {
        MatMul: LowerMatMulRecipe,
        CoalescedMatMul: LowerCoalescedMatMulRecipe,
        Conv: LowerConvRecipe,
        ElementWise: LowerElementWiseRecipe,
        Range: LowerRangeRecipe,
        Reduction: LowerReductionRecipe,
        Transpose: LowerTransposeRecipe,
        ConvTranspose: LowerConvTransposeRecipe,
        CumSum: LowerCumSumRecipe,
        GatherElements: LowerGatherElementsRecipe,
        GatherND: LowerGatherNDRecipe,
        ScatterND: LowerScatterNDRecipe,
        MaxPool: LowerMaxPoolRecipe,
        Resize: LowerResizeRecipe,
        ReverseSequence: LowerReverseSequenceRecipe,
    },
};
