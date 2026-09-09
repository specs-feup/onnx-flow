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

// Loop Lowering Recipes
import { LowerMatMulRecipe } from "./loop-lowering/recipes/LowerMatMulRecipe.js";
import { LowerCoalescedMatMulRecipe } from "./loop-lowering/recipes/LowerCoalescedMatMulRecipe.js";
import { LowerConvRecipe } from "./loop-lowering/recipes/LowerConvRecipe.js";
import { LowerElementWiseRecipe } from "./loop-lowering/recipes/LowerElementWiseRecipe.js";
import { LowerRangeRecipe } from "./loop-lowering/recipes/LowerRangeRecipe.js";
import { LowerReductionRecipe } from "./loop-lowering/recipes/LowerReductionRecipe.js";
import { LowerTransposeRecipe } from "./loop-lowering/recipes/LowerTransposeRecipe.js";
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

    // Loop Lowering Recipes
    LowerMatMulRecipe,
    LowerCoalescedMatMulRecipe,
    LowerConvRecipe,
    LowerElementWiseRecipe,
    LowerRangeRecipe,
    LowerReductionRecipe,
    LowerTransposeRecipe,
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
    },
    loopLowering: {
        MatMul: LowerMatMulRecipe,
        CoalescedMatMul: LowerCoalescedMatMulRecipe,
        Conv: LowerConvRecipe,
        ElementWise: LowerElementWiseRecipe,
        Range: LowerRangeRecipe,
        Reduction: LowerReductionRecipe,
        Transpose: LowerTransposeRecipe,
    },
};
