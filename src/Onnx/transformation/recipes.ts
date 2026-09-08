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

// Export classes for manual instantiation
export {
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