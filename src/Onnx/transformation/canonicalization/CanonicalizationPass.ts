import type OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";

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
import type { DecompositionRecipe } from "../Recipe.js";
import { GraphBuilder } from "../../GraphBuilder.js";
import { topologicalSortOperationNodes } from "../../Utils.js";
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
import { LowerTanhDivRecipe } from "./recipes/LowerTanhDivRecipe.js";
import { LowerAsinhRecipe } from "./recipes/LowerAsinhRecipe.js";
import { LowerAcoshRecipe } from "./recipes/LowerAcoshRecipe.js";
import { LowerAtanhRecipe } from "./recipes/LowerAtanhRecipe.js";
import { LowerCeilRecipe } from "./recipes/LowerCeilRecipe.js";
import { LowerRoundAddRecipe } from "./recipes/LowerRoundAddRecipe.js";
import { LowerRoundStepRecipe } from "./recipes/LowerRoundStepRecipe.js";
import { LowerIsNanRecipe } from "./recipes/LowerIsNanRecipe.js";
import { LowerIsInfRecipe } from "./recipes/LowerIsInfRecipe.js";
import { LowerSqrtRecipe } from "./recipes/LowerSqrtRecipe.js";
import { LowerLogRecipe } from "./recipes/LowerLogRecipe.js";
import { LowerSinRecipe } from "./recipes/LowerSinRecipe.js";
import { LowerCosRecipe } from "./recipes/LowerCosRecipe.js";
import { LowerAsinRecipe } from "./recipes/LowerAsinRecipe.js";
import { LowerAcosRecipe } from "./recipes/LowerAcosRecipe.js";
import { LowerAtanRecipe } from "./recipes/LowerAtanRecipe.js";
//import { LowerReciprocalRecipe } from "./recipes/LowerReciprocalRecipe.js";

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
        ["TanhSigmoid", new LowerTanhSigmoidRecipe()],
        ["TanhDiv", new LowerTanhDivRecipe()],
        ["Asinh", new LowerAsinhRecipe()],
        ["Acosh", new LowerAcoshRecipe()],
        ["Atanh", new LowerAtanhRecipe()],
        ["Ceil", new LowerCeilRecipe()],
        ["RoundAdd", new LowerRoundAddRecipe()],
        ["RoundStep", new LowerRoundStepRecipe()],
        ["IsNaN", new LowerIsNanRecipe()],
        ["IsInf", new LowerIsInfRecipe()],
        ["Sqrt", new LowerSqrtRecipe()],
        ["Log", new LowerLogRecipe()],
        ["Sin", new LowerSinRecipe()],
        ["Cos", new LowerCosRecipe()],
        ["Asin", new LowerAsinRecipe()],
        ["Acos", new LowerAcosRecipe()],
        ["Atan", new LowerAtanRecipe()],
        //["Reciprocal", new LowerReciprocalRecipe()],
    ]);

    public run(graph: OnnxGraph.Class): boolean {
        let globalChanged = false;
        let localChanged = true;

        while (localChanged) {
            localChanged = false;

            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                const recipe = this.registry.get(op.type);

                if (recipe && recipe.canApply(op)) {
                    const builder = new GraphBuilder(graph, `lowering_${op.id}`);
                    recipe.apply(op, builder);

                    localChanged = true;
                    globalChanged = true;
                }
            }
        }

        return globalChanged;
    }
}
