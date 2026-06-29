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

        //["Exp", new LowerExpRecipe()],
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

                if (recipe) {
                    // Ask the recipe for an opportunity
                    const opportunity = recipe.match(op); 
                    
                    if (opportunity) {
                        const builder = new GraphBuilder(graph, `lowering_${op.id}`);
                        
                        // Execute the opportunity
                        opportunity.apply(builder);

                        localChanged = true;
                        globalChanged = true;
                    }
                }
            }
        }

        return globalChanged;
    }
}
