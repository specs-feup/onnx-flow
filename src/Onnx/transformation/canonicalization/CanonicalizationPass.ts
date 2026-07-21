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
import { topologicalSortOperationNodes } from "../../Utils.js";
import { LowerReluRecipe } from "./recipes/LowerReluRecipe.js";
import { LowerSubRecipe } from "./recipes/LowerSubRecipe.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";

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
