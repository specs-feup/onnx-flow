import type OnnxGraph from "../OnnxGraph.js";
import type { GraphPass } from "../PassManager.js";
import type { DecompositionRecipe } from "./Recipe.js";
import { topologicalSortOperationNodes } from "../Utils.js";
import { OpRegistry } from "../Schema/OpRegistry.js";
import { TransformationOpportunity } from "./TransformationOpportunity.js";
import { TrackedGraphBuilder } from "./tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "./tracking/HistoryManager.js";

export class OrchestratorPass implements GraphPass {
    public readonly name: string;
    private recipes: DecompositionRecipe[];

    constructor(name: string, recipes: DecompositionRecipe[]) {
        this.name = name;
        this.recipes = recipes;
    }

    run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean {
        let globalChanged = false;
        let localChanged = true;
        let passCount = 0;
        const registry = OpRegistry.getInstance();

        while (localChanged && passCount < 10) {
            localChanged = false;
            passCount++;
            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                const schema = registry.get(op.type, graph.getTargetOpset());
                const category = schema?.category;

                // Find the first recipe that matches
                const recipe = this.recipes.find(
                    (r) => (r.targetOp === op.type || r.targetOp === category) && r.match(op),
                );

                if (recipe !== undefined) {
                    const opp = new TransformationOpportunity(
                        recipe.name,
                        op.id,
                        `Apply ${recipe.name} to ${op.type}`,
                        (builder) => recipe.apply(op, builder),
                    );

                    console.log(`[${this.name}] Applying opportunity '${opp.id}'`);
                    const trackedBuilder = new TrackedGraphBuilder(
                        graph,
                        opp.id,
                        opp.description,
                        `trans_${recipe.name}_${op.id}`,
                    );
                    opp.apply(trackedBuilder);

                    if (historyManager) {
                        historyManager.pushPatch(trackedBuilder.commitPatch());
                    }

                    localChanged = true;
                    globalChanged = true;
                }
            }
        }

        return globalChanged;
    }
}
