import type OnnxGraph from "../OnnxGraph.js";
import type { GraphPass } from "../PassManager.js";
import type { DecompositionRecipe } from "./Recipe.js";
import { topologicalSortOperationNodes } from "../Utils.js";
import { GraphBuilder } from "../GraphBuilder.js";
import { OpRegistry } from "../Schema/OpRegistry.js";

export class OrchestratorPass implements GraphPass {
    public readonly name: string;
    private recipes: DecompositionRecipe[];

    constructor(name: string, recipes: DecompositionRecipe[]) {
        this.name = name;
        this.recipes = recipes;
    }

    run(graph: OnnxGraph.Class): boolean {
        let globalChanged = false;
        let localChanged = true;
        let passCount = 0;
        const builder = new GraphBuilder(graph);
        const registry = OpRegistry.getInstance();

        while (localChanged && passCount < 10) {
            localChanged = false;
            passCount++;

            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                const schema = registry.get(op.type, 19);
                const category = schema?.category;

                const recipe = this.recipes.find(
                    (r) =>
                        (r.targetOp === op.type ||
                            (category !== undefined && r.targetOp === category)) &&
                        r.canApply(op),
                );

                if (recipe) {
                    console.log(
                        `[${this.name}] Applying recipe '${recipe.name}' to ${op.id} (${op.type})`,
                    );
                    recipe.apply(op, builder);
                    localChanged = true;
                    globalChanged = true;
                }
            }
        }

        return globalChanged;
    }
}
