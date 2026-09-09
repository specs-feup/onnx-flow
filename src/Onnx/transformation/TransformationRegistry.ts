import type OnnxGraph from "../OnnxGraph.js";
import type { DecompositionRecipe } from "./Recipe.js";
import { topologicalSortOperationNodes } from "../Utils.js";
import { OpRegistry } from "../Schema/OpRegistry.js";
import { TransformationOpportunity } from "./TransformationOpportunity.js";

export class TransformationRegistry {
    constructor(private recipes: DecompositionRecipe[]) {}

    /**
     * Scans the entire graph and returns EVERYTHING that could possibly be done.
     */
    public findAllOpportunities(graph: OnnxGraph.Class): TransformationOpportunity[] {
        const opportunities: TransformationOpportunity[] = [];
        const ops = topologicalSortOperationNodes(graph);
        const schemaRegistry = OpRegistry.getInstance();

        for (const op of ops) {
            if (!graph.hasNode(op.id)) continue;

            const schema = schemaRegistry.get(op.type, 19);
            const category = schema?.category;

            for (const recipe of this.recipes) {
                if (recipe.targetOp === op.type || recipe.targetOp === category) {
                    if (recipe.match(op)) {
                        // The Registry creates the executable opportunity!
                        opportunities.push(
                            new TransformationOpportunity(
                                recipe.name,
                                op.id,
                                `Apply ${recipe.name} to ${op.type}`,
                                (builder) => recipe.apply(op, builder),
                            ),
                        );
                    }
                }
            }
        }
        return opportunities;
    }
}
