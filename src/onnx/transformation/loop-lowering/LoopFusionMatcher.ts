import type OnnxGraph from "../../ir/OnnxGraph.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";
import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { LoopChainFinder, type LoopFusionOptions } from "./LoopChainFinder.js";
import { LoopChainLowerer } from "./LoopChainLowerer.js";

export type { LoopFusionOptions, LoopBounds } from "./LoopChainFinder.js";
export { LoopChainFinder } from "./LoopChainFinder.js";
export { LoopChainLowerer } from "./LoopChainLowerer.js";

/**
 * Facade that coordinates `LoopChainFinder` (chain discovery & fusion legality)
 * and `LoopChainLowerer` (ONNX `Loop` emission & rewiring) to produce
 * `TransformationOpportunity` instances for `LoopLoweringPass` and the Explorer.
 */
export class LoopFusionMatcher {
    private readonly finder: LoopChainFinder;
    private readonly lowerer: LoopChainLowerer;

    constructor(options: LoopFusionOptions, customRecipes?: LoopLoweringRecipe[]) {
        this.finder = new LoopChainFinder(options, customRecipes);
        this.lowerer = new LoopChainLowerer(this.finder, options);
    }

    /** Discovers all loop-lowering opportunities in `graph`. */
    public findOpportunities(graph: OnnxGraph.Class): TransformationOpportunity[] {
        const chains = this.finder.findChains(graph);
        const opportunities: TransformationOpportunity[] = [];

        for (const chain of chains) {
            const rootOp = chain[chain.length - 1];

            opportunities.push(
                new TransformationOpportunity(
                    "LoopLowering",
                    chain.map((o) => o.id).join(","),
                    `Fuse ${chain.length} ${chain.length === 1 ? "operation" : "operations"} into an ONNX Loop (Root: ${rootOp.type})`,
                    (builder: GraphBuilder) => {
                        this.lowerer.lowerChain(graph, chain, builder);
                    },
                ),
            );
        }

        return opportunities;
    }
}
