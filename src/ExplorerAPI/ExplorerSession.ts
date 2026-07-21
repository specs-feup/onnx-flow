import type { DecompositionOptions } from "../DecompositionOptions.js";
import type OnnxGraph from "../Onnx/OnnxGraph.js";
import { LowerAveragePoolRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerAveragePoolRecipe.js";
import { LowerClipRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerClipRecipe.js";
import { LowerConcatRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerConcatRecipe.js";
import { LowerDequantizeLinearRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerDequantizeLinearRecipe.js";
import { LowerExpandRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerExpandRecipe.js";
import { LowerGemmRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerGemmRecipe.js";
import { LowerPadRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerPadRecipe.js";
import { LowerQuantizeLinearRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerQuantizeLinearRecipe.js";
import { LowerReluRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerReluRecipe.js";
import { LowerSliceRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerSliceRecipe.js";
import { LowerSoftmaxRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerSoftmaxRecipe.js";
import { LowerSubRecipe } from "../Onnx/transformation/canonicalization/recipes/LowerSubRecipe.js";
import { LoopFusionMatcher } from "../Onnx/transformation/loop-lowering/LoopFusionMatcher.js";
import type { MutationPatch } from "../Onnx/transformation/tracking/GraphActions.js";
import { HistoryManager } from "../Onnx/transformation/tracking/HistoryManager.js";
import { TrackedGraphBuilder } from "../Onnx/transformation/tracking/TrackedGraphBuilder.js";
import type { TransformationOpportunity } from "../Onnx/transformation/TransformationOpportunity.js";
import { TransformationRegistry } from "../Onnx/transformation/TransformationRegistry.js";

export class ExplorerSession {
    public history: HistoryManager;
    public registry: TransformationRegistry;

    constructor(
        public graph: OnnxGraph.Class,
        public options: DecompositionOptions,
    ) {
        this.history = new HistoryManager(graph);

        // 1. Initialize Orchestrator A: The Registry for Canonicalization Recipes
        this.registry = new TransformationRegistry([
            new LowerSoftmaxRecipe(),
            new LowerSliceRecipe(),
            new LowerGemmRecipe(),
            new LowerQuantizeLinearRecipe(),
            new LowerDequantizeLinearRecipe(),
            new LowerPadRecipe(),
            new LowerExpandRecipe(),
            new LowerConcatRecipe(),
            new LowerClipRecipe(),
            new LowerAveragePoolRecipe(),
            new LowerReluRecipe(),
            new LowerSubRecipe(),
        ]);
    }

    /**
     * Scans the current graph topology and yields a comprehensive list of
     * every available transformation (Canonicalization + Loop Lowering).
     */
    public getOpportunities(): TransformationOpportunity[] {
        const opportunities: TransformationOpportunity[] = [];

        // A. Ask the Registry for macro-level canonicalizations
        opportunities.push(...this.registry.findAllOpportunities(this.graph));

        // B. Ask the Matcher for Single-Op loop lowerings
        const singleOpMatcher = new LoopFusionMatcher({
            coalesce: this.options.coalesce,
            fuse: false,
        });
        opportunities.push(...singleOpMatcher.findOpportunities(this.graph));

        // C. Ask the Matcher for Multi-Op (Fused) loop lowerings
        if (this.options.fuse) {
            const fusionMatcher = new LoopFusionMatcher({
                coalesce: this.options.coalesce,
                fuse: true,
            });
            const fusedOpps = fusionMatcher
                .findOpportunities(this.graph)
                // Filter out length-1 chains to avoid duplicating the single-op opportunities
                .filter((opp) => opp.targetNodeId.includes(","));

            opportunities.push(...fusedOpps);
        }

        return opportunities;
    }

    /**
     * Executes an opportunity by ID, wrapping it in the TrackedGraphBuilder
     * so it can be perfectly undone later.
     */
    public applyOpportunity(opportunityId: string): boolean {
        // Regenerate the list to ensure the target is still valid for the current topology
        const opps = this.getOpportunities();
        const targetOpp = opps.find((o) => o.id === opportunityId);

        if (!targetOpp) {
            console.warn(
                `[ExplorerSession] Opportunity ${opportunityId} not found or no longer valid.`,
            );
            return false;
        }

        // Initialize the Tracker for this specific action
        const builder = new TrackedGraphBuilder(
            this.graph,
            targetOpp.id,
            targetOpp.description,
            "explorer_interactive",
        );

        // Execute! The callback will route to either the Registry or the LoopFusionMatcher
        targetOpp.apply(builder);

        // Commit the recorded actions to the timeline
        const patch = builder.commitPatch();
        this.history.pushPatch(patch);

        return true;
    }

    public undo(): MutationPatch | null {
        return this.history.undo();
    }

    public redo(): MutationPatch | null {
        return this.history.redo();
    }
}
