import fs from "fs";
import type { DecompositionOptions } from "../DecompositionOptions.js";
import { createGraph } from "../io/OnnxGraphLoader.js";
import { onnx2json } from "../io/OnnxToJson.js";
import { convertFlowGraphToOnnxJson } from "../io/OnnxGraphExporter.js";
import type OnnxGraph from "../onnx/OnnxGraph.js";
import type { RawOnnxModel } from "../onnx/OnnxTypes.js";
import type { DecompositionRecipe } from "../onnx/transformation/Recipe.js";
import { createDefaultCanonicalizationRecipes } from "../onnx/transformation/recipes.js";
import { LoopFusionMatcher } from "../onnx/transformation/loop-lowering/LoopFusionMatcher.js";
import type { MutationPatch } from "../onnx/transformation/tracking/GraphActions.js";
import { HistoryManager } from "../onnx/transformation/tracking/HistoryManager.js";
import { TrackedGraphBuilder } from "../onnx/transformation/tracking/TrackedGraphBuilder.js";
import type { TransformationOpportunity } from "../onnx/transformation/TransformationOpportunity.js";
import { TransformationRegistry } from "../onnx/transformation/TransformationRegistry.js";
import type { UnifiedExplorerJson } from "./ExplorerSerializer.js";
import { generateUnifiedExplorerJson } from "./ExplorerSerializer.js";

/**
 * Interactive graph exploration session managing transformation discovery,
 * tracked application, undo/redo history, and JSON export.
 */
export class ExplorerSession {
    public history: HistoryManager;
    public registry: TransformationRegistry;

    /** Loads a `.onnx` or `.json` model from disk into a fresh `ExplorerSession`. */
    public static async fromFile(
        filePath: string,
        options: DecompositionOptions,
    ): Promise<ExplorerSession> {
        console.log(`[ExplorerSession] Loading graph from ${filePath}...`);
        const onnxObject: RawOnnxModel = filePath.endsWith(".json")
            ? (JSON.parse(fs.readFileSync(filePath, "utf8")) as RawOnnxModel)
            : await onnx2json(filePath);

        const graph = createGraph(onnxObject);
        return new ExplorerSession(graph, options);
    }

    constructor(
        public graph: OnnxGraph.Class,
        public options: DecompositionOptions,
        customRecipes?: DecompositionRecipe[],
    ) {
        this.history = new HistoryManager(graph);
        this.registry = new TransformationRegistry(
            customRecipes ?? createDefaultCanonicalizationRecipes(),
        );
    }

    /**
     * Scans the current graph topology and yields all available transformation
     * opportunities (canonicalization, single-op loop lowering, and fused loops).
     */
    public getOpportunities(): TransformationOpportunity[] {
        const opportunities: TransformationOpportunity[] = [];

        opportunities.push(...this.registry.findAllOpportunities(this.graph));

        const singleOpMatcher = new LoopFusionMatcher({
            coalesce: this.options.coalesce,
            fuse: false,
            indexingMode: this.options.indexingMode,
        });
        opportunities.push(...singleOpMatcher.findOpportunities(this.graph));

        if (this.options.fuse) {
            const fusionMatcher = new LoopFusionMatcher({
                coalesce: this.options.coalesce,
                fuse: true,
                indexingMode: this.options.indexingMode,
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
     * Executes an opportunity by ID via `TrackedGraphBuilder` so it can be undone/redone.
     */
    public applyOpportunity(opportunityId: string): boolean {
        const opps = this.getOpportunities();
        const targetOpp = opps.find((o) => o.id === opportunityId);

        if (!targetOpp) {
            console.warn(
                `[ExplorerSession] Opportunity ${opportunityId} not found or no longer valid.`,
            );
            return false;
        }

        const builder = new TrackedGraphBuilder(
            this.graph,
            targetOpp.id,
            targetOpp.description,
            "explorer_interactive",
        );

        targetOpp.apply(builder);
        this.history.pushPatch(builder.commitPatch());
        return true;
    }

    public undo(): MutationPatch | null {
        return this.history.undo();
    }

    public redo(): MutationPatch | null {
        return this.history.redo();
    }

    /** Exports the current graph state as a standard ONNX JSON model. */
    public getOutputOnnxJson(): RawOnnxModel {
        return convertFlowGraphToOnnxJson(this.graph);
    }

    /** Exports the current graph state as a Cytoscape-compatible unified explorer payload. */
    public getOutputUnifiedJson(): UnifiedExplorerJson {
        return generateUnifiedExplorerJson(this.graph);
    }
}
