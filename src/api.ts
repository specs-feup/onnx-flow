import fs from "fs";
import { onnx2json } from "./io/OnnxToJson.js";
import { createGraph } from "./io/OnnxGraphLoader.js";
import { convertFlowGraphToOnnxJson, prepareGraphForExport } from "./io/OnnxGraphExporter.js";
import { json2onnx } from "./io/JsonToOnnx.js";
import { GraphBuilder } from "./onnx/ir/GraphBuilder.js";
import inferShapes, { inferNodeShape, propagateToRegion } from "./onnx/schema/InferShapes.js";
import { topologicalSortOperationNodes, safeWriteJson } from "./onnx/utils/index.js";
import { initializeSchemaRegistry } from "./onnx/schema/index.js";
import { OpRegistry } from "./onnx/schema/OpRegistry.js";
import type { DecompositionRecipe } from "./onnx/transformation/Recipe.js";
import type OnnxGraph from "./onnx/ir/OnnxGraph.js";
import { TrackedGraphBuilder } from "./onnx/transformation/tracking/TrackedGraphBuilder.js";
import { HistoryManager } from "./onnx/transformation/tracking/HistoryManager.js";

// --- 1. Core IR & Types ---
export { default as OnnxGraph } from "./onnx/ir/OnnxGraph.js";
export { default as OperationNode } from "./onnx/ir/OperationNode.js";
export { default as TensorNode } from "./onnx/ir/TensorNode.js";
export { default as ConstantNode } from "./onnx/ir/ConstantNode.js";
export { default as RegionArgumentNode } from "./onnx/ir/RegionArgumentNode.js";
export { default as OnnxEdge } from "./onnx/ir/OnnxEdge.js";
export * from "./onnx/ir/OnnxTypes.js";
export * from "./DecompositionOptions.js";

// --- 2. Frontend, Backend & Formatters ---
export { onnx2json, createGraph, json2onnx, convertFlowGraphToOnnxJson, prepareGraphForExport };
export { default as OnnxDotFormatter } from "./onnx/dot/OnnxDotFormatter.js";

// --- 3. Schema, Shape Inference, Validation & Utilities ---
export { initializeSchemaRegistry, OpRegistry, inferShapes, inferNodeShape, propagateToRegion };
export {
    DEFAULT_COMPAT_IR_VERSION,
    DEFAULT_COMPAT_OPSET,
    getDataTypeMinIrVersion,
    getOpsetMinIrVersion,
    resolveExportVersions,
} from "./onnx/schema/Versioning.js";
export * from "./onnx/schema/OpSchema.js";
export {
    default as validateGraph,
    type ValidationOptions,
} from "./onnx/validation/ValidateGraph.js";
export { default as validateScope } from "./onnx/validation/ValidateScope.js";
export * from "./onnx/utils/index.js";
export * from "./onnx/partitioning/index.js";
export {
    OnnxOpsetAdapter,
    applyAdapters,
    freezeOverridableInputs,
} from "./onnx/frontend/Adapters.js";

// --- 4. Transformation Infrastructure & Tracking ---
export { GraphBuilder, TrackedGraphBuilder, HistoryManager };
export * from "./onnx/transformation/GraphPass.js";
export { PassManager } from "./onnx/transformation/PassManager.js";
export * from "./onnx/transformation/Recipe.js";
export { BaseCanonicalizationRecipe } from "./onnx/transformation/canonicalization/BaseCanonicalizationRecipe.js";
export * from "./onnx/transformation/TransformationOpportunity.js";
export * from "./onnx/transformation/TransformationRegistry.js";
export * from "./onnx/transformation/OrchestratorPass.js";
export * from "./onnx/transformation/tracking/GraphActions.js";

// --- 5. Standard Passes & Recipes Catalog (Including Metadata Helpers) ---
export { default as OnnxGraphTransformer } from "./onnx/transformation/index.js";
export { CanonicalizationPass } from "./onnx/transformation/canonicalization/CanonicalizationPass.js";
export { InferShapesPass } from "./onnx/transformation/InferShapesPass.js";
export { PostLoopLoweringPass } from "./onnx/transformation/loop-lowering/PostLoopLoweringPass.js";
export { LoopLoweringPass } from "./onnx/transformation/loop-lowering/LoopLoweringPass.js";
export {
    LoopTilingPass,
    type LoopTilingOptions,
} from "./onnx/transformation/loop-lowering/LoopTilingPass.js";
export {
    LoopFusionMatcher,
    LoopChainFinder,
    LoopChainLowerer,
    type LoopFusionOptions,
    type LoopBounds,
} from "./onnx/transformation/loop-lowering/LoopFusionMatcher.js";
export * from "./onnx/transformation/loop-lowering/LoopBuilder.js";
export * from "./onnx/transformation/loop-lowering/LoopLoweringRecipe.js";
export * from "./onnx/transformation/loop-lowering/RecipeUtils.js";
export { DeadCodeEliminationPass } from "./onnx/transformation/optimization/DeadCodeEliminationPass.js";
export { ShapeOptimizationPass } from "./onnx/transformation/optimization/ShapeOptimizationPass.js";
export {
    Schedule,
    LOOP_LOWERING_RECIPE_NAMES,
    type ScheduleConfig,
    type SchedulePresetName,
    type StopPhase,
} from "./onnx/transformation/schedule/Schedule.js";
export { CacheReadPass } from "./onnx/transformation/schedule/CacheReadPass.js";
export { LoopUnrollingPass } from "./onnx/transformation/schedule/LoopUnrollingPass.js";
export { ScheduleLoopTransformsPass } from "./onnx/transformation/schedule/ScheduleLoopTransformsPass.js";
export * from "./onnx/transformation/RecipeCatalog.js";

/**
 * Applies a list of decomposition recipes to an in-memory OnnxGraph.
 */
export function applyRecipes(
    graph: OnnxGraph.Class,
    recipesToApply: DecompositionRecipe[],
    historyManager?: HistoryManager,
): number {
    initializeSchemaRegistry();
    let totalCount = 0;

    for (const recipe of recipesToApply) {
        let modified = true;
        while (modified) {
            modified = false;
            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                if (recipe.match(op)) {
                    if (historyManager) {
                        const oppId = `${recipe.name}_${op.id}`;
                        const desc = `Apply ${recipe.name} to ${op.type}`;
                        const tracked = new TrackedGraphBuilder(
                            graph,
                            oppId,
                            desc,
                            `trans_${recipe.name}_${op.id}`,
                        );
                        recipe.apply(op, tracked);
                        historyManager.pushPatch(tracked.commitPatch());
                    } else {
                        const builder = new GraphBuilder(graph, `trans_${recipe.name}_${op.id}`);
                        recipe.apply(op, builder);
                    }
                    totalCount++;
                    modified = true;
                    break;
                }
            }
        }
    }

    if (totalCount > 0) {
        inferShapes(graph);
    }

    return totalCount;
}

/**
 * End-to-end transformation: Ingests an ONNX file, applies chosen recipes,
 * and exports the resulting model.
 */
export async function transformModel(
    inputPath: string,
    outputPath: string,
    recipesToApply: DecompositionRecipe[],
): Promise<number> {
    initializeSchemaRegistry();

    const raw = await onnx2json(inputPath);
    const graph = createGraph(raw);

    const count = applyRecipes(graph, recipesToApply);

    const tempJson = `${outputPath}.tmp.json`;
    try {
        const onnxJson = convertFlowGraphToOnnxJson(graph);
        safeWriteJson(tempJson, onnxJson);
        await json2onnx(tempJson, outputPath);
    } finally {
        if (fs.existsSync(tempJson)) {
            fs.unlinkSync(tempJson);
        }
    }

    return count;
}
