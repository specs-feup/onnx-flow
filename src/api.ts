import fs from "fs";
import { onnx2json } from "./onnx2json.js";
import { createGraph } from "./initGraph.js";
import { convertFlowGraphToOnnxJson, prepareGraphForExport } from "./flow2json.js";
import { json2onnx } from "./json2onnx.js";
import { GraphBuilder } from "./Onnx/GraphBuilder.js";
import inferShapes, { inferNodeShape, propagateToRegion } from "./Onnx/InferShapes.js";
import { topologicalSortOperationNodes, safeWriteJson } from "./Onnx/Utils.js";
import { initializeSchemaRegistry } from "./Onnx/Schema/index.js";
import { OpRegistry } from "./Onnx/Schema/OpRegistry.js";
import type { DecompositionRecipe } from "./Onnx/transformation/Recipe.js";
import type OnnxGraph from "./Onnx/OnnxGraph.js";
import { TrackedGraphBuilder } from "./Onnx/transformation/tracking/TrackedGraphBuilder.js";
import { HistoryManager } from "./Onnx/transformation/tracking/HistoryManager.js";

// --- 1. Core IR & Types ---
export { default as OnnxGraph } from "./Onnx/OnnxGraph.js";
export { default as OperationNode } from "./Onnx/OperationNode.js";
export { default as TensorNode } from "./Onnx/TensorNode.js";
export { default as ConstantNode } from "./Onnx/ConstantNode.js";
export { default as RegionArgumentNode } from "./Onnx/RegionArgumentNode.js";
export { default as OnnxEdge } from "./Onnx/OnnxEdge.js";
export { default as OnnxInnerEdge } from "./Onnx/OnnxInnerEdge.js";
export * from "./Onnx/OnnxTypes.js";
export * from "./DecompositionOptions.js";

// --- 2. Frontend, Backend & Formatters ---
export { onnx2json, createGraph, json2onnx, convertFlowGraphToOnnxJson, prepareGraphForExport };
export { default as OnnxDotFormatter } from "./Onnx/dot/OnnxDotFormatter.js";

// --- 3. Schema, Shape Inference, Validation & Utilities ---
export { initializeSchemaRegistry, OpRegistry, inferShapes, inferNodeShape, propagateToRegion };
export {
    DEFAULT_COMPAT_IR_VERSION,
    DEFAULT_COMPAT_OPSET,
    getDataTypeMinIrVersion,
    getOpsetMinIrVersion,
    resolveExportVersions,
} from "./Onnx/Schema/Versioning.js";
export * from "./Onnx/Schema/OpSchema.js";
export {
    default as validateGraph,
    type ValidationOptions,
} from "./Onnx/validation/ValidateGraph.js";
export { default as validateScope } from "./Onnx/validation/ValidateScope.js";
export * from "./Onnx/Utils.js";
export { partitionGraph } from "./Onnx/partitioning/Partition.js";
export { splitByAncestor } from "./Onnx/partitioning/Strategies.js";

// --- 4. Transformation Infrastructure & Tracking ---
export { GraphBuilder, TrackedGraphBuilder, HistoryManager };
export * from "./Onnx/PassManager.js";
export * from "./Onnx/transformation/Recipe.js";
export * from "./Onnx/transformation/TransformationOpportunity.js";
export * from "./Onnx/transformation/TransformationRegistry.js";
export * from "./Onnx/transformation/OrchestratorPass.js";
export * from "./Onnx/transformation/tracking/GraphActions.js";

// --- 5. Standard Passes & Recipes Catalog (Including Metadata Helpers) ---
export { default as OnnxGraphTransformer } from "./Onnx/transformation/index.js";
export { CanonicalizationPass } from "./Onnx/transformation/canonicalization/CanonicalizationPass.js";
export { InferShapesPass } from "./Onnx/transformation/InferShapesPass.js";
export { LoopLoweringPass } from "./Onnx/transformation/loop-lowering/LoopLoweringPass.js";
export {
    LoopTilingPass,
    type LoopTilingOptions,
} from "./Onnx/transformation/loop-lowering/LoopTilingPass.js";
export { LoopFusionMatcher } from "./Onnx/transformation/loop-lowering/LoopFusionMatcher.js";
export * from "./Onnx/transformation/loop-lowering/LoopBuilder.js";
export * from "./Onnx/transformation/loop-lowering/LoopLoweringRecipe.js";
export * from "./Onnx/transformation/loop-lowering/RecipeUtils.js";
export { DeadCodeEliminationPass } from "./Onnx/transformation/optimization/DeadCodeEliminationPass.js";
export { ShapeOptimizationPass } from "./Onnx/transformation/optimization/ShapeOptimizationPass.js";
export {
    Schedule,
    LOOP_LOWERING_RECIPE_NAMES,
    type ScheduleConfig,
    type SchedulePresetName,
    type StopPhase,
} from "./Onnx/transformation/Schedule.js";
export * from "./Onnx/transformation/recipes.js";

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
