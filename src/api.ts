import fs from "fs";
import { onnx2json } from "./onnx2json.js";
import { createGraph } from "./initGraph.js";
import { convertFlowGraphToOnnxJson } from "./flow2json.js";
import { json2onnx } from "./json2onnx.js";
import { GraphBuilder } from "./Onnx/GraphBuilder.js";
import inferShapes from "./Onnx/InferShapes.js";
import { topologicalSortOperationNodes, safeWriteJson } from "./Onnx/Utils.js";
import { initializeSchemaRegistry } from "./Onnx/Schema/index.js";
import type { DecompositionRecipe } from "./Onnx/transformation/Recipe.js";
import type OnnxGraph from "./Onnx/OnnxGraph.js";

export * from "./Onnx/transformation/recipes.js";
export { onnx2json, createGraph, json2onnx, convertFlowGraphToOnnxJson };

/**
 * Applies a list of decomposition recipes to an in-memory OnnxGraph.
 */
export function applyRecipes(
    graph: OnnxGraph.Class,
    recipesToApply: DecompositionRecipe[],
): number {
    let totalCount = 0;
    for (const recipe of recipesToApply) {
        let modified = true;
        while (modified) {
            modified = false;
            const ops = topologicalSortOperationNodes(graph);
            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;
                if (recipe.match(op)) {
                    const builder = new GraphBuilder(graph, `trans_${recipe.name}_${op.id}`);
                    recipe.apply(op, builder);
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
