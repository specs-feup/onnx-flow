/**
 * @file ValueNodeExtractor.ts
 * @description Utility functions for filtering and extracting value-carrying nodes from Cytoscape graph data.
 */

import type { CytoscapeData } from "@/types/Cytoscape.ts";

/**
 * Extracts and filters all value-carrying nodes (tensors, constants, region arguments)
 * from a Cytoscape graph payload.
 *
 * In ONNX-Flow computation graphs, operations consume and produce value nodes.
 * This helper filters out operation nodes and returns only nodes capable of acting
 * as inputs or outputs (`TensorNode`, `ConstantNode`, `RegionArgumentNode`).
 *
 * @param cytoscapeData - The Cytoscape graph payload containing elements.nodes
 * @returns Array of Cytoscape node elements corresponding to value nodes, or undefined/empty array if data is missing
 */
export function valueNodeExtractor(cytoscapeData: CytoscapeData | any): any[] {
    return cytoscapeData?.elements?.nodes?.filter((node: any) => {
        const kind = node.data?.onnxData?.kind;
        return kind === "TensorNode" || 
               kind === "ConstantNode" || 
               kind === "RegionArgumentNode";
    }) || [];
}