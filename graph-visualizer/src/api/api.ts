/**
 * @file api.ts
 * @description Frontend REST API client and graph processing helpers for communicating
 * with the ONNX-Flow Explorer backend server (session lifecycle, graph fetch, compilation,
 * transformation discovery, undo/redo, and model export).
 */

import type { CytoscapeData } from "@/types/Cytoscape.ts";

/**
 * Response payload returned when initializing a new graph explorer session.
 */
export interface StartSessionResponse {
  /** Indicates whether session creation succeeded */
  success: boolean;
  /** Unique session identifier (often matching the model file name) */
  sessionId: string;
  /** Server response status message */
  message: string;
  /** Cytoscape-ready graph JSON payload representing the parsed ONNX model */
  graph: any;
}

/**
 * Represents an available graph optimization or lowering opportunity detected by the compiler.
 */
export interface TransformationOpportunity {
  /** Unique identifier for the transformation opportunity */
  id: string;
  /** Human-readable explanation of the transformation recipe */
  description: string;
  /** Name of the optimization/lowering recipe (e.g. LowerGemmRecipe, LoopFusionMatcher) */
  recipeName: string;
  /** Identifier of the target node or comma-separated chain of nodes being transformed */
  targetNodeId: string;
}

/**
 * Represents metadata for an ONNX model file located on the backend server.
 */
export interface ServerFileInfo {
  /** Name of the ONNX file */
  name: string;
  /** File size in bytes */
  size: number;
  /** Timestamp string of last file modification */
  lastModified: string;
}

/**
 * Recursively processes graph data to flatten nested ONNX OperationNode subgraphs (such as Loop
 * bodies or If then/else branches) into Cytoscape compound nodes with parent-child relationships.
 *
 * @param graphData - The raw Cytoscape graph payload containing elements.nodes and elements.edges
 * @returns The transformed graph payload where nested region elements are converted to compound nodes
 */
export function regionsToCompoundNodes(graphData: any): any {
  if (!graphData || !graphData.elements || !Array.isArray(graphData.elements.nodes)) {
    return graphData;
  }

  const allNodes = [...graphData.elements.nodes];
  const allEdges = [...(graphData.elements.edges || [])];
  const existingNodeIds: Set<string> = new Set(
    allNodes.map((n: any) => n.data?.id).filter(Boolean),
  );

  /**
   * Internal recursive helper to traverse nodes and extract child regions into compound nodes.
   *
   * @param nodes - Node collection to scan for OperationNode regions
   */
  function extractRegions(nodes: any[]) {
    for (const node of nodes) {
      const onnxData = node?.data?.onnxData;
      if (
        onnxData?.kind === "OperationNode" &&
        Array.isArray(onnxData.regions) &&
        onnxData.regions.length > 0
      ) {
        const parentId = node.data.id;

        for (let regionIdx = 0; regionIdx < onnxData.regions.length; regionIdx++) {
          const region = onnxData.regions[regionIdx];
          if (region?.elements?.nodes && Array.isArray(region.elements.nodes)) {
            for (const innerNode of region.elements.nodes) {
              if (!innerNode?.data?.id) continue;

              if (!existingNodeIds.has(innerNode.data.id)) {
                existingNodeIds.add(innerNode.data.id);
                const compoundNode = {
                  ...innerNode,
                  data: {
                    ...innerNode.data,
                    parent: parentId,
                    regionIndex: regionIdx,
                  },
                };
                allNodes.push(compoundNode);
              }
            }
          }

          if (region?.elements?.edges && Array.isArray(region.elements.edges)) {
            allEdges.push(...region.elements.edges);
          }

          if (region?.elements?.nodes) {
            extractRegions(region.elements.nodes);
          }
        }
      }
    }
  }

  extractRegions(allNodes);

  return {
    ...graphData,
    elements: {
      nodes: allNodes,
      edges: allEdges,
    },
  };
}

/**
 * Initializes a new visualizer/editor session for an ONNX model file on the backend server.
 *
 * @param port - Port number where the Express API server is listening (e.g. 3000)
 * @param onnxFilename - Name of the target ONNX model file in the server examples directory
 * @returns Promise resolving to the session start response payload
 * @throws Error if the HTTP request fails or the server returns an error
 */
export async function startNewSession(
  port: number,
  onnxFilename: string,
): Promise<StartSessionResponse> {
  const response = await fetch(`http://localhost:${port}/api/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      onnxFilename,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "Failed to initialize session");
  }

  return response.json();
}

/**
 * Terminates an active graph session and frees associated memory on the backend server.
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the session to terminate
 * @returns Promise resolving when session deletion is confirmed
 * @throws Error if the session cannot be deleted
 */
export async function endSession(
  port: number,
  sessionId: string,
): Promise<void> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "Failed to end session");
  }
  console.log(`Session '${sessionId}' deleted.`);
  return;
}

/**
 * Fetches the list of all available ONNX model files from the backend server, sorted alphabetically.
 *
 * @returns Promise resolving to an array of ServerFileInfo file metadata objects
 * @throws Error if fetching file metadata fails
 */
export async function getAvailableFiles(): Promise<ServerFileInfo[]> {
  const response = await fetch("http://localhost:3000/api/files", { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const jsonResponse = await response.json();
  if (!jsonResponse.success) {
    throw new Error(`API error! message: ${jsonResponse.message}`);
  }

  jsonResponse.files.sort((a: ServerFileInfo, b: ServerFileInfo) => 
    a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true })
  );

  return jsonResponse.files;
}

/**
 * Fetches the Cytoscape graph data payload for a specific active session.
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the target session
 * @returns Promise resolving to the Cytoscape graph data object
 * @throws Error if the request fails
 */
export async function fetchGraph(port: number, sessionId: string): Promise<CytoscapeData> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/graph`, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Queries the backend for available graph transformation recipes (canonicalizations, fusions, lowerings)
 * for the current session's graph.
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the target session
 * @returns Promise resolving to an array of transformation opportunities
 * @throws Error if fetching opportunities fails
 */
export async function fetchTransformationOpportunities(port: number, sessionId: string): Promise<TransformationOpportunity[]> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/opportunities`, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Triggers execution of a specific graph transformation recipe on the backend server.
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the target session
 * @param opportunityId - Unique identifier of the transformation opportunity to apply
 * @returns Promise resolving to the server response payload with updated graph data
 * @throws Error if applying the transformation fails
 */
export async function applyTransformation(port: number, sessionId: string, opportunityId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/apply/${opportunityId}`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

/**
 * Reverts the most recently applied transformation step in the session's history timeline (undo).
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the target session
 * @returns Promise resolving to the server response with reverted graph state
 * @throws Error if the undo request fails
 */
export async function undoTransformation(port: number, sessionId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/undo`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

/**
 * Re-applies the most recently undone transformation step in the session's history timeline (redo).
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the target session
 * @returns Promise resolving to the server response with re-applied graph state
 * @throws Error if the redo request fails
 */
export async function redoTransformation(port: number, sessionId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/redo`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

/**
 * Compiles a modified in-memory Cytoscape graph back into an ONNX representation via the backend compiler.
 *
 * @param port - Port number where the Express API server is listening
 * @param sessionId - Unique identifier of the target session
 * @param graphData - The modified Cytoscape graph payload to compile
 * @returns Promise resolving to compilation result containing success status, message, error details, and updated graph
 */
export async function compileOnnxModel(
  port: number,
  sessionId: string,
  graphData: any,
): Promise<{ success: boolean; message?: string; error?: string; graph?: any }> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/compile`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ graph: graphData }),
  });

  const json = await response.json();
  if (!response.ok || !json.success) {
    return {
      success: false,
      error: json.error || json.message || "Failed to compile ONNX Model",
    };
  }

  return json;
}