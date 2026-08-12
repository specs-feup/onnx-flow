// api.ts (Frontend API Helper)
import type { CytoscapeData } from "@/types/Cytoscape.ts";

interface StartSessionResponse {
  success: boolean;
  sessionId: string;
  message: string;
  graph: any; // Cytoscape-ready JSON payload returned by flow2json
}

export interface TransformationOpportunity {
  id: string;
  description: string;
  recipeName: string;
  targetNodeId: string;
}

export function regionsToCompoundNodes(graphData: any): any {
  if (!graphData || !graphData.elements || !Array.isArray(graphData.elements.nodes)) {
    return graphData;
  }

  const allNodes = [...graphData.elements.nodes];
  const allEdges = [...(graphData.elements.edges || [])];
  const existingNodeIds: Set<string> = new Set(
    allNodes.map((n: any) => n.data?.id).filter(Boolean),
  );

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

export async function endSession(
  port: number,
  sessionId: string,
): Promise<void> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}`, {
    method: "DELETE",
  })

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "Failed to end session");
  }
  console.log(`Session '${sessionId}' deleted.`);
  return;
}

export async function getAvailableFiles(): Promise<any> {
  const response = await fetch("http://localhost:3000/api/files", { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const jsonResponse = await response.json();
  if (!jsonResponse.success) {
    throw new Error(`API error! message: ${jsonResponse.message}`);
  }

  jsonResponse.files.sort((a: any, b: any) => a.name.localeCompare(b.name, undefined, { sensitivity: "base", numeric: true }));

  return jsonResponse.files;
}

export async function fetchGraph(port: number, sessionId: string): Promise<CytoscapeData> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/graph`, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function fetchTransformationOpportunities(port: number, sessionId: string): Promise<TransformationOpportunity[]> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/opportunities`, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function applyTransformation(port: number, sessionId: string, opportunityId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/apply/${opportunityId}`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function undoTransformation(port: number, sessionId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/undo`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function redoTransformation(port: number, sessionId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/sessions/${sessionId}/redo`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

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