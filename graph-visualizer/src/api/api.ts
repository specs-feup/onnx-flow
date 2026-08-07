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
  const compoundNodes = [];
  const compoundEdges = [];

  const idSet: Set<string> = new Set();

  for (const node of graphData.elements.nodes) {
    idSet.add(node.data.id);
  } 

  for (const node of graphData.elements.nodes) {
    if (node.data.onnxData.kind === "OperationNode" && ( 
        node.data.onnxData.opType === "Loop" ||
        node.data.onnxData.opType === "If" ||
        node.data.onnxData.opType === "Scan"
      )) {
      if (node.data.onnxData.regions !== 0) {
        for (const innerNode of node.data.onnxData.regions[0].elements.nodes) {
          if (idSet.has(innerNode.data.id)) {
            continue;
          }
          const newNode = {
            ...innerNode,
            data: {
              ...innerNode.data,
              parent: node.data.id  ,
            }
          }
          compoundNodes.push(newNode);
        }
        compoundEdges.push(...node.data.onnxData.regions[0].elements.edges)
      }
    }
  }

  graphData.elements.nodes.push(...compoundNodes);
  graphData.elements.edges.push(...compoundEdges);

  console.log(graphData)

  return graphData;
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