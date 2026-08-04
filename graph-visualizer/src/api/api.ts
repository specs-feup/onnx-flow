// api.ts (Frontend API Helper)
import type { CytoscapeData } from "../Cytoscape.tsx";

interface StartSessionResponse {
  success: boolean;
  message: string;
  graph: any; // Cytoscape-ready JSON payload returned by flow2json
}

export interface TransformationOpportunity {
  id: string;
  description: string;
  recipeName: string;
  targetNodeId: string;
}

export async function startNewSession(
  graphData: object,
  options: Record<string, any> = {}
): Promise<StartSessionResponse> {
  const response = await fetch("http://localhost:3000/api/session/start", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      graphData,
      options,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error || "Failed to initialize session");
  }

  return response.json();
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

export async function fetchGraph(port: number): Promise<CytoscapeData> {
  const response = await fetch(`http://localhost:${port}/api/graph`, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function fetchTransformationOpportunities(port: number): Promise<TransformationOpportunity[]> {
  const response = await fetch(`http://localhost:${port}/api/opportunities`, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function applyTransformation(port: number, opportunityId: string): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/apply/${opportunityId}`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function undoTransformation(port: number): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/undo`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

export async function redoTransformation(port: number): Promise<any> {
  const response = await fetch(`http://localhost:${port}/api/redo`, { method: "POST" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}