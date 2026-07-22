// api.ts (Frontend API Helper)

interface StartSessionResponse {
  success: boolean;
  message: string;
  graph: any; // Cytoscape-ready JSON payload returned by flow2json
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