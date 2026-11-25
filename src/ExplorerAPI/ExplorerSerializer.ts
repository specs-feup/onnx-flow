import type OnnxGraph from "../Onnx/OnnxGraph.js";
import TensorNode from "../Onnx/TensorNode.js";
import type { NodeSnapshot, EdgeSnapshot } from "../Onnx/transformation/tracking/GraphActions.js";
import { isOnnxNode, asOnnxNode } from "../Onnx/Utils.js";

// TYPES
export type UnifiedNodeData =
    | Exclude<NodeSnapshot, { kind: "OperationNode" }> // Keeps Tensor, Constant, RegionArgument as-is
    | {
          kind: "OperationNode";
          id: string;
          opType: string;
          attributes: Record<string, unknown>;
          inputs: string[];
          regions: UnifiedExplorerJson[];
          metadata?: Record<string, unknown>;
      };

export interface UnifiedNode {
    data: {
        id: string;
        onnxData?: UnifiedNodeData;
        [key: string]: unknown; // Allows Cytoscape's visual properties (label, color, etc.)
    };
    position?: { x: number; y: number };
    classes?: string;
}

export interface UnifiedEdge {
    data: {
        id: string;
        source: string;
        target: string;

        // Explicit boolean so the frontend never has to check for undefined
        isCrossGraph: boolean;

        // The ID of the parent/boundary node (e.g., the Loop node itself)
        // Optional, because it only exists if isCrossGraph is true
        outsideTarget?: string;

        onnxData?: EdgeSnapshot;
        [key: string]: unknown;
    };
    classes?: string;
}

export interface UnifiedExplorerJson {
    elements: {
        nodes: UnifiedNode[];
        edges: UnifiedEdge[];
    };
    [key: string]: unknown; // Allows other Cytoscape top-level properties like zooming
}

interface RawCytoscapeExport {
    elements?: {
        nodes?: UnifiedNode[];
        edges?: UnifiedEdge[];
    };
    [key: string]: unknown;
}

// FUNCTIONS
/**
 * Generates a unified JSON payload that Cytoscape can render visually,
 * but contains the exact ONNX-Flow data needed to reconstruct the graph.
 */
export function generateUnifiedExplorerJson(
    graph: OnnxGraph.Class,
    includeCrossGraphEdges: boolean = true,
    parentScopeNodeIds: Set<string> = new Set(),
    parentOperationId?: string,
): UnifiedExplorerJson {
    // 1. Get the raw Cytoscape JSON
    const rawCy = graph.toCy().json() as unknown as RawCytoscapeExport;

    if (rawCy.elements === undefined) {
        rawCy.elements = { nodes: [], edges: [] };
    } else {
        if (rawCy.elements.nodes === undefined) {
            rawCy.elements.nodes = [];
        }
        if (rawCy.elements.edges === undefined) {
            rawCy.elements.edges = [];
        }
    }

    const cyJson = rawCy as UnifiedExplorerJson;

    // 2. Accumulate all node IDs from the current graph to pass to its children
    const currentScopeNodeIds = new Set(parentScopeNodeIds);
    graph.getNodes().forEach((n) => currentScopeNodeIds.add(n.id));

    // 3. Enrich Nodes
    if (cyJson.elements.nodes.length > 0) {
        cyJson.elements.nodes.forEach((cyNode: UnifiedNode) => {
            const actualNode = graph.getNodeById(cyNode.data.id);
            if (actualNode && isOnnxNode(actualNode)) {
                const snap = asOnnxNode(actualNode).toSnapshot();
                // If it's an OperationNode, we must recursively serialize its subgraphs!
                if (snap.kind === "OperationNode") {
                    cyNode.data.onnxData = {
                        ...snap,
                        // Recursively convert OnnxGraph.Class[] -> UnifiedExplorerJson[]
                        regions: snap.regions.map((regionGraph) =>
                            generateUnifiedExplorerJson(
                                regionGraph,
                                includeCrossGraphEdges,
                                currentScopeNodeIds,
                                cyNode.data.id,
                            ),
                        ),
                    };
                } else {
                    cyNode.data.onnxData = snap;
                }
            }

            for (const key of Object.keys(cyNode.data)) {
                if (key.startsWith("__specs-onnx__")) {
                    delete cyNode.data[key];
                }
            }
        });
    }

    // 4. Enrich Edges
    if (cyJson.elements.edges.length > 0) {
        cyJson.elements.edges.forEach((cyEdge: UnifiedEdge) => {
            cyEdge.data.isCrossGraph = false;
            const actualEdge = graph.getOnnxEdgeById(cyEdge.data.id);
            if (actualEdge != undefined) {
                cyEdge.data.onnxData = actualEdge.toSnapshot();
            }

            for (const key of Object.keys(cyEdge.data)) {
                if (key.startsWith("__specs-onnx__")) {
                    delete cyEdge.data[key];
                }
            }
        });
    }

    // 5. Inject Cross-Graph Edges (Visualization Only)
    if (includeCrossGraphEdges) {
        graph.getOperationNodes().forEach((opNode) => {
            const inputs = opNode.getInputs() ?? [];

            inputs.forEach((input) => {
                // Determine if this input is a formal graph boundary input
                // (e.g., iter, cond_in, carry)
                const isFormalLocalInput =
                    input.is(TensorNode) && input.as(TensorNode).type === "input";

                // If it exists in the parent graph, isn't generated locally, and isn't a formal input,
                // it is an implicit proxy capture.
                if (
                    parentScopeNodeIds.has(input.id) &&
                    input.incomers.length === 0 &&
                    !isFormalLocalInput
                ) {
                    const syntheticEdge: UnifiedEdge = {
                        data: {
                            id: `cross_edge_${input.id}_to_${opNode.id}`,
                            source: input.id, // The outer tensor ID
                            target: opNode.id, // The inner operation
                            isCrossGraph: true,
                            outsideTarget: parentOperationId!,
                        },
                        classes: "cross-graph-capture",
                    };

                    cyJson.elements.edges.push(syntheticEdge);
                }
            });
        });
    }

    return cyJson;
}
