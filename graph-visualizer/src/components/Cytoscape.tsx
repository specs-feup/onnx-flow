/**
 * @file Cytoscape.tsx
 * @description Core graph rendering component wrapping react-cytoscapejs. Manages Cytoscape.js
 * lifecycle, layout execution (fcose, dagre, elk, bfs, grid, concentric), compound node
 * expansion/collapse, radial context menus (cxtmenu), node/edge styling updates, and event handling.
 */

import CytoscapeComponent from "react-cytoscapejs";
import fcose from "cytoscape-fcose";
import cytoscape from "cytoscape";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import dagre from "cytoscape-dagre";
import elk from 'cytoscape-elk';
import cxtmenu from "cytoscape-cxtmenu";
import chroma from "chroma-js";
import expandCollapse from "cytoscape-expand-collapse";

import defaultStylesheet from "@/styles/cytoscape/default.ts";
import type { CytoscapeData } from "@/types/Cytoscape.ts";

// Register layout and UI plugins with the Cytoscape core
cytoscape.use(dagre);
cytoscape.use(fcose);
cytoscape.use(elk);
cytoscape.use(cxtmenu);
expandCollapse(cytoscape);

// --- HELPER FUNCTIONS ---

/**
 * Tracks currently expanded subgraph region index for each compound operation node.
 * Maps `nodeId` -> `regionIndex`.
 */
const expandedRegionsMap = new Map<string, number>();

/**
 * Generates a human-friendly name for a subgraph region (e.g. If branch or Loop body).
 *
 * @param nodeData - The parent OperationNode's data object
 * @param region - Region data object
 * @param index - Index of the region within the operation's regions array
 * @returns Human-readable label for the region
 */
function getRegionName(nodeData: any, region: any, index: number): string {
    if (region?.name) return region.name;
    if (region?.data?.name) return region.data.name;
    const opType = nodeData?.onnxData?.opType;
    if (opType === "If") {
        return index === 0 ? "then_branch" : index === 1 ? "else_branch" : `branch_${index + 1}`;
    }
    if (opType === "Loop" || opType === "Scan") {
        return index === 0 ? "body" : `region_${index + 1}`;
    }
    return `Region ${index + 1}`;
}

/**
 * Expands a specific subgraph region within a compound node and adjusts the visibility of internal child elements.
 *
 * @param cy - The Cytoscape core instance
 * @param api - The cytoscape-expand-collapse API instance
 * @param nodeId - ID of the parent compound node
 * @param regionIndex - Index of the region to display
 */
function showRegion(cy: cytoscape.Core, api: any, nodeId: string, regionIndex: number) {
    const parentNode = cy.getElementById(nodeId);
    if (!parentNode || parentNode.empty()) return;

    if (api && api.isExpandable(parentNode)) {
        api.expand(parentNode);
    }

    expandedRegionsMap.set(nodeId, regionIndex);

    const children = cy.nodes(`[parent = "${nodeId}"]`);
    children.forEach((child) => {
        const childRegionIdx = child.data("regionIndex");
        if (childRegionIdx === regionIndex || childRegionIdx === undefined) {
            child.style("display", "element");
            child.connectedEdges().forEach((edge) => {
                const srcVis = cy.getElementById(edge.data("source")).style("display") !== "none";
                const tgtVis = cy.getElementById(edge.data("target")).style("display") !== "none";
                if (srcVis && tgtVis) {
                    edge.style("display", "element");
                }
            });
        } else {
            child.style("display", "none");
            child.connectedEdges().style("display", "none");
        }
    });
}

/**
 * Collapses and hides all child regions and elements within a compound node.
 *
 * @param cy - The Cytoscape core instance
 * @param api - The cytoscape-expand-collapse API instance
 * @param nodeId - ID of the parent compound node to collapse
 */
function hideAllRegions(cy: cytoscape.Core, api: any, nodeId: string) {
    const parentNode = cy.getElementById(nodeId);
    if (!parentNode || parentNode.empty()) return;

    expandedRegionsMap.delete(nodeId);

    if (api && api.isCollapsible(parentNode)) {
        api.collapse(parentNode);
    } else {
        const children = cy.nodes(`[parent = "${nodeId}"]`);
        children.style("display", "none");
        children.connectedEdges().style("display", "none");
    }
}

/**
 * Collapses all compound nodes across the entire graph.
 *
 * @param cy - The Cytoscape core instance
 * @param api - The cytoscape-expand-collapse API instance
 */
function handleCollapseAll(cy: cytoscape.Core, api: any) {
    expandedRegionsMap.clear();
    cy.elements().removeStyle("display");
    if (api && typeof api.collapseAll === "function") {
        api.collapseAll({ animate: false });
    }
}

/**
 * Expands all compound nodes across the entire graph.
 *
 * @param cy - The Cytoscape core instance
 * @param api - The cytoscape-expand-collapse API instance
 */
function handleExpandAll(cy: cytoscape.Core, api: any) {
    expandedRegionsMap.clear();
    if (api && typeof api.expandAll === "function") {
        api.expandAll({ animate: false });
    }
    cy.elements().removeStyle("display");
}

/**
 * Initializes interactive radial context menus for nodes, edges, and the canvas background.
 *
 * @param cy - The Cytoscape core instance
 * @param api - The cytoscape-expand-collapse API instance
 * @param editorMode - Boolean flag indicating if editor mode actions (add/edit/delete) should be enabled
 * @param onAddNodeRequested - Callback when user clicks 'Add Node' on canvas
 * @param onEditNodeRequested - Callback when user clicks 'Edit' on a node
 * @param onDeleteNodeRequested - Callback when user clicks 'Delete' on a node
 * @param onDeleteEdgeRequested - Callback when user clicks 'Delete' on an edge
 * @returns Array of initialized cxtmenu plugin instances
 */
const setupContextMenus = (
    cy: cytoscape.Core,
    api: any,
    editorMode: boolean = false,
    onAddNodeRequested?: (pos: any) => void,
    onEditNodeRequested?: (node: any) => void,
    onDeleteNodeRequested?: (nodeId: string) => void,
    onDeleteEdgeRequested?: (edgeId: string) => void
) => {
    let clickedPos = { x: 0, y: 0 };
    cy.on("cxttapstart", (e) => { if (e.position) clickedPos = e.position; });

    const nodeMenu = cy.cxtmenu({
        selector: "node",
        activeFillColor: "#2d293300",
        commands: (ele: any) => {
            const onnxData = ele.data("onnxData");
            const nodeId = ele.id();

            const cmds: any[] = [
                { fillColor: "rgba(64, 67, 75, 0.9)", content: "Log Info", select: (e: any) => console.log("Selected node ID:", e.id()) },
                ...(editorMode ? [
                    { fillColor: "rgba(64, 67, 75, 0.9)", content: "Edit", select: (e: any) => onEditNodeRequested?.(e.data()) },
                    { 
                        fillColor: "rgba(75, 26, 38, 0.9)", 
                        content: "Delete", 
                        select: (e: any) => {
                            const deletedId = e.id();
                            cy.remove(e);
                            onDeleteNodeRequested?.(deletedId);
                        } 
                    },
                ] : []),
            ];

            if (onnxData?.kind === "OperationNode" && Array.isArray(onnxData.regions) && onnxData.regions.length > 1) {
                const currentExpandedIdx = expandedRegionsMap.get(nodeId);

                onnxData.regions.forEach((region: any, idx: number) => {
                    const regionName = getRegionName(ele.data(), region, idx);
                    const isThisExpanded = currentExpandedIdx === idx;

                    if (isThisExpanded) {
                        cmds.push({
                            fillColor: "rgba(46, 117, 89, 0.9)",
                            content: `Collapse ${regionName}`,
                            select: () => hideAllRegions(cy, api, nodeId),
                        });
                    } else {
                        cmds.push({
                            fillColor: "rgba(46, 89, 117, 0.9)",
                            content: `Expand ${regionName}`,
                            select: () => showRegion(cy, api, nodeId, idx),
                        });
                    }
                });
            } else {
                if (api?.isCollapsible(ele)) {
                    cmds.push({ fillColor: "rgba(64, 67, 75, 0.9)", content: "Collapse", select: (e: any) => api.collapse(e) });
                }
                if (api?.isExpandable(ele)) {
                    cmds.push({ fillColor: "rgba(64, 67, 75, 0.9)", content: "Expand", select: (e: any) => api.expand(e) });
                }
            }

            return cmds;
        }
    });

    const edgeMenu = cy.cxtmenu({
        selector: "edge",
        activeFillColor: "#2d293300",
        commands: (_ele: any) => {
            const cmds: any[] = [
                {
                    fillColor: "rgba(64, 67, 75, 0.9)",
                    content: "Log Info",
                    select: (e: any) => console.log("Selected edge ID:", e.id())
                },
                ...(editorMode ? [
                    {
                        fillColor: "rgba(75, 26, 38, 0.9)",
                        content: "Delete",
                        select: (e: any) => {
                            const deletedId = e.id();
                            cy.remove(e);
                            onDeleteEdgeRequested?.(deletedId);
                        }
                    }
                ] : []),
            ];
            return cmds;
        }
    });

    const coreMenu = cy.cxtmenu({
        selector: "core",
        activeFillColor: "#533b6e00",
        commands: [
            ...(editorMode ? [
                { fillColor: "rgba(32, 70, 92, 0.79)", content: "＋ Add Node", select: () => onAddNodeRequested?.(clickedPos) },
            ] : []),
            { fillColor: "rgba(64, 67, 75, 0.9)", content: "Expand All", select: () => handleExpandAll(cy, api) },
            { fillColor: "rgba(64, 67, 75, 0.9)", content: "Collapse All", select: () => handleCollapseAll(cy, api) }
        ]
    });

    return [nodeMenu, edgeMenu, coreMenu];
};

/**
 * Updates dynamic styling on the Cytoscape graph (node background colors, highlight for selected node).
 *
 * @param cy - The Cytoscape core instance
 * @param options - Style parameters including active stylesheet, base nodeColor, and selectedNodeId
 */
const updateStyles = (cy: cytoscape.Core, { stylesheet, nodeColor = "#533b6e", selectedNodeId }: any) => {
    cy.nodes().removeStyle("background-color");
    if ((!stylesheet || stylesheet === defaultStylesheet) && nodeColor) {
        cy.nodes().style("background-color", nodeColor);
    }
    if (selectedNodeId) {
        const selected = cy.getElementById(selectedNodeId);
        if (selected?.nonempty()) {
            selected.style("background-color", chroma(nodeColor).brighten(2).hex());
        }
    }
};

// --- MAIN COMPONENT ---

/**
 * Properties for the CytoscapeGraph component.
 */
type Props = {
    /** CSS styling container options */
    style: CSSProperties;
    /** The Cytoscape graph data payload containing nodes and edges */
    cytoscapeData: CytoscapeData | null;
    /** Layout options specification (e.g. fcose, dagre, elk) */
    layout: cytoscape.LayoutOptions;
    /** Active Cytoscape CSS stylesheet */
    stylesheet?: any;
    /** Base color used for node backgrounds */
    nodeColor?: string;
    /** Currently selected node ID for visual highlight */
    selectedNodeId?: string | null;
    /** Callback fired when a node is tapped/selected */
    onNodeSelected?: (node: any, pos: { x: number; y: number }) => void;
    /** Callback fired when an edge is tapped/selected */
    onEdgeSelected?: (edge: any, pos: { x: number; y: number }) => void;
    /** Callback fired when user requests adding a node at a canvas position */
    onAddNodeRequested?: (pos: { x: number; y: number }) => void;
    /** Callback fired when user requests editing a specific node */
    onEditNodeRequested?: (node: any) => void;
    /** Callback fired when user requests deleting a specific node */
    onDeleteNodeRequested?: (nodeId: string) => void;
    /** Callback fired when user requests deleting a specific edge */
    onDeleteEdgeRequested?: (edgeId: string) => void;
    /** Flag indicating whether editor mode is currently active */
    editorMode?: boolean;
};


/**
 * Interactive Cytoscape graph canvas component for ONNX computation graph exploration and editing.
 *
 * @param props - Component properties (styling, data, layout, callbacks, editorMode flag)
 * @returns JSX element containing the Cytoscape canvas container
 */
export default function CytoscapeGraph({
    style, cytoscapeData, layout, stylesheet, nodeColor, selectedNodeId, onNodeSelected, onEdgeSelected, onAddNodeRequested, onEditNodeRequested, onDeleteNodeRequested, onDeleteEdgeRequested, editorMode = false
}: Props) {
    const cyRef = useRef<cytoscape.Core | null>(null);
    const apiRef = useRef<any>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [cyReady, setCyReady] = useState(false);

    // Helper to safely fetch the instance without rewriting checks
    const getCy = () => cyRef.current && !cyRef.current.destroyed() ? cyRef.current : null;

    // 1. Resize Observer
    useEffect(() => {
        if (!containerRef.current) return;
        const observer = new ResizeObserver(() => {
            getCy()?.resize();
            getCy()?.fit();
        });
        observer.observe(containerRef.current);
        return () => observer.disconnect();
    }, []);

    // 2. Expand/Collapse Init
    useEffect(() => {
        const cy = getCy();
        if (cy && !apiRef.current) {
            apiRef.current = cy.expandCollapse({ layoutBy: null, animate: false, cueEnabled: false });
            apiRef.current.collapseAll({ animate: false });
        }
    }, [cyReady]);

    const prevLayoutRef = useRef<cytoscape.LayoutOptions>(layout);
    const hasInitialLayoutRunRef = useRef(false);

    // 3. Layout updates
    useEffect(() => {
        const cy = getCy();
        if (!cy || !cytoscapeData) return;

        const isExplicitLayoutChange = prevLayoutRef.current !== layout;
        prevLayoutRef.current = layout;

        if (!hasInitialLayoutRunRef.current || isExplicitLayoutChange) {
            hasInitialLayoutRunRef.current = true;
            cy.layout(layout).run();
            return;
        }

        if (!editorMode) {
            cy.layout(layout).run();
        }
    }, [layout, cytoscapeData, editorMode, cyReady]);

    // 4. Styling updates
    useEffect(() => {
        const cy = getCy();
        if (cy && cytoscapeData) {
            updateStyles(cy, { stylesheet, nodeColor, selectedNodeId });
            cy.resize();
        }
    }, [nodeColor, selectedNodeId, cytoscapeData, stylesheet, cyReady]);

    // 5. Events & Context Menus
    useEffect(() => {
        const cy = getCy();
        if (!cy || !cytoscapeData) return;

        const onNodeTap = (e: any) => {
            if (!e.target?.isNode?.() || !onNodeSelected) return;
            const rect = containerRef.current?.getBoundingClientRect();
            const pos = e.renderedPosition;
            onNodeSelected(e.target.data(), rect ? { x: rect.left + pos.x, y: rect.top + pos.y } : { x: 10, y: 10 });
        };
        cy.on("tap", "node", onNodeTap);

        const onEdgeTap = (e: any) => {
            if (!e.target?.isEdge?.() || !onEdgeSelected) return;
            const rect = containerRef.current?.getBoundingClientRect();
            const pos = e.renderedPosition;
            if (onEdgeSelected) {
                onEdgeSelected(e.target.data(), rect ? { x: rect.left + pos.x, y: rect.top + pos.y } : { x: 10, y: 10 });
            }
        };
        cy.on("tap", "edge", onEdgeTap);

        const menus = setupContextMenus(cy, apiRef.current, editorMode, onAddNodeRequested, onEditNodeRequested, onDeleteNodeRequested, onDeleteEdgeRequested);

        return () => {
            cy.off("tap", "node", onNodeTap);
            cy.off("tap", "edge", onEdgeTap);
            menus.forEach(m => m.destroy());
        };
    }, [cyReady, cytoscapeData, editorMode, onNodeSelected, onEdgeSelected, onAddNodeRequested, onEditNodeRequested, onDeleteNodeRequested, onDeleteEdgeRequested]);

    return (
        <div style={style} ref={containerRef}>
            {cytoscapeData && (
                <CytoscapeComponent
                    elements={CytoscapeComponent.normalizeElements(cytoscapeData.elements)}
                    style={{ width: "100%", height: "100%" }}
                    stylesheet={stylesheet || defaultStylesheet}
                    layout={editorMode ? { name: "preset" } : layout}
                    cy={(cy) => {
                        cyRef.current = cy;
                        setCyReady(true);
                    }}
                />
            )}
        </div>
    );
}