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

cytoscape.use(dagre);
cytoscape.use(fcose);
cytoscape.use(elk);
cytoscape.use(cxtmenu);
expandCollapse(cytoscape);

// --- HELPER FUNCTIONS ---

const setupContextMenus = (
    cy: cytoscape.Core, 
    api: any, 
    editorMode: boolean = false,
    onAddNodeRequested?: (pos: any) => void,
    onEditNodeRequested?: (node: any) => void,
    onDeleteNodeRequested?: (nodeId: string) => void
) => {
    let clickedPos = { x: 0, y: 0 };
    cy.on("cxttapstart", (e) => { if (e.position) clickedPos = e.position; });

    const nodeMenu = cy.cxtmenu({
        selector: "node",
        activeFillColor: "#2d293300",
        commands: (ele: any) => [
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
            ...(api?.isCollapsible(ele) ? [{ content: "Collapse", select: (e: any) => api.collapse(e) }] : []),
            ...(api?.isExpandable(ele) ? [{ content: "Expand", select: (e: any) => api.expand(e) }] : [])
        ]
    });

    const coreMenu = cy.cxtmenu({
        selector: "core",
        activeFillColor: "#533b6e00",
        commands: [
            ...(editorMode ? [
                { fillColor: "rgba(32, 70, 92, 0.79)", content: "＋ Add Node", select: () => onAddNodeRequested?.(clickedPos) },
            ] : []),
            { fillColor: "rgba(64, 67, 75, 0.9)", content: "Expand All", select: () => api?.expandAll() },
            { fillColor: "rgba(64, 67, 75, 0.9)", content: "Collapse All", select: () => api?.collapseAll() }
        ]
    });

    return [nodeMenu, coreMenu];
};

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

type Props = {
    style: CSSProperties;
    cytoscapeData: CytoscapeData | null;
    layout: cytoscape.LayoutOptions;
    stylesheet?: any;
    nodeColor?: string;
    selectedNodeId?: string | null;
    onNodeSelected?: (node: any, pos: { x: number; y: number }) => void;
    onEdgeSelected?: (edge: any, pos: { x: number; y: number }) => void;
    onAddNodeRequested?: (pos: { x: number; y: number }) => void;
    onEditNodeRequested?: (node: any) => void;
    onDeleteNodeRequested?: (nodeId: string) => void;
    editorMode?: boolean;
};

export default function CytoscapeGraph({
    style, cytoscapeData, layout, stylesheet, nodeColor, selectedNodeId, onNodeSelected, onEdgeSelected, onAddNodeRequested, onEditNodeRequested, onDeleteNodeRequested, editorMode = false
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

    // 3. Layout updates
    useEffect(() => {
        if (cytoscapeData) {
            getCy()?.layout(layout).run();
            apiRef.current?.collapseAll({ animate: false });
        }
    }, [layout, cytoscapeData]);

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

        const menus = setupContextMenus(cy, apiRef.current, editorMode, onAddNodeRequested, onEditNodeRequested, onDeleteNodeRequested);

        return () => {
            cy.off("tap", "node", onNodeTap);
            cy.off("tap", "edge", onEdgeTap);
            menus.forEach(m => m.destroy());
        };
    }, [cyReady, cytoscapeData, editorMode, onNodeSelected, onEdgeSelected, onAddNodeRequested, onEditNodeRequested, onDeleteNodeRequested]);

    return (
        <div style={style} ref={containerRef}>
            {cytoscapeData && (
                <CytoscapeComponent
                    elements={CytoscapeComponent.normalizeElements(cytoscapeData.elements)}
                    style={{ width: "100%", height: "100%" }}
                    stylesheet={stylesheet || defaultStylesheet}
                    layout={layout}
                    cy={(cy) => {
                        cyRef.current = cy;
                        setCyReady(true);
                    }}
                />
            )}
        </div>
    );
}