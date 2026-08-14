/**
 * @file SidePanel.tsx
 * @description Dynamic side panel component that conditionally renders the NodeAdder form
 * (when in Editor Mode) or the TransformationOps optimization panel (when in Visualizer Mode).
 */

import type { CSSProperties } from "react";
import { type TransformationOpportunity } from "@/api/api.ts";
import type { CytoscapeData } from "@/types/Cytoscape.ts";
import TransformationOps from "@/components/visualizer/TransformationOps.tsx";
import NodeAdder from "@/components/editor/NodeAdder.tsx";

/**
 * Properties for the SidePanel component.
 */
interface SidePanelProps {
    /** CSS style properties for the aside container */
    style: CSSProperties;
    /** Object containing transformation opportunities array and setter */
    transformationOps: {
        ops: TransformationOpportunity[];
        setOps: (transformationOps: TransformationOpportunity[]) => void;
    };
    /** State setter for the active Cytoscape graph data */
    setCytoscapeData: (data: CytoscapeData | null) => void;
    /** Object containing undo/redo transformation history stacks */
    transformationsHistory: {
        undo: { stack: string[]; setStack: (history: string[]) => void };
        redo: { stack: string[]; setStack: (history: string[]) => void };
    };
    /** Object containing active editor mode status and toggle handler */
    editorMode: {
        isActive: boolean;
        setMode: (activate: boolean) => void;
    };
    /** Optional canvas coordinates where a new node was dropped */
    newNodePosition?: { x: number; y: number } | null;
    /** Optional callback when a node is submitted (created or updated) */
    onCreateNode?: (nodePayload: any, pos: { x: number; y: number } | null) => void;
    /** List of value-carrying nodes available in the graph for input bindings */
    valueNodes: Array<unknown>;
    /** Full list of graph nodes for region/parent bindings */
    graphNodes?: Array<unknown>;
    /** Optional node data if editing an existing node */
    nodeToEdit?: any;
}

/**
 * Collapsible side panel hosting either the node editor or transformation panel.
 *
 * @param props - SidePanel properties
 * @returns JSX element containing the active side panel view
 */
export default function SidePanel(props: SidePanelProps) {

    return (
        <aside style={props.style}>
            {props.editorMode.isActive ? (
                <NodeAdder
                    key={props.nodeToEdit ? props.nodeToEdit.id : "new-node"}
                    position={props.newNodePosition}
                    onSubmit={props.onCreateNode}
                    valueNodes={props.valueNodes}
                    graphNodes={props.graphNodes}
                    nodeToEdit={props.nodeToEdit}
                />
            ) : (
                <TransformationOps
                    style={props.style}
                    transformationOps={{
                        ops: props.transformationOps.ops,
                        setOps: props.transformationOps.setOps,
                    }}
                    setCytoscapeData={props.setCytoscapeData}
                    transformationsHistory={props.transformationsHistory}
                />
            )}
        </aside>
    );
}
