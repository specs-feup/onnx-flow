import type { CSSProperties } from "react";
import { type TransformationOpportunity } from "@/api/api.ts";
import type { CytoscapeData } from "@/types/Cytoscape.ts";
import TransformationOps from "@/components/visualizer/TransformationOps.tsx";
import NodeAdder from "@/components/editor/NodeAdder.tsx";

export default function SidePanel(props: {
    style: CSSProperties;
    transformationOps: {
        ops: TransformationOpportunity[];
        setOps: (transformationOps: TransformationOpportunity[]) => void;
    };
    setCytoscapeData: (data: CytoscapeData | null) => void;
    transformationsHistory: {
        undo: { stack: string[]; setStack: (history: string[]) => void };
        redo: { stack: string[]; setStack: (history: string[]) => void };
    };
    editorMode: {
        isActive: boolean;
        setMode: (activate: boolean) => void;
    };
    newNodePosition?: { x: number; y: number } | null;
    onCreateNode?: (nodePayload: any, pos: { x: number; y: number } | null) => void;
    valueNodes: Array<unknown>;
    graphNodes?: Array<unknown>;
    nodeToEdit?: any;
}) {
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
