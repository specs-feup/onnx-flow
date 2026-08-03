import type { CSSProperties } from "react";
import { type TransformationOpportunity } from "./api/api.ts";
import type { CytoscapeData } from "./Cytoscape.tsx";
import TransformationOps from "./TransformationOps.tsx";
import NodeAdder from "./NodeAdder.tsx";

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
}) {
    return (
        <>
            {props.editorMode.isActive ? (
                <NodeAdder
                    position={props.newNodePosition}
                    onSubmit={props.onCreateNode}
                    valueNodes={props.valueNodes}
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
        </>
    );
}
