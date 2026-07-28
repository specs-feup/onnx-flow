import type { CSSProperties } from "react";
import {
    applyTransformation,
    fetchGraph,
    fetchTransformationOpportunities,
    type TransformationOpportunity,
} from "./api/api.ts";
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
}) {
    return (
        <>
            {props.editorMode.isActive ? (
                <NodeAdder />
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
