import { useState, type CSSProperties } from "react";
import OnnxUploadButton from "./OnnxUploadButton";
import Dropdown from "./Dropdown";
import ColorPicker from "./colorPicker";
import type { CytoscapeData } from "./Cytoscape.tsx";
import { exportOnnxJson, exportUnifiedJson, fetchGraph, fetchTransformationOpportunities, undoTransformation, type TransformationOpportunity } from "./api/api.ts";



export default function MenuBar(props: {
    style: CSSProperties;
    setCytoscapeData: (cytoscapeData: CytoscapeData | null) => void;
    panelVisibility: { isVisible: boolean; setVisibility: (visible: boolean) => void };
    setLayout: (layout: string) => void;
    nodeColor?: string;
    setNodeColor?: (c: string) => void;
    setTransformationOps: (transformationsOps: TransformationOpportunity[]) => void;
    transformationsHistory: {
        undo: {stack: string[], setStack: (history: string[]) => void}
        redo: {stack: string[], setStack: (history: string[]) => void}
    }
}) {
    const [filename, setFilename] = useState<string | null>(null);

    return (
        <header style={props.style}>
            <h1>ONNX Graph Visualizer</h1>
            <div className="button-group">
                <OnnxUploadButton
                    style={{
                        display: "flex",
                        gridArea: "cytoscape",
                        width: "100%",
                        height: "100%",
                        border: "2px solid white",
                    }}
                    setCytoscapeData={props.setCytoscapeData}
                    setFilename={setFilename}
                />
                <button onClick={() => props.setCytoscapeData(null)}>Clear Graph</button>
                {props.panelVisibility.isVisible ? (
                    <button onClick={() => props.panelVisibility.setVisibility(false)}>
                        Hide Side Panel
                    </button>
                ) : (
                    <button onClick={() => props.panelVisibility.setVisibility(true)}>
                        Display Side Panel
                    </button>
                )}
                <Dropdown setLayout={props.setLayout} />
                <div style={{ display: "flex", alignItems: "center" }}>
                    <ColorPicker
                        value={props.nodeColor}
                        onChange={(c) => props.setNodeColor?.(c)}
                    />
                </div>
                {filename && <button onClick={async () => await fetch(`http://localhost:4000/server/start/${filename}`, { method: "POST" })}>&#9654;</button>}
                <button onClick={async () => props.setCytoscapeData(await fetchGraph(3000))}>Get Graph</button>
                <button onClick={async () => {
                    props.setTransformationOps(await fetchTransformationOpportunities(3000));
                    props.panelVisibility.setVisibility(true);
                }}>Transformation Opportunities</button>
                <button 
                    disabled={props.transformationsHistory.undo.stack.length == 0} 
                    onClick={async () => {
                        if (props.transformationsHistory.undo.stack.length == 0) return;
                        const actualState: string = props.transformationsHistory.undo.stack.pop();
                        const tempRedoStack: string[] = props.transformationsHistory.redo.stack;
                        props.transformationsHistory.redo.setStack([...tempRedoStack, actualState]);
                        props.setCytoscapeData(null);
                        props.setTransformationOps([]);
                        await undoTransformation(3000);
                        props.setCytoscapeData(await fetchGraph(3000));
                        props.setTransformationOps(await fetchTransformationOpportunities(3000));
                    }}
                    >↩ Undo
                </button>
                <button 
                    disabled={props.transformationsHistory.redo.stack.length == 0}
                    onClick={async () => {
                        if (props.transformationsHistory.redo.stack.length == 0) return;
                        const actualState: string = props.transformationsHistory.redo.stack.pop();
                        const tempUndoStack: string[] = props.transformationsHistory.undo.stack;
                        props.transformationsHistory.undo.setStack([...tempUndoStack, actualState]);
                        props.setCytoscapeData(null);
                        props.setTransformationOps([]);
                        await undoTransformation(3000);
                        props.setCytoscapeData(await fetchGraph(3000));
                        props.setTransformationOps(await fetchTransformationOpportunities(3000));
                    }}>↪ Redo
                </button>
                <a href="http://localhost:3000/api/export/onnx-json">&#10515; Onnx in .json</a>
                <a href="http://localhost:3000/api/export/unified-json">&#10515; Unified .json</a>
                
            </div>
        </header>
    );
}
