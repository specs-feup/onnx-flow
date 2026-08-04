import { type CSSProperties } from "react";
import Dropdown from "./Dropdown";
import ColorPicker from "./colorPicker";
import type { CytoscapeData } from "./Cytoscape.tsx";
import { fetchGraph, fetchTransformationOpportunities, undoTransformation, type TransformationOpportunity } from "./api/api.ts";
import { Link } from "react-router-dom";


import  Themess from "./themes.tsx";

export default function MenuBar(props: {
    style: CSSProperties;
    setCytoscapeData: (cytoscapeData: CytoscapeData | null) => void;
    panelVisibility: { isVisible: boolean; setVisibility: (visible: boolean) => void };
    setLayout: (layout: string) => void;
    setStylesheet: (stylesheet: string) => void;
    nodeColor?: string;
    setNodeColor?: (c: string) => void;
    setTransformationOps: (transformationsOps: TransformationOpportunity[]) => void;
    transformationsHistory: {
        undo: {stack: string[], setStack: (history: string[]) => void}
        redo: {stack: string[], setStack: (history: string[]) => void}
    }
    editorMode: {
        isActive: boolean,
        setMode: (activate: boolean) => void
    }
}) {

    return (
        <header style={props.style}>
            <Link to="/">Home</Link>
            <h1>ONNX Graph Visualizer</h1>
            <div className="button-group">
                {/*<button onClick={async () => props.setCytoscapeData(await fetchGraph(3000))}>Get Graph</button>*/}
                <Dropdown setLayout={props.setLayout} />
                <Themess setStylesheet={props.setStylesheet} />
                <div style={{ display: "flex", alignItems: "center" }}>
                    <ColorPicker
                        value={props.nodeColor}
                        onChange={(c) => props.setNodeColor?.(c)}
                    />
                </div>

                {!props.editorMode.isActive ?
                    <button onClick={async () => {
                        props.setTransformationOps(await fetchTransformationOpportunities(3000));
                        props.panelVisibility.setVisibility(props.panelVisibility.isVisible ? false : true);
                    }}>Transformation Opportunities</button>
                    :
                    <button onClick={() => {props.panelVisibility.setVisibility(props.panelVisibility.isVisible ? false : true);}}>&#9776;</button>
                }

                {!props.editorMode.isActive &&    
                <>
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
                </>
                }
                
                <button onClick={() => {props.editorMode.setMode(props.editorMode.isActive ? false : true)}}>Editor</button>
                
            </div>
        </header>
    );
}
