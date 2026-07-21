import { useState, type CSSProperties } from "react";
import OnnxUploadButton from "./OnnxUploadButton";
import Dropdown from "./Dropdown";
import ColorPicker from "./colorPicker";
import type { CytoscapeData } from "./Cytoscape.tsx";

export default function MenuBar(props: {
    style: CSSProperties;
    setCytoscapeData: (cytoscapeData: CytoscapeData | null) => void;
    panelVisibility: { isVisible: boolean; setVisibility: (visible: boolean) => void };
    setLayout: (layout: string) => void;
    nodeColor?: string;
    setNodeColor?: (c: string) => void;
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
            </div>
        </header>
    );
}
