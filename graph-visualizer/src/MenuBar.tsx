import type { CSSProperties } from "react";
import OnnxUploadButton from "./OnnxUploadButton";
import Dropdown from './Dropdown';
export default function MenuBar(props: {
    style: CSSProperties;
    setCytoscapeData: (cytoscapeData: CytoscapeData | null) => void;
    panelVisibility: { isVisible: boolean; setVisibility: (visible: boolean) => void };
    setLayout: (layout: string) => void;
}) {
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
            </div>
        </header>
    );
}
