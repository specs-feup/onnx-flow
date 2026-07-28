import { useRef, useState , type ComponentPropsWithRef } from "react";
import MenuBar from "./MenuBar.tsx";
import SidePanel from "./SidePanel.tsx";
import CytoscapeGraph,{ type CytoscapeData } from "./Cytoscape.tsx";
import NodePopup from "./nodeWindow";
import { fetchGraph, type TransformationOpportunity } from "./api/api.ts"
import "./App.css";
import defaultStylesheet from './styleSheets/default.ts';

function App() {
    const [cytoscapeStylesheet, setCytoscapeStylesheet] = useState<cytoscape.CssStyleDeclaration>(defaultStylesheet);
    const [isSidePanelVisible, setSidePanelVisibility] = useState(false);
    const [cytoscapeData, setCytoscapeData] = useState<CytoscapeData | null>();
    const [cytoscapeLayout, setCytoscapeLayout] = useState<cytoscape.LayoutOptions>({name: "fcose"});
    const [nodeColor, setNodeColor] = useState<string>('#533b6e');
    const [selectedNode, setSelectedNode] = useState<any | null>(null);
    const [transformationOps, setTransformationOps] = useState<TransformationOpportunity[]>([]);
    const [popupPos, setPopupPos] = useState<{ x: number; y: number } | null>(null);
    const [undoStack, setUndoStack] = useState<string[]>([]);
    const [redoStack, setRedoStack] = useState<string[]>([]);

    // fetchGraph(3000).then(data => setCytoscapeData(data)).catch(err => console.log(err));

    return (
        <main
          id="spacer"
          style={{
            display: "grid",
            gridTemplateColumns: isSidePanelVisible ? "80vw 20vw" : "100vw",
            gridTemplateRows: "15vh 85vh",
            gridTemplateAreas: isSidePanelVisible
              ? `"menubar sidepanel" "cytoscape sidepanel"`
              : `"menubar" "cytoscape"`,
          }}
        >
            <MenuBar
              style={{
                gridArea: "menubar",
                color: "white", 
                padding: "25px" 
              }}
              setCytoscapeData={setCytoscapeData}
              panelVisibility={{
                  isVisible: isSidePanelVisible,
                  setVisibility: setSidePanelVisibility,
              }}
              setLayout={setCytoscapeLayout}
              setStylesheet={setCytoscapeStylesheet}
              nodeColor={nodeColor}
              selectedNodeId={selectedNode?.id ?? null}
              setNodeColor={setNodeColor}
              setTransformationOps={setTransformationOps}
              transformationsHistory={{
                undo: {stack: undoStack, setStack: setUndoStack},
                redo: {stack: redoStack, setStack: setRedoStack},
              }}
            />

            {isSidePanelVisible && (
              <SidePanel
                  style={{
                    gridArea: "sidepanel",
                    backgroundColor: "#2c2a30",
                    padding: "10px",
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'scroll',
                    gap: '1rem',
                  }}
                  transformationOps={{
                    ops: transformationOps,
                    setOps: setTransformationOps,
                  }}
                  setCytoscapeData={setCytoscapeData}
                  transformationsHistory={{
                    undo: {stack: undoStack, setStack: setUndoStack},
                    redo: {stack: redoStack, setStack: setRedoStack},
                  }}
              />
            )}
            <NodePopup
              selectedNode={selectedNode}
              popupPos={popupPos}
              onClose={() => {
                setSelectedNode(null);
                setPopupPos(null);
              }}
              stylesheet={cytoscapeStylesheet}
              layout={cytoscapeLayout}
              nodeColor={nodeColor}
            />
            <CytoscapeGraph
              style={{ gridArea: "cytoscape", border: "3px solid rgb(74, 70, 82)", margin: "30px",marginTop: "20px",marginLeft: "25px", borderRadius: "5px", backgroundColor: "#1d1b20"}}
              cytoscapeData={cytoscapeData}
              layout={cytoscapeLayout}
              stylesheet={cytoscapeStylesheet}
              nodeColor={nodeColor}
              selectedNodeId={selectedNode?.id ?? null}
              onNodeSelected={(node: any, pos: {x:number; y:number}) => {
                setSelectedNode(node);
                setPopupPos(pos);
              }}
            />
        </main>
    );
}

export default App;
