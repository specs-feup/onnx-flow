import { useRef, useState , ComponentPropsWithRef } from "react";
import MenuBar from "./MenuBar.tsx";
import SidePanel from "./SidePanel.tsx";

import CytoscapeGraph,{ type CytoscapeData } from "./Cytoscape.tsx";

import "./App.css";

function App() {
    const [isSidePanelVisible, setSidePanelVisibility] = useState(false);
    const [cytoscapeData, setCytoscapeData] = useState<cytoscape.ElementDefinition[] | null>(null);
    const [cytoscapeLayout, setCytoscapeLayout] = useState<string>("fcose");

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
            />

            {isSidePanelVisible && (
              <SidePanel
                  style={{
                    gridArea: "sidepanel",
                    backgroundColor: "#2c2a30",
                    padding: "10px",
                  }}
              />
            )}

            <CytoscapeGraph
              style={{ gridArea: "cytoscape" }}
              cytoscapeData={cytoscapeData}
            />

        </main>
    );
}

export default App;
