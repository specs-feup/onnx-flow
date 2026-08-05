import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import MenuBar from "../MenuBar.tsx";
import SidePanel from "../SidePanel.tsx";
import CytoscapeGraph, { type CytoscapeData } from "../Cytoscape.tsx";
import NodePopup from "../nodeWindow.tsx";
import { endSession, fetchGraph, type TransformationOpportunity } from "../api/api.ts";
import "../App.css";
import defaultStylesheet from "../styleSheets/default.ts";
import { valueNodeExtractor } from "../graphicalEditor/ValueNodeExtractor.ts";

function Visualizer() {
    const {sessionId} = useParams();

    // End Session when closing Tab
    useEffect(() => {
    const endCurrentSession = () => {
        endSession(3000, sessionId!);
    };

    window.addEventListener('beforeunload', endCurrentSession);

    return () => {
      window.removeEventListener('beforeunload', endCurrentSession);
    };
  }, []);

    const [cytoscapeStylesheet, setCytoscapeStylesheet] =
        useState<cytoscape.CssStyleDeclaration>(defaultStylesheet);
    const [isSidePanelVisible, setSidePanelVisibility] = useState(false);
    const [cytoscapeData, setCytoscapeData] = useState<CytoscapeData | null>();
    const [cytoscapeLayout, setCytoscapeLayout] = useState<cytoscape.LayoutOptions>({
        name: "fcose",
    });
    const [nodeColor, setNodeColor] = useState<string>("#533b6e");
    const [selectedNode, setSelectedNode] = useState<any | null>(null);
    const [transformationOps, setTransformationOps] = useState<TransformationOpportunity[]>([]);
    const [popupPos, setPopupPos] = useState<{ x: number; y: number } | null>(null);
    const [undoStack, setUndoStack] = useState<string[]>([]);
    const [redoStack, setRedoStack] = useState<string[]>([]);
    const [editorMode, setEditorMode] = useState(false);
    const [newNodePos, setNewNodePos] = useState<{ x: number; y: number } | null>(null);
    const valueNodes = useMemo(() => valueNodeExtractor(cytoscapeData), [cytoscapeData]);

    if (!cytoscapeData)
        fetchGraph(3000, sessionId!)
            .then((data) => setCytoscapeData(data))
            .catch((err) => console.log(err));

    // NOVA FUNÇÃO: Manipulador para inserir o novo nó no estado do CytoscapeData
    const handleCreateNode = (nodePayload: any, pos: { x: number; y: number } | null) => {
        if (!cytoscapeData) return;

        const positionToUse = pos || { x: 0, y: 0 };
        const newId =
            nodePayload.label === "" ? `node_${Math.random().toString(36)}` : nodePayload.label;

        const newNode = {
            data: {
                id: newId,
                onnxData: nodePayload.onnxData,
            },
            position: positionToUse,
            group: "nodes",
            removed: false,
            selected: false,
            selectable: true,
            locked: false,
            grabbable: true,
            classes: "",
        };

        const newEdges = [];
        if (nodePayload.onnxData.kind === "OperationNode") {
            for (const input of nodePayload.onnxData.inputs) {
                const newEdge = {
                    data: {
                        id: `${Math.random().toString(36)}`,
                        source: input,
                        target: newId,
                    },
                    group: "edges",
                    removed: false,
                    selected: false,
                    selectable: true,
                    locked: false,
                    grabbable: true,
                    classes: "",
                };
                newEdges.push(newEdge);
            }
        }

        // Atualiza o grafo adicionando o novo nó aos elementos existentes
        setCytoscapeData({
            ...cytoscapeData,
            elements: {
                edges: [...cytoscapeData.elements.edges, ...newEdges],
                nodes: [...cytoscapeData.elements.nodes, newNode],
            },
        });

        // Limpa a posição após a criação
        setNewNodePos(null);
    };

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
                    padding: "25px",
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
                    undo: { stack: undoStack, setStack: setUndoStack },
                    redo: { stack: redoStack, setStack: setRedoStack },
                }}
                editorMode={{
                    isActive: editorMode,
                    setMode: setEditorMode,
                }}
            />

            {isSidePanelVisible && (
                <SidePanel
                    style={{
                        gridArea: "sidepanel",
                        backgroundColor: "#2c2a30",
                        padding: "10px",
                        display: "flex",
                        flexDirection: "column",
                        overflow: "scroll",
                        gap: "1rem",
                    }}
                    transformationOps={{
                        ops: transformationOps,
                        setOps: setTransformationOps,
                    }}
                    setCytoscapeData={setCytoscapeData}
                    transformationsHistory={{
                        undo: { stack: undoStack, setStack: setUndoStack },
                        redo: { stack: redoStack, setStack: setRedoStack },
                    }}
                    editorMode={{
                        isActive: editorMode,
                        setMode: setEditorMode,
                    }}
                    newNodePosition={newNodePos}
                    onCreateNode={handleCreateNode}
                    valueNodes={valueNodes}
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
                style={{
                    gridArea: "cytoscape",
                    border: "3px solid rgb(74, 70, 82)",
                    margin: "30px",
                    marginTop: "20px",
                    marginLeft: "25px",
                    borderRadius: "5px",
                    backgroundColor: "#1d1b20",
                }}
                cytoscapeData={cytoscapeData}
                layout={cytoscapeLayout}
                stylesheet={cytoscapeStylesheet}
                nodeColor={nodeColor}
                selectedNodeId={selectedNode?.id ?? null}
                onNodeSelected={(node: any, pos: { x: number; y: number }) => {
                    setSelectedNode(node);
                    setPopupPos(pos);
                }}
                onAddNodeRequested={(pos) => {
                    setNewNodePos(pos);
                    setSidePanelVisibility(true);
                    setEditorMode(true);
                }}
            />
        </main>
    );
}

export default Visualizer;
