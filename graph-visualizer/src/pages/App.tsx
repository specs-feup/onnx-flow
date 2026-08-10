import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import MenuBar from "@/components/MenuBar.tsx";
import SidePanel from "@/components/SidePanel.tsx";
import type { CytoscapeData } from "@/types/Cytoscape.ts";
import CytoscapeGraph from "@/components/Cytoscape.tsx";
import NodePopup from "@/components/visualizer/NodeWindow.tsx";
import EdgePopup from "@/components/visualizer/EdgeWindow.tsx";
import {
    endSession,
    fetchGraph,
    regionsToCompoundNodes,
    type TransformationOpportunity,
} from "@/api/api.ts";
import "@/styles/App.css";
import defaultStylesheet from "@/styles/cytoscape/default.ts";
import { valueNodeExtractor } from "@/utils/ValueNodeExtractor.ts";

function App() {
    const { sessionId } = useParams();

    // End Session when closing Tab
    useEffect(() => {
        const endCurrentSession = () => {
            endSession(3000, sessionId!);
        };

        window.addEventListener("beforeunload", endCurrentSession);

        return () => {
            window.removeEventListener("beforeunload", endCurrentSession);
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
    const [popupPos, setPopupPos] = useState<{ x: number; y: number } | null>(null);
    const [selectedEdge, setSelectedEdge] = useState<any | null>(null);
    const [edgePopupPos, setEdgePopupPos] = useState<{ x: number; y: number } | null>(null);
    const [transformationOps, setTransformationOps] = useState<TransformationOpportunity[]>([]);
    const [undoStack, setUndoStack] = useState<string[]>([]);
    const [redoStack, setRedoStack] = useState<string[]>([]);
    const [editorMode, setEditorMode] = useState(false);
    const [newNodePos, setNewNodePos] = useState<{ x: number; y: number } | null>(null);
    const [nodeToEdit, setNodeToEdit] = useState<any | null>(null);
    const valueNodes = useMemo(() => valueNodeExtractor(cytoscapeData), [cytoscapeData]);

    if (!cytoscapeData)
        fetchGraph(3000, sessionId!)
            .then((data) => setCytoscapeData(regionsToCompoundNodes(data)))
            .catch((err) => console.log(err));

    const handleNodeSubmit = (nodePayload: any, pos: { x: number; y: number } | null) => {
        if (!cytoscapeData) return;

        if (nodeToEdit) {
            // --- LÓGICA DE EDIÇÃO ---
            const originalId = nodeToEdit.id;
            const newId = nodePayload.onnxData.id;

            // 1. Substituir os dados do nó existente
            const updatedNodes = cytoscapeData.elements.nodes.map((node) => {
                if (node.data.id === originalId) {
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            id: newId,
                            onnxData: nodePayload.onnxData,
                        },
                    };
                }
                return node;
            });

            // 2. Atualizar as arestas existentes
            // Alterar o ID nas arestas que saíam ou entravam neste nó,
            // e depois remover as antigas arestas de entrada para as reconstruir.
            let updatedEdges = cytoscapeData.elements.edges
                .map((edge) => {
                    const newEdge = { ...edge };
                    if (newEdge.data.source === originalId) newEdge.data.source = newId;
                    if (newEdge.data.target === originalId) newEdge.data.target = newId;
                    return newEdge;
                })
                .filter((edge) => edge.data.target !== newId); // Limpar os inputs antigos

            // 3. Gerar as novas arestas de entrada (se for OperationNode)
            const newEdges = [];
            if (nodePayload.onnxData.kind === "OperationNode") {
                for (const input of nodePayload.onnxData.inputs) {
                    newEdges.push({
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
                    });
                }
            }

            setCytoscapeData({
                ...cytoscapeData,
                elements: {
                    nodes: updatedNodes,
                    edges: [...updatedEdges, ...newEdges],
                },
            });

            // Fechar o modo de edição
            setNodeToEdit(null);
            setSidePanelVisibility(false);

        } else {
            // --- LÓGICA DE CRIAÇÃO (A tua lógica original) ---
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

            setCytoscapeData({
                ...cytoscapeData,
                elements: {
                    edges: [...cytoscapeData.elements.edges, ...newEdges],
                    nodes: [...cytoscapeData.elements.nodes, newNode],
                },
            });
            setNewNodePos(null);
        }
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
                    onCreateNode={handleNodeSubmit}
                    valueNodes={valueNodes}
                    nodeToEdit={nodeToEdit} 
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
            <EdgePopup
                selectedEdge={selectedEdge}
                popupPos={edgePopupPos}
                onClose={() => {
                    setSelectedEdge(null);
                    setEdgePopupPos(null);
                }}
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
                    setSelectedEdge(null);
                    setEdgePopupPos(null);
                    if (editorMode) {
                        setSidePanelVisibility(true);
                        setNodeToEdit(node);
                        setSelectedNode(null);
                        setPopupPos(null);
                    } else {
                        setSelectedNode(node);
                        setPopupPos(pos);
                        setNodeToEdit(null);
                    }
                    
                }}
                onEdgeSelected={(edge: any, pos: { x: number; y: number }) => {
                    setSelectedEdge(edge);
                    setEdgePopupPos(pos);
                    setSelectedNode(null);
                    setPopupPos(null);
                }}
                onAddNodeRequested={(pos) => {
                    setNewNodePos(pos);
                    setNodeToEdit(null);
                    setSidePanelVisibility(true);
                    setEditorMode(true);
                }}
            />
        </main>
    );
}

export default App;
