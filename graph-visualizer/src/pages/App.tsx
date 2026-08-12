import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import MenuBar from "@/components/MenuBar.tsx";
import SidePanel from "@/components/SidePanel.tsx";
import CytoscapeGraph from "@/components/Cytoscape.tsx";
import NodePopup from "@/components/visualizer/NodeWindow.tsx";
import EdgePopup from "@/components/visualizer/EdgeWindow.tsx";
import RestoreModal from "@/components/editor/RestoreModal.tsx";
import type { CytoscapeData } from "@/types/Cytoscape.ts";
import {
    fetchGraph,
    regionsToCompoundNodes,
    type TransformationOpportunity,
} from "@/api/api.ts";
import "@/styles/App.css";
import defaultStylesheet from "@/styles/cytoscape/default.ts";
import { valueNodeExtractor } from "@/utils/ValueNodeExtractor.ts";

function App() {
    const { sessionId } = useParams();

    const SAVED_GRAPH_KEY = `saved_cytoscape_data_${sessionId || "default"}`;
    const PENDING_RESTORE_KEY = `pending_restore_${sessionId || "default"}`;

    const saveGraphToSession = (data: CytoscapeData) => {
        try {
            sessionStorage.setItem(SAVED_GRAPH_KEY, JSON.stringify(data));
            sessionStorage.setItem(PENDING_RESTORE_KEY, "true");
        } catch (err) {
            console.error("Error saving graph to sessionStorage:", err);
        }
    };

    // End Session when closing Tab
    useEffect(() => {
        const endCurrentSession = () => {
            const url = `http://localhost:3000/api/sessions/${sessionId}/end`;
            navigator.sendBeacon(url);
        };

        window.addEventListener("beforeunload", endCurrentSession);

        return () => {
            window.removeEventListener("beforeunload", endCurrentSession);
        };
    }, [sessionId]);

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
    const [isRestoreModalOpen, setRestoreModalOpen] = useState(false);
    const valueNodes = useMemo(() => valueNodeExtractor(cytoscapeData), [cytoscapeData]);

    if (!cytoscapeData)
        fetchGraph(3000, sessionId!)
            .then((data) => setCytoscapeData(regionsToCompoundNodes(data)))
            .catch((err) => console.log(err));

    const handleToggleEditorMode = (activate: boolean) => {
        if (activate) {
            const hasPending = sessionStorage.getItem(PENDING_RESTORE_KEY) === "true";
            const savedData = sessionStorage.getItem(SAVED_GRAPH_KEY);
            if (hasPending && savedData) {
                setRestoreModalOpen(true);
                return;
            }
        }
        setEditorMode(activate);
    };

    const handleRestoreChanges = () => {
        const savedDataStr = sessionStorage.getItem(SAVED_GRAPH_KEY);
        if (savedDataStr) {
            try {
                const restoredData = JSON.parse(savedDataStr);
                setCytoscapeData(restoredData);
            } catch (e) {
                console.error("Error parsing saved graph data:", e);
            }
        }
        sessionStorage.removeItem(PENDING_RESTORE_KEY);
        setRestoreModalOpen(false);
        setEditorMode(true);
    };

    const handleDiscardChanges = () => {
        sessionStorage.removeItem(SAVED_GRAPH_KEY);
        sessionStorage.removeItem(PENDING_RESTORE_KEY);
        setRestoreModalOpen(false);
        setEditorMode(true);
    };

    const handleNodeDelete = (nodeId: string) => {
        if (!cytoscapeData) return;
        const updatedNodes = cytoscapeData.elements.nodes.filter((node: any) => node.data.id !== nodeId);
        const updatedEdges = cytoscapeData.elements.edges.filter(
            (edge: any) => edge.data.source !== nodeId && edge.data.target !== nodeId
        );
        const newData: CytoscapeData = {
            ...cytoscapeData,
            elements: {
                nodes: updatedNodes,
                edges: updatedEdges,
            },
        };
        if (nodeToEdit?.id === nodeId) {
            setNodeToEdit(null);
        }
        setCytoscapeData(newData);
        saveGraphToSession(newData);
    };

    const handleNodeSubmit = (nodePayload: any, pos: { x: number; y: number } | null) => {
        if (!cytoscapeData) return;

        let newData: CytoscapeData;

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

            newData = {
                ...cytoscapeData,
                elements: {
                    nodes: updatedNodes,
                    edges: [...updatedEdges, ...newEdges],
                },
            };

            // Fechar o modo de edição
            setNodeToEdit(null);
            setSidePanelVisibility(false);

        } else {
            // --- LÓGICA DE CRIAÇÃO ---
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

            newData = {
                ...cytoscapeData,
                elements: {
                    edges: [...cytoscapeData.elements.edges, ...newEdges],
                    nodes: [...cytoscapeData.elements.nodes, newNode],
                },
            };
            setNewNodePos(null);
        }

        setCytoscapeData(newData);
        saveGraphToSession(newData);
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
            <RestoreModal
                isOpen={isRestoreModalOpen}
                onRestore={handleRestoreChanges}
                onDiscard={handleDiscardChanges}
            />

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
                    setMode: handleToggleEditorMode,
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
                        overflowY: "auto",
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
                        setMode: handleToggleEditorMode,
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
                editorMode={editorMode}
                onNodeSelected={(node: any, pos: { x: number; y: number }) => {
                    setSelectedEdge(null);
                    setEdgePopupPos(null);
                    setSelectedNode(node);
                    setPopupPos(pos);
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
                onEditNodeRequested={(node: any) => {
                    setSelectedEdge(null);
                    setEdgePopupPos(null);
                    setSelectedNode(null);
                    setPopupPos(null);
                    setNodeToEdit(node);
                    setSidePanelVisibility(true);
                    setEditorMode(true);
                }}
                onDeleteNodeRequested={handleNodeDelete}
            />
        </main>
    );
}

export default App;
