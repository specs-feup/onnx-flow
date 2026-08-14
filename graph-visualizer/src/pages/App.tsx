import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";

import MenuBar from "@/components/MenuBar.tsx";
import SidePanel from "@/components/SidePanel.tsx";
import CytoscapeGraph from "@/components/Cytoscape.tsx";
import NodePopup from "@/components/visualizer/NodeWindow.tsx";
import EdgePopup from "@/components/visualizer/EdgeWindow.tsx";
import RestoreModal from "@/components/editor/RestoreModal.tsx";
import CompileModal from "@/components/editor/CompileModal.tsx";
import type { CytoscapeData } from "@/types/Cytoscape.ts";
import {
    compileOnnxModel,
    fetchGraph,
    regionsToCompoundNodes,
    type TransformationOpportunity,
} from "@/api/api.ts";
import "@/styles/App.css";
import defaultStylesheet from "@/styles/cytoscape/default.ts";
import { valueNodeExtractor } from "@/utils/ValueNodeExtractor.ts";
import { StandardOps } from "@specs-feup/onnx-flow/Onnx/Schema/definitions/StandardOps";
import { AttributeType } from "@specs-feup/onnx-flow/Onnx/OnnxTypes";

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
    const [originalGraph, setOriginalGraph] = useState<CytoscapeData | null>(null);
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
    const [compileResult, setCompileResult] = useState<{ success: boolean; message: string } | null>(null);
    const [isCompileModalOpen, setCompileModalOpen] = useState(false);
    const valueNodes = useMemo(() => valueNodeExtractor(cytoscapeData), [cytoscapeData]);

    if (!cytoscapeData)
        fetchGraph(3000, sessionId!)
            .then((data) => {
                const formatted = regionsToCompoundNodes(data);
                setOriginalGraph(formatted);
                setCytoscapeData(formatted);
            })
            .catch((err) => console.log(err));

    const handleCompileModel = async () => {
        if (!cytoscapeData || !sessionId) return;
        try {
            const res = await compileOnnxModel(3000, sessionId, cytoscapeData);
            if (res.success) {
                // Clear session storage for the modified graph upon successful compilation
                sessionStorage.removeItem(SAVED_GRAPH_KEY);
                sessionStorage.removeItem(PENDING_RESTORE_KEY);

                if (res.graph) {
                    const formattedGraph = regionsToCompoundNodes(res.graph);
                    setOriginalGraph(formattedGraph);
                    setCytoscapeData(formattedGraph);
                }
                setCompileResult({
                    success: true,
                    message: res.message || "ONNX Model compiled successfully!",
                });
            } else {
                setCompileResult({
                    success: false,
                    message: res.error || "Failed to compile ONNX Model",
                });
            }
        } catch (err) {
            setCompileResult({
                success: false,
                message: err instanceof Error ? err.message : String(err),
            });
        }
        setCompileModalOpen(true);
    };

    const handleToggleEditorMode = (activate: boolean) => {
        if (activate) {
            const hasPending = sessionStorage.getItem(PENDING_RESTORE_KEY) === "true";
            const savedData = sessionStorage.getItem(SAVED_GRAPH_KEY);
            if (hasPending && savedData) {
                setRestoreModalOpen(true);
                return;
            }
        } else {
            // Returning to visualizer: restore original graph while keeping changes in session storage
            if (originalGraph) {
                setCytoscapeData(originalGraph);
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
                sessionStorage.setItem(PENDING_RESTORE_KEY, "true");
            } catch (e) {
                console.error("Error parsing saved graph data:", e);
            }
        }
        setRestoreModalOpen(false);
        setEditorMode(true);
    };

    const handleDiscardChanges = () => {
        sessionStorage.removeItem(SAVED_GRAPH_KEY);
        sessionStorage.removeItem(PENDING_RESTORE_KEY);
        setRestoreModalOpen(false);
        setEditorMode(true);
    };

    const getDescendantNodeIds = (parentIds: Set<string>, allNodes: any[]): Set<string> => {
        const toDelete = new Set<string>(parentIds);
        let added = true;
        while (added) {
            added = false;
            for (const n of allNodes) {
                if (n.data?.parent && toDelete.has(n.data.parent) && !toDelete.has(n.data.id)) {
                    toDelete.add(n.data.id);
                    added = true;
                }
            }
        }
        return toDelete;
    };

    const handleNodeDelete = (nodeId: string) => {
        if (!cytoscapeData) return;
        const deletedNodeIds = getDescendantNodeIds(new Set([nodeId]), cytoscapeData.elements.nodes);
        const updatedNodes = cytoscapeData.elements.nodes.filter((node: any) => !deletedNodeIds.has(node.data.id));
        const updatedEdges = cytoscapeData.elements.edges.filter(
            (edge: any) => !deletedNodeIds.has(edge.data.source) && !deletedNodeIds.has(edge.data.target)
        );
        const newData: CytoscapeData = {
            ...cytoscapeData,
            elements: {
                nodes: updatedNodes,
                edges: updatedEdges,
            },
        };
        if (nodeToEdit && deletedNodeIds.has(nodeToEdit.id)) {
            setNodeToEdit(null);
        }
        setCytoscapeData(newData);
        saveGraphToSession(newData);
    };

    const handleEdgeDelete = (edgeId: string) => {
        if (!cytoscapeData) return;
        const deletedEdge = cytoscapeData.elements.edges.find((e: any) => e.data.id === edgeId);
        const updatedEdges = cytoscapeData.elements.edges.filter((e: any) => e.data.id !== edgeId);

        let updatedNodes = cytoscapeData.elements.nodes;
        if (deletedEdge) {
            const targetId = deletedEdge.data?.target;
            const sourceId = deletedEdge.data?.source;
            if (targetId && sourceId) {
                updatedNodes = updatedNodes.map((node: any) => {
                    if (node.data.id === targetId && node.data.onnxData?.kind === "OperationNode") {
                        const inputs = node.data.onnxData.inputs || [];
                        const nextInputs = inputs.filter((inp: string) => inp !== sourceId);
                        return {
                            ...node,
                            data: {
                                ...node.data,
                                onnxData: {
                                    ...node.data.onnxData,
                                    inputs: nextInputs,
                                },
                            },
                        };
                    }
                    return node;
                });
            }
        }

        const newData: CytoscapeData = {
            ...cytoscapeData,
            elements: {
                nodes: updatedNodes,
                edges: updatedEdges,
            },
        };
        if (selectedEdge && (selectedEdge.id === edgeId || selectedEdge.data?.id === edgeId)) {
            setSelectedEdge(null);
            setEdgePopupPos(null);
        }
        setCytoscapeData(newData);
        saveGraphToSession(newData);
    };

    const handleNodeSubmit = (nodePayload: any, pos: { x: number; y: number } | null) => {
        if (!cytoscapeData) return;

        let newData: CytoscapeData;

        // Extract GRAPH attributes and construct region graphs with copied nodes and edges
        const buildRegionsAndParentMap = (
            opType: string,
            attributes: Record<string, any> | undefined,
            targetLoopId: string,
            existingNodes: any[],
            existingEdges: any[]
        ) => {
            const regions: any[] = [];
            const selectedNodesMap = new Map<string, number>(); // nodeId -> regionIndex

            if (!attributes) return { regions, selectedNodesMap };

            const opDef = StandardOps.find((op) => op.opType === opType);
            let regionIdx = 0;

            const processGraphAttr = (val: any) => {
                const nodeIds: string[] = Array.isArray(val)
                    ? val.map((v: any) => (typeof v === "string" ? v : (v?.value || v?.id || v?.data?.id || String(v))))
                    : typeof val === "string" && val.length > 0
                    ? val.split(",").map((s: string) => s.trim()).filter(Boolean)
                    : (val && typeof val === "object" && Array.isArray((val as any).elements?.nodes))
                    ? (val as any).elements.nodes.map((n: any) => n.data?.id || n.id).filter(Boolean)
                    : [];

                const selectedIdSet = new Set(nodeIds);
                nodeIds.forEach((id) => selectedNodesMap.set(id, regionIdx));

                const regionNodes = selectedIdSet.size > 0
                    ? nodeIds
                          .map((id) => existingNodes.find((n: any) => (n.data?.id || n.id) === id))
                          .filter(Boolean)
                          .map((node: any) => ({
                              ...JSON.parse(JSON.stringify(node)),
                              data: {
                                  ...(node.data || {}),
                                  parent: targetLoopId,
                                  regionIndex: regionIdx,
                              },
                          }))
                    : [];

                const regionEdges = existingEdges
                    .filter((e: any) => selectedIdSet.has(e.data?.source) && selectedIdSet.has(e.data?.target))
                    .map((e: any) => ({
                        ...JSON.parse(JSON.stringify(e)),
                        data: {
                            ...(e.data || {}),
                        },
                    }));

                regions.push({
                    elements: {
                        nodes: regionNodes,
                        edges: regionEdges,
                    },
                });
                regionIdx++;
            };

            if (opDef && opDef.attributes) {
                Object.values(opDef.attributes).forEach((attr) => {
                    if (attr.type === AttributeType.GRAPH || attr.type === AttributeType.GRAPHS) {
                        const val = attributes[attr.name];
                        processGraphAttr(val);
                    }
                });
            } else {
                Object.entries(attributes).forEach(([key, val]) => {
                    if (key === "body" || key === "then_branch" || key === "else_branch") {
                        processGraphAttr(val);
                    }
                });
            }

            return { regions, selectedNodesMap };
        };

        if (nodeToEdit) {
            // --- LÓGICA DE EDIÇÃO ---
            const originalId = nodeToEdit.id;
            const newId = nodePayload.onnxData.id || originalId;
            nodePayload.onnxData.id = newId;

            let updatedRegions: any[] = nodePayload.onnxData.regions || [];
            const selectedNodesMap = new Map<string, number>();

            if (nodePayload.onnxData.kind === "OperationNode") {
                const { regions, selectedNodesMap: sMap } = buildRegionsAndParentMap(
                    nodePayload.onnxData.opType,
                    nodePayload.onnxData.attributes,
                    newId,
                    cytoscapeData.elements.nodes,
                    cytoscapeData.elements.edges
                );
                if (regions.length > 0) {
                    updatedRegions = regions;
                }
                nodePayload.onnxData.regions = updatedRegions;
                sMap.forEach((rIdx, nId) => selectedNodesMap.set(nId, rIdx));
            }

            // 1. Substituir os dados do nó existente e atualizar referências de parent
            const updatedNodes = cytoscapeData.elements.nodes.map((node: any) => {
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
                if (selectedNodesMap.has(node.data.id)) {
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            parent: newId,
                            regionIndex: selectedNodesMap.get(node.data.id),
                        },
                    };
                }
                if (node.data.parent === originalId) {
                    const nextData = { ...node.data };
                    delete nextData.parent;
                    delete nextData.regionIndex;
                    return {
                        ...node,
                        data: nextData,
                    };
                }
                // If a node was renamed (originalId -> newId), update inputs array on OperationNodes referencing it
                if (originalId !== newId && node.data.onnxData?.kind === "OperationNode" && Array.isArray(node.data.onnxData.inputs)) {
                    if (node.data.onnxData.inputs.includes(originalId)) {
                        return {
                            ...node,
                            data: {
                                ...node.data,
                                onnxData: {
                                    ...node.data.onnxData,
                                    inputs: node.data.onnxData.inputs.map((inp: string) => (inp === originalId ? newId : inp)),
                                },
                            },
                        };
                    }
                }
                return node;
            });

            // 2. Atualizar as arestas existentes
            let updatedEdges = cytoscapeData.elements.edges.map((edge: any) => {
                const newEdge: any = { ...edge };
                if (newEdge.data.source === originalId) newEdge.data.source = newId;
                if (newEdge.data.target === originalId) newEdge.data.target = newId;
                return newEdge;
            });

            // Only clear incoming edges for OperationNode (which will be recreated from inputs below)
            if (nodePayload.onnxData.kind === "OperationNode") {
                updatedEdges = updatedEdges.filter((edge: any) => edge.data.target !== newId);
            }

            // 3. Gerar as novas arestas de entrada e nós de saída (se for OperationNode)
            const newEdges = [];
            const newOutputNodes = [];
            const newOutputEdges = [];

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

                // Retrieve schema outputs (or fallback to [{ name: "output" }])
                const schemaOutputs = nodePayload.schemaOutputs && nodePayload.schemaOutputs.length > 0
                    ? nodePayload.schemaOutputs
                    : [{ name: "output" }];

                const existingOutgoingEdges = updatedEdges.filter((e: any) => e.data.source === newId);
                const basePos = nodeToEdit.position || { x: 0, y: 0 };

                if (existingOutgoingEdges.length < schemaOutputs.length) {
                    for (let i = existingOutgoingEdges.length; i < schemaOutputs.length; i++) {
                        const outputDef = schemaOutputs[i];
                        const randomOutputId = `output_${newId}_${outputDef.name || "val"}_${Math.random().toString(36).substr(2, 6)}`;
                        
                        const outputNode = {
                            data: {
                                id: randomOutputId,
                                onnxData: {
                                    id: randomOutputId,
                                    kind: "TensorNode",
                                    tensorType: "intermediate",
                                    literalType: 0,
                                    shape: [],
                                    metadata: {},
                                },
                            },
                            position: {
                                x: basePos.x + 180,
                                y: basePos.y + i * 60,
                            },
                            group: "nodes",
                            removed: false,
                            selected: false,
                            selectable: true,
                            locked: false,
                            grabbable: true,
                            classes: "",
                        };
                        newOutputNodes.push(outputNode);

                        const outputEdge = {
                            data: {
                                id: `${Math.random().toString(36)}`,
                                source: newId,
                                target: randomOutputId,
                            },
                            group: "edges",
                            removed: false,
                            selected: false,
                            selectable: true,
                            locked: false,
                            grabbable: true,
                            classes: "",
                        };
                        newOutputEdges.push(outputEdge);
                    }
                }
            }

            newData = {
                ...cytoscapeData,
                elements: {
                    nodes: [...updatedNodes, ...newOutputNodes],
                    edges: [...updatedEdges, ...newEdges, ...newOutputEdges],
                },
            };

            // Fechar o modo de edição
            setNodeToEdit(null);
            setSidePanelVisibility(false);

        } else {
            // --- LÓGICA DE CRIAÇÃO ---
            const positionToUse = pos || { x: 0, y: 0 };
            const newId =
                nodePayload.label === "" || !nodePayload.label
                    ? (nodePayload.onnxData.id || `node_${Math.random().toString(36).substr(2, 9)}`)
                    : nodePayload.label;
            nodePayload.onnxData.id = newId;

            let constructedRegions: any[] = nodePayload.onnxData.regions || [];
            const selectedNodesMap = new Map<string, number>();

            if (nodePayload.onnxData.kind === "OperationNode") {
                const { regions, selectedNodesMap: sMap } = buildRegionsAndParentMap(
                    nodePayload.onnxData.opType,
                    nodePayload.onnxData.attributes,
                    newId,
                    cytoscapeData.elements.nodes,
                    cytoscapeData.elements.edges
                );
                if (regions.length > 0) {
                    constructedRegions = regions;
                }
                nodePayload.onnxData.regions = constructedRegions;
                sMap.forEach((rIdx, nId) => selectedNodesMap.set(nId, rIdx));
            }

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

            const updatedExistingNodes = cytoscapeData.elements.nodes.map((node: any) => {
                if (selectedNodesMap.has(node.data.id)) {
                    return {
                        ...node,
                        data: {
                            ...node.data,
                            parent: newId,
                            regionIndex: selectedNodesMap.get(node.data.id),
                        },
                    };
                }
                return node;
            });

            const newEdges = [];
            const newOutputNodes = [];
            const newOutputEdges = [];

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

                // Retrieve schema outputs (or fallback to [{ name: "output" }])
                const schemaOutputs = nodePayload.schemaOutputs && nodePayload.schemaOutputs.length > 0
                    ? nodePayload.schemaOutputs
                    : [{ name: "output" }];

                schemaOutputs.forEach((outputDef: any, idx: number) => {
                    const randomOutputId = `output_${newId}_${outputDef.name || "val"}_${Math.random().toString(36).substr(2, 6)}`;
                    
                    const outputNode = {
                        data: {
                            id: randomOutputId,
                            onnxData: {
                                id: randomOutputId,
                                kind: "TensorNode",
                                tensorType: "intermediate",
                                literalType: 0,
                                shape: [],
                                metadata: {},
                            },
                        },
                        position: {
                            x: positionToUse.x + 180,
                            y: positionToUse.y + idx * 60,
                        },
                        group: "nodes",
                        removed: false,
                        selected: false,
                        selectable: true,
                        locked: false,
                        grabbable: true,
                        classes: "",
                    };
                    newOutputNodes.push(outputNode);

                    const outputEdge = {
                        data: {
                            id: `${Math.random().toString(36)}`,
                            source: newId,
                            target: randomOutputId,
                        },
                        group: "edges",
                        removed: false,
                        selected: false,
                        selectable: true,
                        locked: false,
                        grabbable: true,
                        classes: "",
                    };
                    newOutputEdges.push(outputEdge);
                });
            }

            newData = {
                ...cytoscapeData,
                elements: {
                    edges: [...cytoscapeData.elements.edges, ...newEdges, ...newOutputEdges],
                    nodes: [...updatedExistingNodes, newNode, ...newOutputNodes],
                },
            };
            setNewNodePos(null);
            setSidePanelVisibility(false);
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

            <CompileModal
                isOpen={isCompileModalOpen}
                result={compileResult}
                onClose={() => setCompileModalOpen(false)}
            />

            <MenuBar
                style={{
                    gridArea: "menubar",
                    color: "white",
                    padding: "25px",
                }}
                setCytoscapeData={(data) => {
                    if (!editorMode && data) setOriginalGraph(data);
                    setCytoscapeData(data);
                }}
                panelVisibility={{
                    isVisible: isSidePanelVisible,
                    setVisibility: setSidePanelVisibility,
                }}
                setLayout={(l: any) => setCytoscapeLayout(typeof l === "string" ? { name: l } : l)}
                setStylesheet={(sheet: any) => setCytoscapeStylesheet(sheet)}
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
                onCompileOnnxModel={handleCompileModel}
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
                    graphNodes={cytoscapeData?.elements?.nodes || []}
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
                onDeleteEdgeRequested={handleEdgeDelete}
            />
        </main>
    );
}

export default App;
