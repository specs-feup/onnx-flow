import OnnxEdge from "./Onnx/OnnxEdge.js";
import type OnnxGraph from "./Onnx/OnnxGraph.js";
import type {
    ConcreteValueNode,
    RawOnnxAttribute,
    RawOnnxModel,
    RawOnnxNode,
    RawOnnxValueInfo,
    TensorProto,
} from "./Onnx/OnnxTypes.js";
import { AttributeType, DataType } from "./Onnx/OnnxTypes.js";
import TensorNode from "./Onnx/TensorNode.js";
import ConstantNode from "./Onnx/ConstantNode.js";
import { asOnnxNode, isOnnxNode, topologicalSortOperationNodes } from "./Onnx/Utils.js";
import RegionArgumentNode from "./Onnx/RegionArgumentNode.js";
import { OpRegistry } from "./Onnx/Schema/OpRegistry.js";
import type { EdgeSnapshot, NodeSnapshot } from "./Onnx/transformation/tracking/GraphActions.js";

export type UnifiedNodeData =
    | Exclude<NodeSnapshot, { kind: "OperationNode" }> // Keeps Tensor, Constant, RegionArgument as-is
    | {
          kind: "OperationNode";
          id: string;
          opType: string;
          attributes: Record<string, unknown>;
          inputs: string[];
          regions: UnifiedExplorerJson[];
          metadata?: Record<string, unknown>;
      };

export interface UnifiedNode {
    data: {
        id: string;
        onnxData?: UnifiedNodeData;
        [key: string]: unknown; // Allows Cytoscape's visual properties (label, color, etc.)
    };
    position?: { x: number; y: number };
    classes?: string;
}

export interface UnifiedEdge {
    data: {
        id: string;
        source: string;
        target: string;

        // Explicit boolean so the frontend never has to check for undefined
        isCrossGraph: boolean;

        // The ID of the parent/boundary node (e.g., the Loop node itself)
        // Optional, because it only exists if isCrossGraph is true
        innerTarget?: string;

        onnxData?: EdgeSnapshot;
        [key: string]: unknown;
    };
    classes?: string;
}

export interface UnifiedExplorerJson {
    elements: {
        nodes: UnifiedNode[];
        edges: UnifiedEdge[];
    };
    [key: string]: unknown; // Allows other Cytoscape top-level properties like zooming
}

interface RawCytoscapeExport {
    elements?: {
        nodes?: UnifiedNode[];
        edges?: UnifiedEdge[];
    };
    [key: string]: unknown;
}

const IR_VERSION = 9;
const OPSET_IMPORT = 19;

export function prepareGraphForExport(graph: OnnxGraph.Class): void {
    const mapNodeAndInputs: { nodeId: string; inputs: string[] }[] = [];
    const mapNodeAndOutput: { nodeId: string; output: string }[] = [];

    for (const opNode of graph.getOperationNodes()) {
        const inputs = opNode.getInputs()?.map((n) => n.id);
        if (inputs) {
            mapNodeAndInputs.push({ nodeId: opNode.id, inputs });
        }

        opNode.getOutgoers.targets.forEach((target) => {
            if (target.is(TensorNode) || target.is(ConstantNode)) {
                mapNodeAndOutput.push({ nodeId: opNode.id, output: target.id });
            }
        });
    }

    // Add missing edges
    mapNodeAndInputs.forEach(({ nodeId, inputs }) => {
        const opNode = graph.getNodeById(nodeId);
        for (const inputId of inputs) {
            const rawNode = graph.getNodeById(inputId);

            // Phase 3 Support: Input can be TensorNode OR ConstantNode
            let inputNode: ConcreteValueNode | undefined = undefined;
            if (rawNode !== undefined && rawNode.is(TensorNode)) inputNode = rawNode.as(TensorNode);
            else if (rawNode !== undefined && rawNode.is(ConstantNode))
                inputNode = rawNode.as(ConstantNode);

            if (inputNode) {
                const alreadyConnected = inputNode.getOutgoers.some(
                    (e) => e.target.id === opNode?.id,
                );
                if (!alreadyConnected && opNode) {
                    const type = inputNode.literalType;
                    const shape = inputNode.shape;
                    graph
                        .addEdge(inputNode, opNode)
                        .init(new OnnxEdge.Builder(type, shape))
                        .as(OnnxEdge);
                }
            }
        }
    });
}

export function convertFlowGraphToOnnxJson(
    graph: OnnxGraph.Class,
    name?: string,
    bodyCount: number = 0,
): RawOnnxModel {
    const modelInputs: RawOnnxValueInfo[] = [];
    const modelOutputs: RawOnnxValueInfo[] = [];

    // 1. Convert Initializers (Constants)
    const modelInitializers = convertInitializers(graph);

    const modelNodes: RawOnnxNode[] = [];

    function sanitizeTensor(tensor: Partial<TensorProto>): TensorProto {
        // TensorProto keys except name and rawData (handled differently)
        const allowedKeys = [
            "dataType",
            "dims",
            "floatData",
            "int32Data",
            "stringData",
            "int64Data",
            "doubleData",
            "uint64Data",
            "externalData",
        ];

        const sanitized: Record<string, unknown> = { name: tensor.name };

        for (const key of allowedKeys) {
            const value = (tensor as Record<string, unknown>)[key];
            if (value !== undefined && value !== null) {
                if (Array.isArray(value)) {
                    sanitized[key] = value.map((v) => {
                        if (typeof v === "bigint") return Number(v);
                        if (typeof v === "string") return Number(v);
                        return v;
                    });
                } else if (
                    key.endsWith("Data") &&
                    tensor.rawData !== undefined &&
                    ((value as { length?: number }).length ?? 0) === 0 // only override if missing
                ) {
                    // Try decoding rawData if value array is empty
                    const dtype = tensor.dataType ?? DataType.INT64;
                    const buffer = Buffer.isBuffer(tensor.rawData.data)
                        ? tensor.rawData.data
                        : Buffer.from(tensor.rawData.data as number[]);
                    if (dtype === DataType.INT64) {
                        const arr: bigint[] = [];
                        for (let i = 0; i < buffer.length; i += 8) {
                            arr.push(buffer.readBigInt64LE(i));
                        }
                        sanitized["int64Data"] = arr;
                    } else if (dtype === DataType.INT32) {
                        const arr: number[] = [];
                        for (let i = 0; i < buffer.length; i += 4) {
                            arr.push(buffer.readInt32LE(i));
                        }
                        sanitized["int32Data"] = arr;
                    } else if (dtype === DataType.FLOAT) {
                        const arr: number[] = [];
                        for (let i = 0; i < buffer.length; i += 4) {
                            arr.push(buffer.readFloatLE(i));
                        }
                        sanitized["floatData"] = arr;
                    }
                } else {
                    sanitized[key] = value;
                }
            }
        }

        // Special handling for rawData
        if (tensor.rawData && Buffer.isBuffer(tensor.rawData) && tensor.rawData.length > 0) {
            sanitized["rawData"] = {
                type: "Buffer",
                data: Array.from(tensor.rawData),
            };
        }

        return sanitized as TensorProto;
    }

    function convertInitializers(graph: OnnxGraph.Class): TensorProto[] {
        const initializers: TensorProto[] = [];
        for (const node of graph.getConstantNodes()) {
            const original = node.constantValue;
            const serialized = sanitizeTensor({ ...original, name: node.id });
            initializers.push(serialized);
        }
        return initializers;
    }

    // 2. Export Inputs
    for (const node of graph.getInputTensorNodes()) {
        modelInputs.push({
            name: node.id,
            type: {
                tensorType: {
                    elemType: node.literalType,
                    shape: {
                        dim: node.shape.map((d) => {
                            if (typeof d === "string") return { dimParam: d };
                            return d == null ? {} : { dimValue: d };
                        }),
                    },
                },
            },
        });
    }

    // 2b. Constant inputs
    for (const node of graph.getConstantNodes()) {
        if (node.isInput) {
            modelInputs.push({
                name: node.id,
                type: {
                    tensorType: {
                        elemType: node.literalType,
                        shape: {
                            dim: node.shape.map((d) =>
                                typeof d === "string" ? { dimParam: d } : { dimValue: d },
                            ),
                        },
                    },
                },
            });
        }
    }

    // 3. Export Outputs
    for (const node of graph.getOutputTensorNodes()) {
        modelOutputs.push({
            name: node.id,
            type: {
                tensorType: {
                    elemType: node.literalType,
                    shape: {
                        dim: node.shape.map((d) => {
                            if (typeof d === "string") return { dimParam: d };
                            return d == null ? {} : { dimValue: d };
                        }),
                    },
                },
            },
        });
    }

    // 4. Nodes
    const opNodes = topologicalSortOperationNodes(graph);

    for (const opNode of opNodes) {
        const opType = opNode.type;

        // Handle Inputs: Resolve RegionArgumentNode back to original name
        const inputs =
            opNode.getInputs()?.map((n) => {
                if (n.is(RegionArgumentNode)) {
                    return n.as(RegionArgumentNode).originalName; // Restore implicit link
                }
                return n.id;
            }) ?? [];

        const outputs = opNode.getOutgoers.targets.toArray().map((n) => n.id);

        const baseAttrs: RawOnnxAttribute[] = [];

        // Get the schema for this operation to correctly type numerical attributes
        const schema = OpRegistry.getInstance().get(opType, OPSET_IMPORT);

        // Serialize attributes
        for (const [name, value] of Object.entries(opNode.attributes)) {
            // Skip handled regions logic
            if (typeof value === "object" && ("g" in value || "type" in value)) {
                const valObj = value as Record<string, unknown>;
                baseAttrs.push({
                    name,
                    type: (valObj["type"] as number | string | undefined) ?? AttributeType.GRAPH,
                    ...valObj,
                } as unknown as RawOnnxAttribute);
                continue;
            }

            const attrDef = schema?.attributes === undefined ? undefined : schema.attributes[name];
            const expectedType = attrDef?.type;

            const attr: RawOnnxAttribute = { name };
            if (Array.isArray(value)) {
                // Determine if it should be FLOATS or INTS
                if (expectedType === AttributeType.FLOATS) {
                    attr.floats = value as number[];
                    attr.type = AttributeType.FLOATS;
                } else {
                    attr.ints = value as (number | string)[];
                    attr.type = AttributeType.INTS;
                }
            } else if (typeof value === "number") {
                // Distinguish between FLOAT and INT based on the schema
                if (expectedType === AttributeType.FLOAT) {
                    attr.f = value;
                    attr.type = AttributeType.FLOAT;
                } else {
                    attr.i = value;
                    attr.type = AttributeType.INT;
                }
            } else if (typeof value === "string") {
                attr.s = value;
                attr.type = AttributeType.STRING;
            } else if (
                typeof value === "object" &&
                "type" in value &&
                (value as Record<string, unknown>)["type"] === "TENSOR"
            ) {
                attr.t = sanitizeTensor(value as Partial<TensorProto>);
                attr.type = AttributeType.TENSOR;
            }

            if (attr.type !== undefined) baseAttrs.push(attr);
        }

        // Handle Regions (Subgraphs)
        // We use the 'regions' array on the OpNode and map back to standard ONNX attr names.

        if (opType === "Loop") {
            const bodyGraph = opNode.regions[0]; // Convention: Loop body is region 0
            const bodyJson = convertFlowGraphToOnnxJson(
                bodyGraph,
                `loop_body_${bodyCount}`,
                bodyCount + 1,
            ).graph!;
            baseAttrs.push({
                name: "body",
                type: AttributeType.GRAPH,
                g: bodyJson,
            });
        } else if (opType === "If") {
            const thenGraph = opNode.regions[0];
            const elseGraph = opNode.regions[1];

            //thenGraph
            const thenJson = convertFlowGraphToOnnxJson(
                thenGraph,
                `then_${bodyCount}`,
                bodyCount + 1,
            ).graph!;
            baseAttrs.push({ name: "then_branch", type: AttributeType.GRAPH, g: thenJson });
            //elseGraph
            const elseJson = convertFlowGraphToOnnxJson(
                elseGraph,
                `else_${bodyCount}`,
                bodyCount + 1,
            ).graph!;
            baseAttrs.push({ name: "else_branch", type: AttributeType.GRAPH, g: elseJson });
        } else if (opType === "Scan") {
            const bodyGraph = opNode.regions[0];
            const gJson = convertFlowGraphToOnnxJson(
                bodyGraph,
                `scan_${bodyCount}`,
                bodyCount + 1,
            ).graph!;
            baseAttrs.push({ name: "body", type: AttributeType.GRAPH, g: gJson });
        }

        modelNodes.push({
            opType,
            input: inputs,
            output: outputs,
            attribute: baseAttrs,
        });
    }

    return {
        irVersion: IR_VERSION,
        opsetImport: [{ version: OPSET_IMPORT }],
        graph: {
            name: name ?? "Graph",
            initializer: modelInitializers,
            node: modelNodes,
            input: modelInputs,
            output: modelOutputs,
        },
    };
}

/**
 * Generates a unified JSON payload that Cytoscape can render visually,
 * but contains the exact ONNX-Flow data needed to reconstruct the graph.
 */
export function generateUnifiedExplorerJson(
    graph: OnnxGraph.Class,
    includeCrossGraphEdges: boolean = true,
    parentScopeNodeIds: Set<string> = new Set(),
    parentOperationId?: string,
    parentJson?: UnifiedExplorerJson,
): UnifiedExplorerJson {
    // 1. Get the raw Cytoscape JSON
    const rawCy = graph.toCy().json() as unknown as RawCytoscapeExport;

    if (rawCy.elements === undefined) {
        rawCy.elements = { nodes: [], edges: [] };
    } else {
        if (rawCy.elements.nodes === undefined) {
            rawCy.elements.nodes = [];
        }
        if (rawCy.elements.edges === undefined) {
            rawCy.elements.edges = [];
        }
    }

    const cyJson = rawCy as UnifiedExplorerJson;

    // 2. Accumulate all node IDs from the current graph to pass to its children
    const currentScopeNodeIds = new Set(parentScopeNodeIds);
    graph.getNodes().forEach((n) => currentScopeNodeIds.add(n.id));

    // 3. Enrich Nodes
    if (cyJson.elements.nodes.length > 0) {
        cyJson.elements.nodes.forEach((cyNode: UnifiedNode) => {
            const actualNode = graph.getNodeById(cyNode.data.id);
            if (actualNode && isOnnxNode(actualNode)) {
                const snap = asOnnxNode(actualNode).toSnapshot();
                // If it's an OperationNode, we must recursively serialize its subgraphs!
                if (snap.kind === "OperationNode") {
                    cyNode.data.onnxData = {
                        ...snap,
                        // Recursively convert OnnxGraph.Class[] -> UnifiedExplorerJson[]
                        regions: snap.regions.map((regionGraph) =>
                            generateUnifiedExplorerJson(
                                regionGraph,
                                includeCrossGraphEdges,
                                currentScopeNodeIds,
                                cyNode.data.id,
                                cyJson,
                            ),
                        ),
                    };
                } else {
                    cyNode.data.onnxData = snap;
                }
            }

            for (const key of Object.keys(cyNode.data)) {
                if (key.startsWith("__specs-onnx__")) {
                    delete cyNode.data[key];
                }
            }
        });
    }

    // 4. Enrich Edges
    if (cyJson.elements.edges.length > 0) {
        cyJson.elements.edges.forEach((cyEdge: UnifiedEdge) => {
            cyEdge.data.isCrossGraph = false;
            const actualEdge = graph.getOnnxEdgeById(cyEdge.data.id);
            if (actualEdge != undefined) {
                cyEdge.data.onnxData = actualEdge.toSnapshot();
            }

            for (const key of Object.keys(cyEdge.data)) {
                if (key.startsWith("__specs-onnx__")) {
                    delete cyEdge.data[key];
                }
            }
        });
    }

    // 5. Inject Cross-Graph Edges (Visualization Only)
    if (includeCrossGraphEdges) {
        graph.getOperationNodes().forEach((opNode) => {
            const inputs = opNode.getInputs() ?? [];

            inputs.forEach((input) => {
                // Determine if this input is a formal graph boundary input
                // (e.g., iter, cond_in, carry)
                const isFormalLocalInput =
                    input.is(TensorNode) && input.as(TensorNode).type === "input";

                // If it exists in the parent graph, isn't generated locally, and isn't a formal input,
                // it is an implicit proxy capture.
                if (
                    parentScopeNodeIds.has(input.id) &&
                    input.incomers.length === 0 &&
                    !isFormalLocalInput
                ) {
                    const syntheticEdge: UnifiedEdge = {
                        data: {
                            id: `cross_edge_${input.id}_to_${opNode.id}`,
                            source: input.id, // The outer tensor ID
                            target: parentOperationId!,
                            isCrossGraph: true,
                            innerTarget: opNode.id, // The inner operation
                        },
                        classes: "cross-graph-capture",
                    };

                    parentJson?.elements.edges.push(syntheticEdge);
                }
            });
        });
    }

    return cyJson;
}
