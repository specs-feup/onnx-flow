import OnnxEdge from "./Onnx/OnnxEdge.js";
import OnnxGraph from "./Onnx/OnnxGraph.js";
import { AttributeProto, AttributeType, DataType } from "./Onnx/OnnxTypes.js";
import TensorNode from "./Onnx/TensorNode.js";
import ConstantNode from "./Onnx/ConstantNode.js";
import { topologicalSortOperationNodes } from "./Onnx/Utils.js";
import RegionArgumentNode from "./Onnx/RegionArgumentNode.js";

const IR_VERSION = 9;
const OPSET_IMPORT = 19;

export function prepareGraphForExport(graph: OnnxGraph.Class): void {
    const mapNodeAndInputs: { nodeId: string; inputs: string[] }[] = [];
    const mapNodeAndOutput: { nodeId: string; output: string }[] = [];

    for (const opNode of graph.getOperationNodes()) {
        const inputs = opNode.getInputs().map((n) => n.id);
        mapNodeAndInputs.push({ nodeId: opNode.id, inputs });

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
            let inputNode: TensorNode.Class | ConstantNode.Class | undefined = undefined;
            if (rawNode?.is(TensorNode)) inputNode = rawNode.as(TensorNode);
            else if (rawNode?.is(ConstantNode)) inputNode = rawNode.as(ConstantNode);

            if (inputNode) {
                const alreadyConnected = inputNode.getOutgoers?.some(
                    (e) => e.target.id === opNode.id,
                );
                if (!alreadyConnected) {
                    const type = inputNode.literalType ?? AttributeType.UNDEFINED;
                    const shape = inputNode.shape ?? [];
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
): any {
    const modelInputs: any[] = [];
    const modelOutputs: any[] = [];

    // 1. Convert Initializers (Constants)
    const modelInitializers = convertInitializers(graph);

    const modelNodes: any[] = [];

    function sanitizeTensor(tensor: any): any {
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

        const sanitized: any = { name: tensor.name };

        for (const key of allowedKeys) {
            const value = tensor[key];
            if (value !== undefined && value !== null) {
                if (Array.isArray(value)) {
                    sanitized[key] = value.map((v) => (typeof v === "string" ? Number(v) : v));
                } else if (
                    key.endsWith("Data") &&
                    tensor.rawData &&
                    !value.length // only override if missing
                ) {
                    // Try decoding rawData if value array is empty
                    const dtype = tensor.dataType ?? DataType.INT64;
                    const buffer = Buffer.from(tensor.rawData.data);
                    if (dtype === DataType.INT64) {
                        sanitized.int64Data = [];
                        for (let i = 0; i < buffer.length; i += 8) {
                            sanitized.int64Data.push(buffer.readBigInt64LE(i));
                        }
                    } else if (dtype === DataType.INT32) {
                        sanitized.int32Data = [];
                        for (let i = 0; i < buffer.length; i += 4) {
                            sanitized.int32Data.push(buffer.readInt32LE(i));
                        }
                    } else if (dtype === DataType.FLOAT) {
                        sanitized.floatData = [];
                        for (let i = 0; i < buffer.length; i += 4) {
                            sanitized.floatData.push(buffer.readFloatLE(i));
                        }
                    }
                } else {
                    sanitized[key] = value;
                }
            }
        }

        // Special handling for rawData
        if (tensor.rawData && Buffer.isBuffer(tensor.rawData) && tensor.rawData.length > 0) {
            sanitized.rawData = {
                type: "Buffer",
                data: Array.from(tensor.rawData),
            };
        }

        return sanitized;
    }

    function convertInitializers(graph: OnnxGraph.Class): any[] {
        const initializers: any[] = [];
        for (const node of graph.getConstantNodes()) {
            const original = node.constantValue;
            if (!original) continue;
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
        const opType = opNode.type ?? "UnknownOp";

        // Handle Inputs: Resolve RegionArgumentNode back to original name
        const inputs =
            opNode.getInputs()?.map((n) => {
                if (n.is(RegionArgumentNode)) {
                    return n.as(RegionArgumentNode).originalName; // Restore implicit link
                }
                return n.id;
            }) ?? [];

        const outputs = opNode.getOutgoers.targets.toArray().map((n) => n.id);

        const baseAttrs: AttributeProto[] = [];

        // Serialize attributes
        for (const [name, value] of Object.entries(opNode.attributes || {})) {
            // Skip handled regions logic
            if (value && typeof value === "object" && ("g" in value || "type" in value)) {
                baseAttrs.push({ name, ...value });
                continue;
            }
            // ... (Your standard attribute serialization logic) ...
            // e.g. ints, i, s, t, etc.
            // (Copying your previous logic simplified for brevity)
            const attr: any = { name };
            if (Array.isArray(value)) {
                attr.ints = value;
                attr.type = AttributeType.INTS;
            } else if (typeof value === "number") {
                attr.i = value;
                attr.type = AttributeType.INT;
            } else if (typeof value === "string") {
                attr.s = value;
                attr.type = AttributeType.STRING;
            } else if (value && (value as any).type === "TENSOR") {
                attr.t = sanitizeTensor(value);
                attr.type = AttributeType.TENSOR;
            }

            if (attr.type !== undefined) baseAttrs.push(attr);
        }

        // Handle Regions (Subgraphs)
        // We use the 'regions' array on the OpNode and map back to standard ONNX attr names.

        if (opType === "Loop") {
            const bodyGraph = opNode.regions[0]; // Convention: Loop body is region 0
            if (bodyGraph) {
                const bodyJson = convertFlowGraphToOnnxJson(
                    bodyGraph,
                    `loop_body_${bodyCount}`,
                    bodyCount + 1,
                ).graph;
                baseAttrs.push({
                    name: "body",
                    type: AttributeType.GRAPH,
                    g: bodyJson,
                });
            }
        } else if (opType === "If") {
            const thenGraph = opNode.regions[0];
            const elseGraph = opNode.regions[1];

            if (thenGraph) {
                const gJson = convertFlowGraphToOnnxJson(
                    thenGraph,
                    `then_${bodyCount}`,
                    bodyCount + 1,
                ).graph;
                baseAttrs.push({ name: "then_branch", type: AttributeType.GRAPH, g: gJson });
            }
            if (elseGraph) {
                const gJson = convertFlowGraphToOnnxJson(
                    elseGraph,
                    `else_${bodyCount}`,
                    bodyCount + 1,
                ).graph;
                baseAttrs.push({ name: "else_branch", type: AttributeType.GRAPH, g: gJson });
            }
        } else if (opType === "Scan") {
            const bodyGraph = opNode.regions[0];
            if (bodyGraph) {
                const gJson = convertFlowGraphToOnnxJson(
                    bodyGraph,
                    `scan_${bodyCount}`,
                    bodyCount + 1,
                ).graph;
                baseAttrs.push({ name: "body", type: AttributeType.GRAPH, g: gJson });
            }
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
