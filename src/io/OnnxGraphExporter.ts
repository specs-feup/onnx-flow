import OnnxEdge from "../onnx/OnnxEdge.js";
import type OnnxGraph from "../onnx/OnnxGraph.js";
import type {
    ConcreteValueNode,
    RawOnnxAttribute,
    RawOnnxModel,
    RawOnnxNode,
    RawOnnxValueInfo,
    TensorProto,
} from "../onnx/OnnxTypes.js";
import { AttributeType, DataType } from "../onnx/OnnxTypes.js";
import type OperationNode from "../onnx/OperationNode.js";
import TensorNode from "../onnx/TensorNode.js";
import ConstantNode from "../onnx/ConstantNode.js";
import RegionArgumentNode from "../onnx/RegionArgumentNode.js";
import { toU8, topologicalSortOperationNodes } from "../onnx/utils/index.js";
import { OpRegistry } from "../onnx/schema/OpRegistry.js";
import { resolveExportVersions } from "../onnx/schema/Versioning.js";

const ALLOWED_TENSOR_KEYS = [
    "dataType",
    "dims",
    "floatData",
    "int32Data",
    "stringData",
    "int64Data",
    "doubleData",
    "uint64Data",
    "externalData",
] as const;

/**
 * Exports an in-memory `OnnxGraph` (including nested `Loop`/`If`/`Scan` regions)
 * into a protobuf-compatible `RawOnnxModel` JSON object.
 */
export class OnnxGraphExporter {
    /**
     * Ensures every operation input referenced via `opNode.getInputs()` has a
     * corresponding incoming `OnnxEdge` in the graph before export.
     */
    public static prepareForExport(graph: OnnxGraph.Class): void {
        const nodeInputs: { nodeId: string; inputs: string[] }[] = [];

        for (const opNode of graph.getOperationNodes()) {
            const inputs = opNode.getInputs()?.map((n) => n.id);
            if (inputs !== undefined) {
                nodeInputs.push({ nodeId: opNode.id, inputs });
            }
        }

        for (const { nodeId, inputs } of nodeInputs) {
            const opNode = graph.getNodeById(nodeId);
            if (opNode === undefined) continue;

            for (const inputId of inputs) {
                const rawNode = graph.getNodeById(inputId);
                let inputNode: ConcreteValueNode | undefined;
                if (rawNode !== undefined && rawNode.is(TensorNode)) {
                    inputNode = rawNode.as(TensorNode);
                } else if (rawNode !== undefined && rawNode.is(ConstantNode)) {
                    inputNode = rawNode.as(ConstantNode);
                }

                if (inputNode === undefined) continue;
                const alreadyConnected = inputNode.getOutgoers.some(
                    (e) => e.target.id === opNode.id,
                );
                if (!alreadyConnected) {
                    graph
                        .addEdge(inputNode, opNode)
                        .init(new OnnxEdge.Builder(inputNode.literalType, inputNode.shape))
                        .as(OnnxEdge);
                }
            }
        }
    }

    /**
     * Converts `graph` into a `RawOnnxModel` with automatically resolved IR and opset versions.
     */
    public static exportToOnnxJson(
        graph: OnnxGraph.Class,
        name?: string,
        bodyCount: number = 0,
        options?: { targetOpset?: number; irVersion?: number },
    ): RawOnnxModel {
        const resolvedVersions =
            bodyCount === 0
                ? resolveExportVersions(graph, options)
                : { irVersion: graph.getIrVersion(), opsetVersion: graph.getTargetOpset() };
        const effectiveOpset = options?.targetOpset ?? resolvedVersions.opsetVersion;

        const modelInitializers = graph
            .getConstantNodes()
            .toArray()
            .map((node) =>
                OnnxGraphExporter.sanitizeTensor({ ...node.constantValue, name: node.id }),
            );

        const modelInputs: RawOnnxValueInfo[] = [
            ...graph
                .getInputTensorNodes()
                .toArray()
                .map((node) => OnnxGraphExporter.buildValueInfo(node, true)),
            ...graph
                .getConstantNodes()
                .toArray()
                .filter((node) => node.isInput)
                .map((node) => OnnxGraphExporter.buildValueInfo(node, false)),
        ];

        const modelOutputs: RawOnnxValueInfo[] = graph
            .getOutputTensorNodes()
            .toArray()
            .map((node) => OnnxGraphExporter.buildValueInfo(node, true));

        const opNodes = topologicalSortOperationNodes(graph);
        const modelNodes: RawOnnxNode[] = opNodes.map((opNode) =>
            OnnxGraphExporter.serializeOperationNode(
                opNode,
                bodyCount,
                effectiveOpset,
                resolvedVersions.irVersion,
            ),
        );

        return {
            irVersion: resolvedVersions.irVersion,
            opsetImport: [{ domain: "", version: effectiveOpset }],
            graph: {
                name: name ?? "Graph",
                initializer: modelInitializers,
                node: modelNodes,
                input: modelInputs,
                output: modelOutputs,
            },
        };
    }

    private static buildValueInfo(
        node: ConcreteValueNode,
        allowUnknownDims: boolean,
    ): RawOnnxValueInfo {
        return {
            name: node.id,
            type: {
                tensorType: {
                    elemType: node.literalType,
                    shape: {
                        dim: node.shape.map((d) => {
                            if (typeof d === "string") return { dimParam: d };
                            if (d == null) return {};
                            if (allowUnknownDims && d < 0) return {};
                            return { dimValue: d };
                        }),
                    },
                },
            },
        };
    }

    private static sanitizeTensor(tensor: Partial<TensorProto>): TensorProto {
        const sanitized: Record<string, unknown> = { name: tensor.name };

        for (const key of ALLOWED_TENSOR_KEYS) {
            const value = (tensor as Record<string, unknown>)[key];
            if (value === undefined || value === null) continue;

            if (Array.isArray(value)) {
                sanitized[key] = value.map((v) =>
                    typeof v === "bigint" || typeof v === "string" ? Number(v) : v,
                );
            } else if (
                key.endsWith("Data") &&
                tensor.rawData !== undefined &&
                ((value as { length?: number }).length ?? 0) === 0
            ) {
                OnnxGraphExporter.decodeFallbackRawData(tensor, sanitized);
            } else {
                sanitized[key] = value;
            }
        }

        const rawBytes = toU8(tensor.rawData);
        if (rawBytes !== undefined && rawBytes.length > 0) {
            sanitized["rawData"] = {
                type: "Buffer",
                data: Array.from(rawBytes),
            };
        }

        return sanitized as TensorProto;
    }

    private static decodeFallbackRawData(
        tensor: Partial<TensorProto>,
        sanitized: Record<string, unknown>,
    ): void {
        const u8 = toU8(tensor.rawData);
        if (u8 === undefined) return;
        const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
        const dtype = tensor.dataType ?? DataType.INT64;

        if (dtype === DataType.INT64) {
            const arr: bigint[] = [];
            for (let i = 0; i + 8 <= dv.byteLength; i += 8) {
                arr.push(dv.getBigInt64(i, true));
            }
            sanitized["int64Data"] = arr;
        } else if (dtype === DataType.INT32) {
            const arr: number[] = [];
            for (let i = 0; i + 4 <= dv.byteLength; i += 4) {
                arr.push(dv.getInt32(i, true));
            }
            sanitized["int32Data"] = arr;
        } else if (dtype === DataType.FLOAT) {
            const arr: number[] = [];
            for (let i = 0; i + 4 <= dv.byteLength; i += 4) {
                arr.push(dv.getFloat32(i, true));
            }
            sanitized["floatData"] = arr;
        }
    }

    private static serializeOperationNode(
        opNode: OperationNode.Class,
        bodyCount: number,
        effectiveOpset: number,
        irVersion: number,
    ): RawOnnxNode {
        const opType = opNode.type;
        const inputs =
            opNode.getInputs()?.map((n) => {
                if (n.is(RegionArgumentNode)) {
                    return n.as(RegionArgumentNode).originalName;
                }
                return n.id;
            }) ?? [];
        const outputs = opNode.getOutgoers.targets.toArray().map((n) => n.id);

        const baseAttrs = OnnxGraphExporter.serializeAttributes(opNode, effectiveOpset);
        baseAttrs.push(
            ...OnnxGraphExporter.serializeRegions(opNode, bodyCount, effectiveOpset, irVersion),
        );

        return {
            name: opNode.id,
            opType,
            input: inputs,
            output: outputs,
            attribute: baseAttrs,
        };
    }

    private static serializeAttributes(
        opNode: OperationNode.Class,
        effectiveOpset: number,
    ): RawOnnxAttribute[] {
        const attrs: RawOnnxAttribute[] = [];
        const schema = OpRegistry.getInstance().get(opNode.type, effectiveOpset);

        for (const [name, value] of Object.entries(opNode.attributes)) {
            if (typeof value === "object" && ("g" in value || "type" in value)) {
                const valObj = value as Record<string, unknown>;
                attrs.push({
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
                if (expectedType === AttributeType.FLOATS) {
                    attr.floats = value as number[];
                    attr.type = AttributeType.FLOATS;
                } else if (expectedType === AttributeType.STRINGS || typeof value[0] === "string") {
                    attr.strings = (value as string[]).map((s) =>
                        Buffer.from(s, "utf8").toString("base64"),
                    );
                    attr.type = AttributeType.STRINGS;
                } else {
                    attr.ints = value as (number | string)[];
                    attr.type = AttributeType.INTS;
                }
            } else if (typeof value === "number") {
                if (
                    expectedType === AttributeType.FLOAT ||
                    (expectedType === undefined && !Number.isInteger(value))
                ) {
                    attr.f = value;
                    attr.type = AttributeType.FLOAT;
                } else {
                    attr.i = value;
                    attr.type = AttributeType.INT;
                }
            } else if (typeof value === "string") {
                attr.s = Buffer.from(value, "utf8").toString("base64");
                attr.type = AttributeType.STRING;
            } else if (
                typeof value === "object" &&
                "type" in value &&
                (value as Record<string, unknown>)["type"] === "TENSOR"
            ) {
                attr.t = OnnxGraphExporter.sanitizeTensor(value as Partial<TensorProto>);
                attr.type = AttributeType.TENSOR;
            }

            if (attr.type !== undefined) attrs.push(attr);
        }

        return attrs;
    }

    private static serializeRegions(
        opNode: OperationNode.Class,
        bodyCount: number,
        effectiveOpset: number,
        irVersion: number,
    ): RawOnnxAttribute[] {
        const exportSubgraph = (region: OnnxGraph.Class, label: string, attrName: string) => ({
            name: attrName,
            type: AttributeType.GRAPH,
            g: OnnxGraphExporter.exportToOnnxJson(region, label, bodyCount + 1, {
                targetOpset: effectiveOpset,
                irVersion,
            }).graph!,
        });

        if (opNode.type === "Loop") {
            return [exportSubgraph(opNode.regions[0], `loop_body_${bodyCount}`, "body")];
        }
        if (opNode.type === "If") {
            return [
                exportSubgraph(opNode.regions[0], `then_${bodyCount}`, "then_branch"),
                exportSubgraph(opNode.regions[1], `else_${bodyCount}`, "else_branch"),
            ];
        }
        if (opNode.type === "Scan") {
            return [exportSubgraph(opNode.regions[0], `scan_${bodyCount}`, "body")];
        }
        return [];
    }
}

/** Restores any missing incoming input edges on operations prior to graph export. */
export function prepareGraphForExport(graph: OnnxGraph.Class): void {
    OnnxGraphExporter.prepareForExport(graph);
}

/** Converts an `OnnxGraph` into a protobuf-compatible `RawOnnxModel` JSON object. */
export function convertFlowGraphToOnnxJson(
    graph: OnnxGraph.Class,
    name?: string,
    bodyCount: number = 0,
    options?: { targetOpset?: number; irVersion?: number },
): RawOnnxModel {
    return OnnxGraphExporter.exportToOnnxJson(graph, name, bodyCount, options);
}
