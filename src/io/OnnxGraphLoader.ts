import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../onnx/OnnxGraph.js";
import TensorNode from "../onnx/TensorNode.js";
import ConstantNode from "../onnx/ConstantNode.js";
import RegionArgumentNode from "../onnx/RegionArgumentNode.js";
import type {
    AttributeMap,
    KnownDim,
    KnownShape,
    OpsetImport,
    RawOnnxAttribute,
    RawOnnxDim,
    RawOnnxModel,
    RawOnnxNode,
    RawOnnxTensorType,
    RawOnnxValueInfo,
    Shape,
    TensorProto,
    ValueNode,
} from "../onnx/OnnxTypes.js";
import { AttributeType, DataType } from "../onnx/OnnxTypes.js";
import { LATEST_ONNX_OPSET } from "../onnx/utils/Constants.js";
import inferShapes from "../onnx/InferShapes.js";
import { applyAdapters } from "../onnx/frontend/Adapters.js";
import { tryAsValueNode, uniq, UNKNOWN_SHAPE } from "../onnx/utils/index.js";
import { GraphBuilder } from "../onnx/GraphBuilder.js";

interface GraphVersionMeta {
    irVersion?: number;
    opsetImport?: OpsetImport[];
    sourceOpset?: number;
    targetOpset?: number;
}

const REGION_ATTR_NAMES = new Set(["body", "then_branch", "else_branch"]);

/**
 * Ingests a decoded `RawOnnxModel` (after running frontend opset adapters) into
 * an in-memory `OnnxGraph`, including recursive control-flow subgraphs and
 * implicit variable captures via `RegionArgumentNode`.
 */
export class OnnxGraphLoader {
    /**
     * Runs frontend opset adapters on `data` and builds a shape-inferred `OnnxGraph`.
     */
    public static load(data: RawOnnxModel, targetOpset?: number): OnnxGraph.Class {
        const rawIr = data.irVersion ?? data.ir_version;
        const irVersion = rawIr !== undefined ? Number(rawIr) : undefined;
        const rawOpsets = data.opsetImport ?? data.opset_import;
        const opsetImport: OpsetImport[] | undefined = rawOpsets?.map((o) => ({
            domain: o.domain ?? "",
            version: Number(o.version ?? LATEST_ONNX_OPSET),
        }));
        const sourceOpset = opsetImport?.find(
            (o) => o.domain === "" || o.domain === "ai.onnx",
        )?.version;

        applyAdapters(data);

        const versionMeta: GraphVersionMeta = {
            ...(irVersion !== undefined ? { irVersion } : {}),
            ...(opsetImport !== undefined ? { opsetImport } : {}),
            ...(sourceOpset !== undefined ? { sourceOpset } : {}),
            ...(targetOpset !== undefined ? { targetOpset } : {}),
        };

        return OnnxGraphLoader.createGraphWithCaptures(data, undefined, versionMeta).graph;
    }

    private static parseShape(shape: RawOnnxTensorType["shape"]): KnownShape {
        if (!shape?.dim) return [];
        return shape.dim.map((dim: RawOnnxDim): KnownDim => {
            const dimParam = dim.dimParam ?? dim.dim_param;
            const dimValue = dim.dimValue ?? dim.dim_value;

            if (typeof dimParam === "string" && dimParam !== "") {
                return dimParam;
            }
            if (dimValue !== undefined) {
                return Number(dimValue);
            }
            return UNKNOWN_SHAPE[0];
        });
    }

    private static addTensorFromValueInfo(
        graph: OnnxGraph.Class,
        vi: RawOnnxValueInfo,
        kind: TensorNode.TensorKind,
    ): void {
        if (graph.hasNode(vi.name)) return;
        const tensorType = vi.type?.tensorType ?? vi.type?.tensor_type;
        const shape = OnnxGraphLoader.parseShape(tensorType?.shape);
        const elemType = Number(tensorType?.elemType ?? 0);
        graph
            .addNode(vi.name)
            .init(new TensorNode.Builder(elemType, shape, kind))
            .as(TensorNode);
    }

    private static addInitializers(data: RawOnnxModel, graph: OnnxGraph.Class): void {
        for (const tensor of data.graph?.initializer ?? []) {
            if (tensor.name === undefined || !graph.hasNode(tensor.name)) {
                graph.addNode(tensor.name).init(new ConstantNode.Builder(tensor)).as(ConstantNode);
            }
        }
    }

    private static addValueInfoNodes(data: RawOnnxModel, graph: OnnxGraph.Class): void {
        const valueInfos = data.graph?.valueInfo;
        if (!valueInfos) return;

        // Skip Constant outputs here — addNodes registers them as ConstantNodes directly.
        const constantOutputs = new Set<string>();
        for (const node of data.graph?.node ?? []) {
            if (node.opType === "Constant") {
                for (const out of node.output ?? []) {
                    if (out !== "") constantOutputs.add(out);
                }
            }
        }

        for (const vi of valueInfos) {
            if (!constantOutputs.has(vi.name)) {
                OnnxGraphLoader.addTensorFromValueInfo(graph, vi, "intermediate");
            }
        }
    }

    private static parseStandardAttribute(attr: RawOnnxAttribute): unknown {
        switch (attr.type) {
            case AttributeType.FLOAT:
            case "FLOAT":
                return Number(attr.f);
            case AttributeType.INT:
            case "INT":
                return Number(attr.i);
            case AttributeType.STRING:
            case "STRING":
                return attr.s;
            case AttributeType.FLOATS:
            case "FLOATS":
                return attr.floats!.map(Number);
            case AttributeType.INTS:
            case "INTS":
                return attr.ints!.map(Number);
            case AttributeType.TENSOR:
            case "TENSOR":
                return attr.t;
            default:
                return undefined;
        }
    }

    /**
     * Parses operations in topological order, resolving implicit parent-scope captures
     * into local `RegionArgumentNode` proxies and bubbling child captures upward.
     */
    private static addNodes(
        builder: GraphBuilder,
        data: RawOnnxModel,
        graph: OnnxGraph.Class,
        parentGraph?: OnnxGraph.Class,
    ): string[] {
        const captures: string[] = [];
        const captureMap = new Map<string, RegionArgumentNode.Class>();
        const rawNodes = data.graph?.node ?? [];
        const nodesToAdd = new Set<number>(rawNodes.map((_: RawOnnxNode, i: number) => i));

        const resolveInput = (name: string): ValueNode | undefined => {
            if (graph.hasNode(name)) return tryAsValueNode(graph.getNodeById(name));
            if (captureMap.has(name)) return tryAsValueNode(captureMap.get(name));

            if (parentGraph !== undefined && parentGraph.hasNode(name)) {
                const parentNode = parentGraph.getNodeById(name);
                const parentVal = tryAsValueNode(parentNode);
                const type = parentVal?.literalType ?? DataType.UNDEFINED;
                const shape: Shape = parentVal?.shape ?? [];

                const argNode = graph
                    .addNode(uniq(graph, `capture_${name}`))
                    .init(new RegionArgumentNode.Builder(captures.length, name, type, shape))
                    .as(RegionArgumentNode);

                captureMap.set(name, argNode);
                captures.push(name);
                return argNode;
            }

            return undefined;
        };

        let loopCount = 0;

        while (nodesToAdd.size > 0) {
            let madeProgress = false;
            loopCount++;

            if (loopCount > nodesToAdd.size * 2 + 10) {
                console.warn(
                    "[initGraph] Potential cycle or missing input detected, processing remaining nodes anyway.",
                );
                madeProgress = true;
            }

            const currentBatch = Array.from(nodesToAdd);

            for (const nodeIndex of currentBatch) {
                const node = rawNodes[nodeIndex];

                const allInputsResolved = node.input?.every((inputName: string) => {
                    if (inputName === "") return true;
                    return resolveInput(inputName) !== undefined;
                });

                if (allInputsResolved === undefined && loopCount <= nodesToAdd.size * 2) {
                    continue;
                }

                // Materialize Constant ops directly as ConstantNodes
                if (
                    node.opType === "Constant" &&
                    node.output !== undefined &&
                    node.output.length > 0
                ) {
                    const outName = node.output[0];
                    const valAttr = node.attribute?.find(
                        (a: RawOnnxAttribute) => a.name === "value",
                    );
                    const constantValue: TensorProto | undefined = valAttr?.t;
                    if (!graph.hasNode(outName) && constantValue !== undefined) {
                        graph
                            .addNode(outName)
                            .init(new ConstantNode.Builder(constantValue))
                            .as(ConstantNode);
                    }
                    nodesToAdd.delete(nodeIndex);
                    madeProgress = true;
                    continue;
                }

                const inputs: ValueNode[] = [];
                for (const inputName of node.input ?? []) {
                    if (inputName === "") continue;
                    const resolved = resolveInput(inputName);
                    if (resolved !== undefined) inputs.push(resolved);
                }

                const { attributes, regions } = OnnxGraphLoader.parseAttributesAndRegions(
                    node,
                    graph,
                    resolveInput,
                );

                const validOutputs = (node.output ?? []).filter((o: string) => o !== "");
                builder.createOpWithExact(
                    nodeIndex.toString(),
                    node.opType!,
                    inputs,
                    validOutputs,
                    attributes as AttributeMap,
                    undefined,
                    regions,
                );

                nodesToAdd.delete(nodeIndex);
                madeProgress = true;
            }

            if (!madeProgress) {
                break;
            }
        }

        return captures;
    }

    /**
     * Parses standard attributes and control-flow subgraphs in a single pass,
     * preserving canonical region order (`If`: `[then, else]`, `Loop`/`Scan`: `[body]`)
     * and bubbling child captures up into `graph`.
     */
    private static parseAttributesAndRegions(
        node: RawOnnxNode,
        graph: OnnxGraph.Class,
        resolveInput: (name: string) => ValueNode | undefined,
    ): { attributes: Record<string, unknown>; regions: OnnxGraph.Class[] } {
        const attributes: Record<string, unknown> = {};
        const namedRegions = new Map<string, OnnxGraph.Class>();
        const genericRegions: OnnxGraph.Class[] = [];

        for (const attr of node.attribute ?? []) {
            if (
                attr.g != null &&
                (attr.type === AttributeType.GRAPH ||
                    attr.type === "GRAPH" ||
                    REGION_ATTR_NAMES.has(attr.name))
            ) {
                const { graph: subGraph, captures: subCaptures } =
                    OnnxGraphLoader.createGraphWithCaptures({ graph: attr.g }, graph);

                for (const capName of subCaptures) {
                    resolveInput(capName);
                }

                if (REGION_ATTR_NAMES.has(attr.name)) {
                    namedRegions.set(attr.name, subGraph);
                } else {
                    genericRegions.push(subGraph);
                }
            } else if (!REGION_ATTR_NAMES.has(attr.name)) {
                const parsed = OnnxGraphLoader.parseStandardAttribute(attr);
                if (parsed !== undefined) {
                    attributes[attr.name] = parsed;
                }
            }
        }

        const regions: OnnxGraph.Class[] = [];
        if (node.opType === "If") {
            const thenRegion = namedRegions.get("then_branch");
            const elseRegion = namedRegions.get("else_branch");
            if (thenRegion !== undefined) regions.push(thenRegion);
            if (elseRegion !== undefined) regions.push(elseRegion);
        } else if (node.opType === "Loop" || node.opType === "Scan") {
            const bodyRegion = namedRegions.get("body");
            if (bodyRegion !== undefined) regions.push(bodyRegion);
        } else {
            regions.push(...namedRegions.values(), ...genericRegions);
        }

        return { attributes, regions };
    }

    private static createGraphWithCaptures(
        data: RawOnnxModel,
        parentGraph?: OnnxGraph.Class,
        versionMeta?: GraphVersionMeta,
    ): { graph: OnnxGraph.Class; captures: string[] } {
        const builderOpts: { targetOpset?: number; irVersion?: number } = {};
        if (versionMeta?.targetOpset !== undefined) {
            builderOpts.targetOpset = versionMeta.targetOpset;
        } else if (parentGraph?.hasExplicitTargetOpset() === true) {
            builderOpts.targetOpset = parentGraph.getTargetOpset();
        }
        if (parentGraph?.hasExplicitIrVersion() === true) {
            builderOpts.irVersion = parentGraph.getIrVersion();
        }

        const graph = Graph.create().init(new OnnxGraph.Builder(builderOpts)).as(OnnxGraph);

        if (parentGraph === undefined) {
            if (versionMeta?.irVersion !== undefined) {
                graph.setIrVersion(versionMeta.irVersion, false);
            }
        } else if (!parentGraph.hasExplicitIrVersion()) {
            graph.setIrVersion(parentGraph.getIrVersion(), false);
        }

        const opsetImport = versionMeta?.opsetImport ?? parentGraph?.getOpsetImport();
        if (opsetImport !== undefined && opsetImport.length > 0) {
            graph.setOpsetImport(opsetImport);
        }

        const sourceOpset = versionMeta?.sourceOpset ?? parentGraph?.getSourceOpset();
        if (sourceOpset !== undefined) {
            graph.setSourceOpset(sourceOpset);
        }

        const builder = new GraphBuilder(graph);

        OnnxGraphLoader.addInitializers(data, graph);
        for (const input of data.graph?.input ?? []) {
            OnnxGraphLoader.addTensorFromValueInfo(graph, input, "input");
        }
        for (const output of data.graph?.output ?? []) {
            OnnxGraphLoader.addTensorFromValueInfo(graph, output, "output");
        }
        OnnxGraphLoader.addValueInfoNodes(data, graph);

        const captures = OnnxGraphLoader.addNodes(builder, data, graph, parentGraph);
        inferShapes(graph);

        return { graph, captures };
    }
}

/** Converts a raw ONNX model object into a shape-inferred `OnnxGraph`. */
export function createGraph(data: RawOnnxModel, targetOpset?: number): OnnxGraph.Class {
    return OnnxGraphLoader.load(data, targetOpset);
}
