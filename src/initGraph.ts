import OnnxGraph from "./Onnx/OnnxGraph.js";
import TensorNode from "./Onnx/TensorNode.js";
import Graph from "@specs-feup/flow/graph/Graph";
import type {
    AttributeMap,
    KnownDim,
    KnownShape,
    RawOnnxAttribute,
    RawOnnxDim,
    RawOnnxModel,
    RawOnnxNode,
    RawOnnxTensorType,
    RawOnnxValueInfo,
    Shape,
    TensorProto,
    ValueNode,
} from "./Onnx/OnnxTypes.js";
import { AttributeType, DataType } from "./Onnx/OnnxTypes.js";
import inferShapes from "./Onnx/InferShapes.js";
import { applyAdapters } from "./Onnx/Frontend/Adapters.js";
import ConstantNode from "./Onnx/ConstantNode.js";
import RegionArgumentNode from "./Onnx/RegionArgumentNode.js";
import { tryAsValueNode, uniq, UNKNOWN_SHAPE } from "./Onnx/Utils.js";
import { GraphBuilder } from "./Onnx/GraphBuilder.js";

// Helper function to convert shape to number[]
function parseShape(shape: RawOnnxTensorType["shape"]): KnownShape {
    if (!shape?.dim) return [];
    return shape.dim.map((dim: RawOnnxDim): KnownDim => {
        // Handle both camelCase and snake_case formats
        const dimParam = dim.dimParam ?? dim.dim_param;
        const dimValue = dim.dimValue ?? dim.dim_value;

        if (typeof dimParam === "string" && dimParam !== "") {
            return dimParam; // symbolic dimension, e.g., "batch"
        }

        if (dimValue !== undefined) {
            return Number(dimValue); // explicitly cast to number
        } else {
            return UNKNOWN_SHAPE[0]; // unknown shape
        }
    });
}

function addValueInfoNodes(data: RawOnnxModel, graph: OnnxGraph.Class): void {
    if (!data.graph || !data.graph.valueInfo) return;

    // Collect all outputs of Constant nodes so we don't create dummy intermediates for them
    const constantOutputs = new Set<string>();
    for (const node of data.graph.node ?? []) {
        if (node.opType === "Constant") {
            for (const out of node.output ?? []) {
                if (out) constantOutputs.add(out);
            }
        }
    }

    data.graph.valueInfo.forEach((vi: RawOnnxValueInfo) => {
        const name = vi.name;

        // Skip if we already created it as input/output/initializer/capture
        if (graph.hasNode(name)) return;

        // Skip valueInfo for Constant outputs — they'll be created as proper "constant" tensors in addNodes
        if (constantOutputs.has(name)) return;

        const tensorType = vi.type?.tensorType ?? vi.type?.tensor_type;
        const shape = parseShape(tensorType?.shape);
        const elemType = Number(tensorType?.elemType ?? 0);

        graph
            .addNode(name)
            .init(new TensorNode.Builder(elemType, shape, "intermediate"))
            .as(TensorNode);
    });
}

// Add initializers
function addInitializers(data: RawOnnxModel, graph: OnnxGraph.Class) {
    if (!data.graph || !data.graph.initializer) {
        return;
    }

    data.graph.initializer.forEach((tensor: TensorProto) => {
        if (tensor.name === undefined || !graph.hasNode(tensor.name)) {
            graph.addNode(tensor.name).init(new ConstantNode.Builder(tensor)).as(ConstantNode);
        }
    });
}

// Add input nodes to the graph
function addInputNodes(data: RawOnnxModel, graph: OnnxGraph.Class) {
    data.graph?.input?.forEach((input: RawOnnxValueInfo) => {
        if (!graph.hasNode(input.name)) {
            const tensorType = input.type?.tensorType ?? input.type?.tensor_type;
            const shape = parseShape(tensorType?.shape);
            const eltype = Number(tensorType?.elemType ?? 0);
            graph
                .addNode(input.name)
                .init(new TensorNode.Builder(eltype, shape, "input"))
                .as(TensorNode);
        }
    });
}

// Add output nodes to the graph
function addOutputNodes(data: RawOnnxModel, graph: OnnxGraph.Class) {
    data.graph?.output?.forEach((output: RawOnnxValueInfo) => {
        // If output node already exists (e.g. created by addNodes as intermediate), update type to 'output'
        // But here we usually create placeholders.
        if (!graph.hasNode(output.name)) {
            const tensorType = output.type?.tensorType ?? output.type?.tensor_type;
            const shape = parseShape(tensorType?.shape);
            const eltype = Number(tensorType?.elemType ?? 0);
            graph
                .addNode(output.name)
                .init(new TensorNode.Builder(eltype, shape, "output"))
                .as(TensorNode);
        }
    });
}

/**
 * Main node adder.
 * Returns the list of external variables 'captured' by this graph scope.
 */
function addNodes(
    builder: GraphBuilder,
    data: RawOnnxModel,
    graph: OnnxGraph.Class,
    parentGraph?: OnnxGraph.Class,
): string[] {
    const captures: string[] = [];
    const captureMap = new Map<string, RegionArgumentNode.Class>(); // Name -> Node

    const nodesToAdd = new Set<number>(data.graph!.node!.map((_: RawOnnxNode, i: number) => i));

    // Helper to resolve an input name to a node in the current scope or capture it
    function resolveInput(name: string): ValueNode | undefined {
        // 1. Local scope (Tensor, Constant, or existing Capture)
        if (graph.hasNode(name)) return tryAsValueNode(graph.getNodeById(name));

        // 2. Already captured in this pass but not yet fully registered? (Redundant with 1, but safe)
        if (captureMap.has(name)) return tryAsValueNode(captureMap.get(name));

        // 3. Check parent scope (Implicit Capture)
        if (parentGraph) {
            // Recursive lookup: The parent might have it local, OR it might capture it from grandparent
            // We assume the parent graph is fully built (or at least the relevant nodes exist).
            // However, since we parse recursively, the parent *is* built up to the point of this Loop node.

            // Note: OnnxGraph.hasNode is purely local.
            if (parentGraph.hasNode(name)) {
                const parentNode = parentGraph.getNodeById(name);

                // Determine type/shape from parent for the proxy node
                let type = DataType.UNDEFINED;
                let shape: Shape = [];

                if (parentNode !== undefined) {
                    if (parentNode.is(TensorNode)) {
                        const tn = parentNode.as(TensorNode);
                        type = tn.literalType;
                        shape = tn.shape;
                    } else if (parentNode.is(ConstantNode)) {
                        const cn = parentNode.as(ConstantNode);
                        type = cn.literalType;
                        shape = cn.shape;
                    } else if (parentNode.is(RegionArgumentNode)) {
                        const ra = parentNode.as(RegionArgumentNode);
                        type = ra.literalType;
                        shape = ra.shape;
                    }
                }

                // Create proxy node in CURRENT graph
                const argNode = graph
                    .addNode(uniq(graph, `capture_${name}`))
                    .init(new RegionArgumentNode.Builder(captures.length, name, type, shape))
                    .as(RegionArgumentNode);

                captureMap.set(name, argNode);
                captures.push(name);
                return argNode;
            }
        }

        return undefined; // Not found (might be waiting for topological sort, or strictly local)
    }

    let madeProgress = true;
    let loopCount = 0;

    // Simple topo-sort loop (relaxed for cycles, but mostly to handle order)
    while (nodesToAdd.size > 0) {
        madeProgress = false;
        loopCount++;

        // Safety break for cycles or unresolvable inputs (*2+10 as a threshold)
        if (loopCount > nodesToAdd.size * 2 + 10) {
            console.warn(
                "[initGraph] Potential cycle or missing input detected, processing remaining nodes anyway.",
            );
            madeProgress = true; // force proceed
        }

        const currentBatch = Array.from(nodesToAdd); // snapshot

        for (const nodeIndex of currentBatch) {
            const node = data.graph?.node![nodeIndex];
            if (!node) continue;

            // 1. Check if we can resolve all inputs (Local or Capture)
            const allInputsResolved = node.input?.every((inputName: string) => {
                if (inputName === "") return true; // Optional input
                return resolveInput(inputName) !== undefined;
            });

            if (allInputsResolved === undefined && loopCount <= nodesToAdd.size * 2) {
                continue; // Wait for inputs
            }

            // --- Node Processing ---
            // A. Handle Constant
            if (
                node.opType === "Constant" &&
                node.output !== undefined &&
                node.output.length &&
                node.output.length > 0
            ) {
                const name = node.output[0];
                let constantValue: TensorProto | undefined = undefined;

                // Try to find 'value' attribute
                const valAttr = node.attribute?.find((a: RawOnnxAttribute) => a.name === "value");
                if (valAttr && valAttr.t) constantValue = valAttr.t;

                if (!graph.hasNode(name) && constantValue) {
                    graph
                        .addNode(name)
                        .init(new ConstantNode.Builder(constantValue))
                        .as(ConstantNode);
                }
                nodesToAdd.delete(nodeIndex);
                madeProgress = true;
                continue;
            }

            // B. Resolve Inputs (Actual wiring)
            const inputs: ValueNode[] = [];
            node.input?.forEach((inputName: string) => {
                if (inputName === "") return;
                const resolved = resolveInput(inputName);
                if (resolved) inputs.push(resolved);
            });

            // C. Parse Attributes & Regions
            const attributes: Record<string, unknown> = {};
            const regions: OnnxGraph.Class[] = [];

            // We need to maintain the order of regions as expected by OperationNode
            // e.g. If -> [then, else].
            // We'll collect them temporarily and sort them if necessary,
            // but for now, we rely on the parser processing them in order or specific handlers.

            if (node.attribute) {
                for (const attr of node.attribute) {
                    // 1. Handle Subgraphs
                    if ((attr.type === AttributeType.GRAPH || attr.type === "GRAPH") && attr.g) {
                        // RECURSION: Parse the subgraph with the current graph as Parent
                        const { graph: subGraph, captures: subCaptures } = createGraphWithCaptures(
                            { graph: attr.g }, // Wrap to match expected data format
                            graph, // Parent
                        );

                        regions.push(subGraph);

                        // **Bubble Up Captures**:
                        // If the child captured 'X', and 'X' is not local to ME,
                        // then *I* must also capture 'X' from *my* parent.
                        // This ensures the chain of RegionArgumentNodes goes all the way up.
                        for (const capName of subCaptures) {
                            // Try to resolve it in MY scope (Local or Capture from Parent)
                            const resolvedCap = resolveInput(capName);
                            if (resolvedCap) {
                                // We don't necessarily add it to 'inputs' of the Loop Op itself
                                // (ONNX doesn't require explicit inputs for implicit captures on the Loop node inputs list),
                                // BUT `RegionArgumentNode` logic relies on the graph structure being valid.
                                // In strict IR, the Loop Op *should* probably have these as inputs.
                                // For now, we just ensure the node exists in this graph so the child can link to it.
                            }
                        }
                    }
                    // 2. Handle Standard Attributes
                    else if (
                        attr.name !== "body" &&
                        attr.name !== "then_branch" &&
                        attr.name !== "else_branch"
                    ) {
                        switch (attr.type) {
                            case AttributeType.FLOAT:
                            case "FLOAT":
                                attributes[attr.name] = Number(attr.f);
                                break;
                            case AttributeType.INT:
                            case "INT":
                                attributes[attr.name] = Number(attr.i);
                                break;
                            case AttributeType.STRING:
                            case "STRING":
                                attributes[attr.name] = attr.s;
                                break;
                            case AttributeType.FLOATS:
                            case "FLOATS":
                                attributes[attr.name] = attr.floats!.map(Number);
                                break;
                            case AttributeType.INTS:
                            case "INTS":
                                attributes[attr.name] = attr.ints!.map(Number);
                                break;
                            case AttributeType.TENSOR:
                            case "TENSOR":
                                attributes[attr.name] = attr.t;
                                break;
                        }
                    }
                }
            }

            // Re-map regions correctly based on attribute names if order matters
            if (node.opType === "If" || node.opType === "Loop" || node.opType === "Scan") {
                const orderedRegions: OnnxGraph.Class[] = [];
                const attrMap = new Map<string, RawOnnxAttribute>();
                node.attribute?.forEach((a: RawOnnxAttribute) => attrMap.set(a.name, a));

                if (node.opType === "If") {
                    // Parse 'then_branch'
                    const tA = attrMap.get("then_branch");
                    if (tA?.g)
                        orderedRegions.push(createGraphWithCaptures({ graph: tA.g }, graph).graph);
                    // Parse 'else_branch'
                    const eA = attrMap.get("else_branch");
                    if (eA?.g)
                        orderedRegions.push(createGraphWithCaptures({ graph: eA.g }, graph).graph);
                } else {
                    //Loop or Scan case
                    const bA = attrMap.get("body");
                    if (bA?.g)
                        orderedRegions.push(createGraphWithCaptures({ graph: bA.g }, graph).graph);
                }
                // Replace the generic 'regions' from the loop with this ordered list
                if (orderedRegions.length > 0) regions.splice(0, regions.length, ...orderedRegions);
            }

            const opId = nodeIndex.toString();
            const validOutputs = (node.output ?? []).filter((o: string) => o !== "");

            builder.createOpWithExact(
                opId,
                node.opType!,
                inputs, // Resolved ValueNodes
                validOutputs, // Exact ONNX tensor names
                attributes as AttributeMap,
                undefined, // Shapes will be inferred automatically by the builder
                regions,
            );

            nodesToAdd.delete(nodeIndex);
            madeProgress = true;
        }

        if (!madeProgress) {
            // Force break to prevent infinite loop on malformed graphs
            break;
        }
    }

    return captures;
}

function createGraphWithCaptures(
    data: RawOnnxModel,
    parentGraph?: OnnxGraph.Class,
): { graph: OnnxGraph.Class; captures: string[] } {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const builder = new GraphBuilder(graph);

    addInitializers(data, graph);
    addInputNodes(data, graph);
    addOutputNodes(data, graph);
    addValueInfoNodes(data, graph);

    const captures = addNodes(builder, data, graph, parentGraph);

    inferShapes(graph);

    return { graph, captures };
}

export function createGraph(data: RawOnnxModel): OnnxGraph.Class {
    applyAdapters(data);
    return createGraphWithCaptures(data, undefined).graph;
}
