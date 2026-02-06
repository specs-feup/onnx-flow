import OnnxGraph from "./Onnx/OnnxGraph.js";
import TensorNode from "./Onnx/TensorNode.js";
import OperationNode from "./Onnx/OperationNode.js";
import OnnxEdge from "./Onnx/OnnxEdge.js";
import Graph from "@specs-feup/flow/graph/Graph";
import { AttributeProto, AttributeType, TensorProto } from "./Onnx/OnnxTypes.js";
import inferShapes from "./Onnx/InferShapes.js";

function addValueInfoNodes(data: any, graph: OnnxGraph.Class) {
    if (!data.graph.valueInfo) return;

    // Collect all outputs of Constant nodes so we don't create dummy intermediates for them
    const constantOutputs = new Set<string>();
    for (const node of data.graph.node ?? []) {
        if (node.opType === "Constant") {
            for (const out of node.output ?? []) {
                if (out) constantOutputs.add(out);
            }
        }
    }

    data.graph.valueInfo.forEach((vi: any) => {
        const name = vi.name;

        // Skip if we already created it as input/output/initializer
        if (graph.hasNode(name)) return;

        // Skip valueInfo for Constant outputs — they'll be created as proper "constant" tensors in addNodes
        if (constantOutputs.has(name)) return;

        const shape = parseShape(vi.type.tensorType ? vi.type.tensorType.shape : []);
        const elemType = vi.type.tensorType ? vi.type.tensorType.elemType : 0;

        graph
            .addNode(name)
            .init(new TensorNode.Builder(elemType, shape, "intermediate"))
            .as(TensorNode);

        definedVars.push(name);
    });
}

// Helper function to convert shape to number[]
function parseShape(shape: any): (number | string)[] {
    if (!shape?.dim) return [];
    return shape.dim.map((dim: any) => {
        if (typeof dim.dimParam === "string") {
            return dim.dimParam;
        }
        if (dim.dimValue !== undefined && dim.dimValue !== null) {
            return Number(dim.dimValue);
        } else if (dim.dimParam !== undefined && dim.dimParam !== "") {
            return dim.dimParam; // symbolic dimension, e.g., "batch"
        } else {
            return undefined; // unknown dimension size
        }
    });
}

const definedVars: string[] = [];

// Add initializers
function addInitializers(data: any, graph: OnnxGraph.Class) {
    if (!data.graph.initializer) {
        return;
    }

    data.graph.initializer.forEach((tensor: any) => {
        const shape = tensor.dims.map((d: number) => Number(d));
        const elemType = tensor.dataType;

        graph
            .addNode(tensor.name)
            .init(new TensorNode.Builder(elemType, shape, "initializer", undefined, tensor))
            .as(TensorNode);

        definedVars.push(tensor.name);
    });
}

// Add input nodes to the graph
function addInputNodes(data: any, graph: OnnxGraph.Class) {
    data.graph.input.forEach((input: any) => {
        const shape = parseShape(input.type.tensorType ? input.type.tensorType.shape : []);
        const eltype = input.type.tensorType ? input.type.tensorType.elemType : 0;
        graph
            .addNode(input.name)
            .init(new TensorNode.Builder(eltype, shape, "input"))
            .as(TensorNode);
        definedVars.push(input.name);
    });
}

// Add output nodes to the graph
function addOutputNodes(data: any, graph: OnnxGraph.Class) {
    data.graph.output.forEach((output: any) => {
        const shape = parseShape(output.type.tensorType ? output.type.tensorType.shape : []);
        const eltype = output.type.tensorType ? output.type.tensorType.elemType : 0;
        graph
            .addNode(output.name)
            .init(new TensorNode.Builder(eltype, shape, "output"))
            .as(TensorNode);
    });
}

// Add operation nodes to the graph
function addNodes(
    data: any,
    graph: OnnxGraph.Class,
    mapNodeAndOutput: any[],
    mapNodeAndInputs: any[],
    maingraph?: OnnxGraph.Class,
) {
    let index = 0;
    const nodesToAdd = new Set<number>(data.graph.node.map((_: any, i: number) => i));
    const addedNodes = new Set<number>();

    while (nodesToAdd.size > 0) {
        for (const nodeIndex of nodesToAdd) {
            const node = data.graph.node[nodeIndex];
            const allInputsDefined = node.input.every((input: string) =>
                definedVars.includes(input),
            );

            if (node.opType === "Constant" && node.output?.length > 0) {
                const name = node.output[0];

                let constantValue: TensorProto | undefined = undefined;
                const extraAttrs: AttributeProto[] = [];

                for (const attr of node.attribute ?? []) {
                    if (attr.t) {
                        constantValue = attr.t;
                    } else {
                        extraAttrs.push(attr);
                    }
                }

                const dataType = constantValue?.dataType ?? AttributeType.UNDEFINED;
                const shape = constantValue?.dims ?? [];

                if (!graph.hasNode(name)) {
                    graph
                        .addNode(name)
                        .init(
                            new TensorNode.Builder(
                                dataType,
                                shape,
                                "constant",
                                constantValue,
                                undefined,
                                extraAttrs,
                            ),
                        )
                        .as(TensorNode);
                }

                definedVars.push(name);
                addedNodes.add(nodeIndex);
                continue;
            }

            const inputs = [];
            node.input.forEach((input: any) => {
                if (graph.hasNode(input)) {
                    inputs.push(graph.getNodeById(input));
                } else {
                    if (maingraph && maingraph.hasNode(input)) {
                        inputs.push(maingraph.getNodeById(input));
                    }
                }
            });

            if (allInputsDefined) {
                const attributes: Record<string, any> = {};
                if (node.attribute) {
                    for (const attr of node.attribute) {
                        if (!attr.name || attr.name === "body") continue;

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
                                attributes[attr.name] = attr.floats;
                                break;
                            case AttributeType.INTS:
                            case "INTS":
                                attributes[attr.name] = attr.ints;
                                break;
                            case AttributeType.TENSOR:
                            case "TENSOR":
                                attributes[attr.name] = attr.t;
                                break;
                            default:
                                console.warn(node);
                                console.warn(
                                    `[addNodes] Unhandled attribute type '${attr.type}' for '${attr.name}'`,
                                );
                        }
                    }
                }

                // Handle Loop's body subgraph
                let opBuilder;
                if (node.opType === "Loop") {
                    const bodyAttr = node.attribute.find(
                        (attr: any) => attr.name === "body" && attr.g,
                    );
                    const subgraph = bodyAttr
                        ? createGraph({ graph: bodyAttr.g }, graph)
                        : undefined;
                    opBuilder = new OperationNode.Builder(
                        node.opType,
                        inputs,
                        attributes,
                        subgraph,
                    );
                } else if (node.opType === "If") {
                    const thenAttr = node.attribute.find(
                        (attr: any) => attr.name === "then_branch" && attr.g,
                    );
                    const elseAttr = node.attribute.find(
                        (attr: any) => attr.name === "else_branch" && attr.g,
                    );

                    const thenGraph = thenAttr
                        ? createGraph({ graph: thenAttr.g }, graph)
                        : undefined;
                    const elseGraph = elseAttr
                        ? createGraph({ graph: elseAttr.g }, graph)
                        : undefined;

                    const subgraphs = {
                        thenBranch: thenGraph,
                        elseBranch: elseGraph,
                    };

                    opBuilder = new OperationNode.Builder(
                        node.opType,
                        inputs,
                        attributes,
                        subgraphs,
                    );
                } else if (node.opType === "Scan") {
                    const bodyAttr = node.attribute.find(
                        (attr: any) => attr.name === "body" && attr.g,
                    );
                    const subgraph = bodyAttr
                        ? createGraph({ graph: bodyAttr.g }, graph)
                        : undefined;
                    opBuilder = new OperationNode.Builder(node.opType, inputs, attributes, {
                        body: subgraph,
                    });
                } else {
                    opBuilder = new OperationNode.Builder(node.opType, inputs, attributes);
                }

                graph.addNode(index.toString()).init(opBuilder).as(OperationNode);

                node.output.forEach((output: any) => {
                    if (!graph.hasNode(output)) {
                        const inferredShape = inputs[0]?.tryAs(TensorNode)?.shape ?? [];
                        const inferredType =
                            inputs[0]?.tryAs(TensorNode)?.literalType ?? AttributeType.UNDEFINED;

                        graph
                            .addNode(output)
                            .init(
                                new TensorNode.Builder(inferredType, inferredShape, "intermediate"),
                            )
                            .as(TensorNode);
                    } else {
                        // Node already exists (e.g. from valueInfo); DO NOT change its shape/type.
                    }

                    mapNodeAndOutput.push({ nodeId: index.toString(), output });
                    definedVars.push(output);
                });

                mapNodeAndInputs.push({ nodeId: index.toString(), inputs: node.input });
                addedNodes.add(nodeIndex);
                index++;
            }
        }

        addedNodes.forEach((nodeIndex) => nodesToAdd.delete(nodeIndex));
        addedNodes.clear();
    }
}

// Calculate dimensions and add edges to the graph
function addEdges(graph: OnnxGraph.Class, mapNodeAndOutput: any[], mapNodeAndInputs: any[]) {
    mapNodeAndInputs.forEach((node) => {
        const opNode = graph.getNodeById(node.nodeId);
        if (opNode && node.inputs) {
            node.inputs.forEach((input: string) => {
                const inputNode = graph.getNodeById(input)?.tryAs(TensorNode);
                if (inputNode && !inputNode.isConstant()) {
                    const sourceShape = inputNode.shape;
                    const sourceElemType = inputNode.literalType;
                    graph
                        .addEdge(inputNode, opNode)
                        .init(new OnnxEdge.Builder(sourceElemType, sourceShape))
                        .as(OnnxEdge);
                } else {
                    const nodeWithCorrespondingOutput = mapNodeAndOutput.find(
                        (elem) => elem.output === input,
                    );
                    if (nodeWithCorrespondingOutput) {
                        const outputNode = graph
                            .getNodeById(nodeWithCorrespondingOutput.nodeId)
                            ?.tryAs(TensorNode);
                        if (outputNode && !outputNode.isConstant()) {
                            graph
                                .addEdge(outputNode, opNode)
                                .init(
                                    new OnnxEdge.Builder(outputNode.literalType, outputNode.shape),
                                )
                                .as(OnnxEdge);
                        }
                    }
                }
            });

            mapNodeAndOutput.forEach((nodeAndOutput) => {
                if (nodeAndOutput.nodeId === opNode.id) {
                    const outputNode = graph.getNodeById(nodeAndOutput.output)?.tryAs(TensorNode);
                    if (outputNode && !outputNode.isConstant()) {
                        graph
                            .addEdge(opNode, outputNode)
                            .init(new OnnxEdge.Builder(outputNode.literalType, outputNode.shape))
                            .as(OnnxEdge);
                    }
                }
            });
        }
    });
}

// Create the graph using the implemented classes
export function createGraph(data: any, mainGraph?: OnnxGraph.Class): OnnxGraph.Class {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    addInitializers(data, graph);
    addInputNodes(data, graph);
    addOutputNodes(data, graph);
    addValueInfoNodes(data, graph);

    const mapNodeAndOutput: any[] = [];
    const mapNodeAndInputs: any[] = [];

    addNodes(data, graph, mapNodeAndOutput, mapNodeAndInputs, mainGraph);
    addEdges(graph, mapNodeAndOutput, mapNodeAndInputs);

    inferShapes(graph);

    return graph;
}
