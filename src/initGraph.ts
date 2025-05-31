import OnnxGraph from "./Onnx/OnnxGraph.js";
import TensorNode from "./Onnx/TensorNode.js";
import OperationNode from "./Onnx/OperationNode.js";
import OnnxEdge from "./Onnx/OnnxEdge.js";
import Graph from "@specs-feup/flow/graph/Graph";
import { NodeCollection } from "@specs-feup/flow/graph/NodeCollection";


// Helper function to convert shape to number[]
function parseShape(shape: any): number[] {
    return shape.dim.map((dim: any) => parseInt(dim.dimValue, 10));
}

let definedVars: string[] = [];


// Add initializers
function addInitializers(data: any, graph: OnnxGraph.Class) {
    if (!data.graph.initializer) {
        return;
    }

    data.graph.initializer.forEach((tensor: any) => {
        const shape = tensor.dims.map((d: number) => Number(d));
        const elemType = tensor.dataType;
        graph.addNode(tensor.name).init(new TensorNode.Builder(elemType, shape, 'initializer')).as(TensorNode);
        definedVars.push(tensor.name);
    });
}


// Add input nodes to the graph
function addInputNodes(data: any, graph: OnnxGraph.Class) {
    data.graph.input.forEach((input: any) => {
        const shape = parseShape(input.type.tensorType.shape);
        graph.addNode(input.name).init(new TensorNode.Builder(input.type.tensorType.elemType, shape, 'input')).as(TensorNode);
        definedVars.push(input.name);
    });
}


// Add output nodes to the graph
function addOutputNodes(data: any, graph: OnnxGraph.Class) {
    data.graph.output.forEach((output: any) => {
        const shape = parseShape(output.type.tensorType.shape);
        graph.addNode(output.name).init(new TensorNode.Builder(output.type.tensorType.elemType, shape, 'output')).as(TensorNode);
    });
}


// Add operation nodes to the graph
function addNodes(data: any, graph: OnnxGraph.Class, mapNodeAndOutput: any[], mapNodeAndInputs: any[]) {
  let index = 0;
  const nodesToAdd = new Set<number>(data.graph.node.map((_: any, i: number) => i));
  const addedNodes = new Set<number>();

  while (nodesToAdd.size > 0) {
    for (const nodeIndex of nodesToAdd) {
      const node = data.graph.node[nodeIndex];
      const allInputsDefined = node.input.every((input: string) => definedVars.includes(input));

      if (allInputsDefined) {
        const attributes: Record<string, any> = {};
        if (node.attribute) {
          for (const attr of node.attribute) {
            if ('i' in attr) attributes[attr.name] = attr.i;
            else if ('f' in attr) attributes[attr.name] = attr.f;
            else if ('s' in attr) attributes[attr.name] = attr.s;
            else if ('ints' in attr) attributes[attr.name] = attr.ints;
            else if ('floats' in attr) attributes[attr.name] = attr.floats;
            else if ('g' in attr) continue; // handled below
            else console.warn(`Unhandled attribute type in ${node.opType}.${attr.name}`);
          }
        }

        // Handle Loop's body subgraph
        let opBuilder;
        if (node.opType === "Loop") {
          const bodyAttr = node.attribute.find((attr: any) => attr.name === "body" && attr.g);
          const subgraph = bodyAttr ? createGraph({ graph: bodyAttr.g }) : undefined;
          opBuilder = new OperationNode.Builder(node.opType, attributes, subgraph);
        } else {
          opBuilder = new OperationNode.Builder(node.opType, attributes);
        }

        graph.addNode(index.toString()).init(opBuilder).as(OperationNode);

        node.output.forEach((output: any) => {
          if (!graph.hasNode(output)) {
            const shape: number[] = [];
            graph.addNode(output).init(new TensorNode.Builder(0, shape, 'intermediate')).as(TensorNode);
          }

          mapNodeAndOutput.push({ nodeId: index.toString(), output });
          definedVars.push(output);
        });

        mapNodeAndInputs.push({ nodeId: index.toString(), inputs: node.input });
        addedNodes.add(nodeIndex);
        index++;
      }
    }

    addedNodes.forEach(nodeIndex => nodesToAdd.delete(nodeIndex));
    addedNodes.clear();
  }
}


// Calculate dimensions and add edges to the graph
function addEdges(graph: OnnxGraph.Class, mapNodeAndOutput: any[], mapNodeAndInputs: any[]) {
    mapNodeAndInputs.forEach(node => {
        const opNode = graph.getNodeById(node.nodeId);
        if (opNode) {
            node.inputs.forEach((input: string) => {
                const inputNode = graph.getNodeById(input)?.tryAs(TensorNode);
                if (inputNode) {
                    const sourceShape = inputNode.shape;
                    const sourceElemType = inputNode.literalType;
                    graph.addEdge(inputNode, opNode).init(new OnnxEdge.Builder(sourceElemType, sourceShape)).as(OnnxEdge);
                } else {
                    const nodeWithCorrespondingOutput = mapNodeAndOutput.find(elem => elem.output === input);
                    if (nodeWithCorrespondingOutput) {
                        const outputNode = graph.getNodeById(nodeWithCorrespondingOutput.nodeId);
                        if (outputNode) {
                            graph.addEdge(outputNode, opNode).init(new OnnxEdge.Builder()).as(OnnxEdge);
                        }
                    }
                }
            });

            mapNodeAndOutput.forEach(nodeAndOutput => {
                if (nodeAndOutput.nodeId === opNode.id) {
                    const outputNode = graph.getNodeById(nodeAndOutput.output);
                    if (outputNode) {
                        graph.addEdge(opNode, outputNode).init(new OnnxEdge.Builder()).as(OnnxEdge);
                    }
                }
            });
        }
    });
}


// Find dimensions
function findDims(outputNodes: NodeCollection<OperationNode.Class>, graph: OnnxGraph.Class) {
    outputNodes.forEach(node => {
        const incomingEdges = node.getIncomers;
        incomingEdges.forEach(edge => {
            if (edge.shape.length === 0) {
                const sourceNode = edge.source.tryAs(OperationNode);
                if (sourceNode) {
                    const sourceNodes = graph.arrayCollection(OperationNode, [sourceNode]);
                    findDims(sourceNodes, graph);
                }
            }
        });

        const firstEdge = incomingEdges[0].tryAs(OnnxEdge);
        let outgoingEdges = node.getOutgoers;

        if (firstEdge) {
            let firstEdgeDims = firstEdge.shape;
            let firstEdgeElemType = firstEdge.literalType;
            if (node.type === "MatMul") {
                const secondEdge = incomingEdges[1].tryAs(OnnxEdge);
                if (secondEdge) {
                    let secondEdgeDims = secondEdge.shape;
                    if (firstEdgeDims[1] === secondEdgeDims[0]) {
                        outgoingEdges.forEach(edge => {
                            edge.shape = [firstEdgeDims[0], secondEdgeDims[1]];
                            edge.literalType = firstEdgeElemType;
                        });
                    } else {
                        outgoingEdges.forEach(edge => {
                            edge.shape = [firstEdgeDims[1], secondEdgeDims[0]];
                            edge.literalType = firstEdgeElemType;
                        });
                    }
                }
            } else {
                outgoingEdges.forEach(edge => {
                    edge.shape = firstEdgeDims;
                    edge.literalType = firstEdgeElemType;
                });
            }
        }
    });
}


// Create the graph using the implemented classes
export function createGraph(data: any): OnnxGraph.Class {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    addInitializers(data, graph);
    addInputNodes(data, graph);
    addOutputNodes(data, graph);

    const mapNodeAndOutput: any[] = [];
    const mapNodeAndInputs: any[] = [];

    addNodes(data, graph, mapNodeAndOutput, mapNodeAndInputs);
    addEdges(graph, mapNodeAndOutput, mapNodeAndInputs);

    const lastOpNodes = graph.getOutputTensorNodes().incomers.sources.filterIs(OperationNode);
    findDims(lastOpNodes, graph);

    return graph;
}
