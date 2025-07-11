import OnnxGraph from "./Onnx/OnnxGraph.js";
import TensorNode from "./Onnx/TensorNode.js";
import OperationNode from "./Onnx/OperationNode.js";
import OnnxEdge from "./Onnx/OnnxEdge.js";
import Graph from "@specs-feup/flow/graph/Graph";
import { NodeCollection } from "@specs-feup/flow/graph/NodeCollection";
import { AttributeProto, AttributeType, TensorProto } from "./Onnx/OnnxTypes.js";

const BASE_TEN = 10;

// Helper function to convert shape to number[]
function parseShape(shape: any): number[] {
    return shape.dim.map((dim: any) => parseInt(dim.dimValue, BASE_TEN));
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

    graph.addNode(tensor.name)
      .init(new TensorNode.Builder(elemType, shape, "initializer", undefined, tensor))
      .as(TensorNode);

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
function addNodes(data: any, graph: OnnxGraph.Class, mapNodeAndOutput: any[], mapNodeAndInputs: any[], maingraph?: OnnxGraph.Class) {
  let index = 0;
  const nodesToAdd = new Set<number>(data.graph.node.map((_: any, i: number) => i));
  const addedNodes = new Set<number>();

  while (nodesToAdd.size > 0) {
    for (const nodeIndex of nodesToAdd) {
      const node = data.graph.node[nodeIndex];
      const allInputsDefined = node.input.every((input: string) => definedVars.includes(input));
    
    
    if (node.opType === "Constant" && node.output?.length > 0) {
        const name = node.output[0];

        let constantValue: TensorProto | undefined = undefined;
        let extraAttrs: AttributeProto[] = [];

        for (const attr of node.attribute ?? []) {
            if (attr.t) {
                constantValue = attr.t;
            } else {
                extraAttrs.push(attr);
            }
        }

        graph.addNode(name)
        .init(new TensorNode.Builder(AttributeType.UNDEFINED, [], "constant", constantValue, undefined, extraAttrs))
        .as(TensorNode);


        definedVars.push(name);
        addedNodes.add(nodeIndex);
        continue;
    }

      const inputs = [];
      node.input.forEach((input: any) => {
          if (graph.hasNode(input)){
            inputs.push(graph.getNodeById(input));
          }
          else{
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
                default:
                    console.warn(`[addNodes] Unhandled attribute type '${attr.type}' for '${attr.name}'`);
                }
            }
        }


        // Handle Loop's body subgraph
        let opBuilder;
        if (node.opType === "Loop") {
          const bodyAttr = node.attribute.find((attr: any) => attr.name === "body" && attr.g);
          const subgraph = bodyAttr ? createGraph({ graph: bodyAttr.g }, graph) : undefined;
          opBuilder = new OperationNode.Builder(node.opType, inputs, attributes, subgraph);
        } else {
          opBuilder = new OperationNode.Builder(node.opType, inputs, attributes);
        }

        graph.addNode(index.toString()).init(opBuilder).as(OperationNode);

        node.output.forEach((output: any) => {
          if (!graph.hasNode(output)) {
            graph.addNode(output).init(new TensorNode.Builder(AttributeType.UNDEFINED, [], 'intermediate')).as(TensorNode);
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
        if (opNode && node.inputs) {
            node.inputs.forEach((input: string) => {
                const inputNode = graph.getNodeById(input)?.tryAs(TensorNode);
                if (inputNode && !inputNode.isConstant()) {
                    const sourceShape = inputNode.shape;
                    const sourceElemType = inputNode.literalType;
                    graph.addEdge(inputNode, opNode).init(new OnnxEdge.Builder(sourceElemType, sourceShape)).as(OnnxEdge);
                } else {
                    const nodeWithCorrespondingOutput = mapNodeAndOutput.find(elem => elem.output === input);
                    if (nodeWithCorrespondingOutput) {
                        const outputNode = graph.getNodeById(nodeWithCorrespondingOutput.nodeId)?.tryAs(TensorNode);
                        if (outputNode && !outputNode.isConstant()) {
                            graph.addEdge(outputNode, opNode).init(new OnnxEdge.Builder()).as(OnnxEdge);
                        }
                    }
                }
            });

            mapNodeAndOutput.forEach(nodeAndOutput => {
                if (nodeAndOutput.nodeId === opNode.id) {
                    const outputNode = graph.getNodeById(nodeAndOutput.output)?.tryAs(TensorNode);
                    if (outputNode && !outputNode.isConstant()) {
                        graph.addEdge(opNode, outputNode).init(new OnnxEdge.Builder()).as(OnnxEdge);
                    }
                }
            });
        }
    });
}


// Infer Intermediate Shapes
function inferShapes(graph: OnnxGraph.Class): void {
  /* The nodes were inserted only after *all* their inputs existed, so
     the numeric string-ids are already a topological ordering.        */
  const ops = graph.getOperationNodes()
                   .toArray()
                   .sort((a, b) => Number(a.id) - Number(b.id));

  for (const node of ops) {
    const inputs  = node.getInputs?.() ?? [];
    const infos   = inputs.map(inp => {
      const edge  = graph.getEdge(inp.id, node.id)?.tryAs(OnnxEdge);
      const tns   = inp.tryAs(TensorNode);

      return {
        shape : edge?.shape       ?? tns?.shape       ?? [],
        dtype : edge?.literalType ?? tns?.literalType ?? AttributeType.UNDEFINED
      };
    });

    /* special MatMul rule ------------------------------------------ */
    if (node.type === "MatMul" && infos.length >= 2) {
      const [a,b]    = infos;
      const resShape = (a.shape[1] === b.shape[0])
                     ? [a.shape[0], b.shape[1]]
                     : [a.shape[1], b.shape[0]];

      node.getOutgoers.forEach(e => {
        e.shape       = resShape;
        e.literalType = a.dtype;
      });
      continue;
    }

    /* fall-back: copy first available shape ------------------------ */
    const first = infos.find(i => i.shape.length);
    if (first) {
      node.getOutgoers.forEach(e => {
        e.shape       = first.shape;
        e.literalType = first.dtype;
      });
    }
  }
}



// Create the graph using the implemented classes
export function createGraph(data: any, mainGraph?: OnnxGraph.Class): OnnxGraph.Class {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

    addInitializers(data, graph);
    addInputNodes(data, graph);
    addOutputNodes(data, graph);

    const mapNodeAndOutput: any[] = [];
    const mapNodeAndInputs: any[] = [];

    addNodes(data, graph, mapNodeAndOutput, mapNodeAndInputs, mainGraph);
    addEdges(graph, mapNodeAndOutput, mapNodeAndInputs);

    inferShapes(graph);

    return graph;
}