import OnnxGraph from "./Onnx/OnnxGraph.js";
import TensorNode from "./Onnx/TensorNode.js";
import OperationNode from "./Onnx/OperationNode.js";
import OnnxEdge from "./Onnx/OnnxEdge.js";
import Graph from "@specs-feup/flow/graph/Graph";
import { AttributeProto, AttributeType, TensorProto } from "./Onnx/OnnxTypes.js";
import { topologicalSortOperationNodes } from "./flow2json.js";

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

        const dataType = constantValue?.dataType ?? AttributeType.UNDEFINED;
        const shape = constantValue?.dims ?? [];
        graph.addNode(name)
          .init(new TensorNode.Builder(dataType, shape, "constant", constantValue, undefined, extraAttrs))
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
        } else if (node.opType === "If") {
          const thenAttr = node.attribute.find((attr: any) => attr.name === "then_branch" && attr.g);
          const elseAttr = node.attribute.find((attr: any) => attr.name === "else_branch" && attr.g);

          const thenGraph = thenAttr ? createGraph({ graph: thenAttr.g }, graph) : undefined;
          const elseGraph = elseAttr ? createGraph({ graph: elseAttr.g }, graph) : undefined;

          const subgraphs = {
            thenBranch: thenGraph,
            elseBranch: elseGraph
          };

          opBuilder = new OperationNode.Builder(node.opType, inputs, attributes, subgraphs);

        } else if (node.opType === "Scan") {
          const bodyAttr = node.attribute.find((attr: any) => attr.name === "body" && attr.g);
          const subgraph = bodyAttr ? createGraph({ graph: bodyAttr.g }, graph) : undefined;
          opBuilder = new OperationNode.Builder(node.opType, inputs, attributes, { body: subgraph });

        } else {
          opBuilder = new OperationNode.Builder(node.opType, inputs, attributes);
        }

        graph.addNode(index.toString()).init(opBuilder).as(OperationNode);

        node.output.forEach((output: any) => {
          if (!graph.hasNode(output)) {
            const inferredShape = inputs[0]?.tryAs(TensorNode)?.shape ?? [];
            const inferredType = inputs[0]?.tryAs(TensorNode)?.literalType ?? AttributeType.UNDEFINED;

            graph.addNode(output)
              .init(new TensorNode.Builder(inferredType, inferredShape, 'intermediate'))
              .as(TensorNode);
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
                            graph.addEdge(outputNode, opNode).init(new OnnxEdge.Builder(outputNode.literalType, outputNode.shape)).as(OnnxEdge);
                        }
                    }
                }
            });

            mapNodeAndOutput.forEach(nodeAndOutput => {
                if (nodeAndOutput.nodeId === opNode.id) {
                    const outputNode = graph.getNodeById(nodeAndOutput.output)?.tryAs(TensorNode);
                    if (outputNode && !outputNode.isConstant()) {
                        graph.addEdge(opNode, outputNode).init(new OnnxEdge.Builder(outputNode.literalType, outputNode.shape)).as(OnnxEdge);
                    }
                }
            });
        }
    });
}


// Infer Intermediate Shapes
export function inferShapes(graph: OnnxGraph.Class): void {
  const ops = topologicalSortOperationNodes(graph);
  
  for (const node of ops) {
    const inputs = node.getInputs?.() ?? [];
    const infos = inputs.map(inp => {
      const tns = inp.tryAs(TensorNode);

      let interEdge = null;
      if (tns.type === "intermediate") interEdge = tns.getIncomers.first;

      const directEdge = graph.getEdge(inp.id, node.id)?.tryAs(OnnxEdge);

      // Optional Debug: Log all sources to help debug which one is being used
      /*
      console.log(`INFO CALC FOR node ${node.id}, input ${inp.id}`);
      console.log("Intermediate edge:", interEdge?.shape, interEdge?.literalType);
      console.log("Direct edge:", directEdge?.shape, directEdge?.literalType);
      console.log("Tensor node:", tns?.shape, tns?.literalType);
      */

      return {
        shape: interEdge?.shape ?? directEdge?.shape ?? tns?.shape ?? [],
        dtype: interEdge?.literalType ?? directEdge?.literalType ?? tns?.literalType ?? AttributeType.UNDEFINED
      };
    });

    let outShape: number[] = [];
    let outDtype = infos[0]?.dtype ?? AttributeType.UNDEFINED;

    switch (node.type) {
      case "Transpose":
        if (infos.length >= 1) {
          const [orgMatrix] = infos;
          const [N, M] = orgMatrix.shape;
          outShape = [M, N]
        }
        break;
      case "MatMul":
        if (infos.length >= 2) {
          const [a, b] = infos;
          if (a.shape.length === 2 && b.shape.length === 2) {
            outShape = [a.shape[0], b.shape[1]];
          } else {
            console.warn("MatMul with non-2D tensors:", a.shape, b.shape);
            outShape = []; // fallback
          }
        }
        break;

      case "Unsqueeze": {
        // Input 0 = tensor, Input 1 = axes (constant)
        //console.log("Unsq infos:", infos);

        const tensorShape = infos[0].shape ?? [];
        const axesNode = inputs[1]?.tryAs(TensorNode);

        let axes: number[] = axesNode?.constantValue?.dims ?? axesNode?.constantValue?.int32Data ?? axesNode?.constantValue?.int64Data.map(v => Number(v)) ?? axesNode?.constantValue?.stringData.map(v => Number(v)) ?? [];

        //console.log("Unsq tensorShape:", tensorShape);
        //console.log("Unsq axes:", axes);

        if (axes.length > 0) {
          outShape = [...tensorShape]; // may be [] if scalar
          axes.sort((a, b) => a - b).forEach(axis => {
            outShape.splice(axis, 0, 1);
          });
        }
        break;
      }

      case "Squeeze": {
        const inputShape = infos[0].shape;
        const axesNode = inputs[1]?.tryAs(TensorNode);
        const axes = axesNode?.constantValue?.int64Data?.map(Number);
        if (!axes || axes.length === 0) {
          // If no axes provided, remove all dims of size 1
          outShape = inputShape.filter(dim => dim !== 1);
        } else {
          const axisSet = new Set(axes);
          outShape = inputShape.filter((dim, idx) => !axisSet.has(idx) || dim !== 1);
        }
        break;
      }

      // ScatterElements preserves the shape of the inputs (default behaviour)
      case "Gather": {
        const dataShape = infos[0].shape ?? [];
        const indicesShape = infos[1].shape ?? [];
        const axis = node.getAttributes["axis"] ?? 0;

        //console.log("Gather dataShape:", dataShape);
        //console.log("Gather indicesShape:", indicesShape);
        //console.log("Gather axis:", axis);

        if (dataShape.length >= axis) {
          outShape = [
            ...dataShape.slice(0, axis),
            ...indicesShape,
            ...dataShape.slice(axis + 1),
          ];
          //console.log("Gather outShape:", outShape);
        } else {
          console.warn(`Invalid axis ${axis} for data shape [${dataShape}]`);
        }
        break;
      }

      case "Reshape":
        // Input 0 = tensor, Input 1 = target shape
        const shapeInput = inputs[1]?.tryAs(TensorNode);
        const shapeProto = shapeInput?.constantValue;
        //console.log("Reshape shapeProto:", shapeProto);
        //console.log("Reshape outShape:", shapeProto?.int64Data ? shapeProto.int64Data : "NO DATA");
        if (shapeProto?.int64Data) {
          outShape = Array.from(shapeProto.int64Data.map(n => Number(n)));
        }
        break;

      case "Transpose": {
        const inputShape = infos[0].shape;
        const perm = node.getAttributes["perm"] ?? inputShape.map((_, i) => i); // default: reverse dims
        outShape = perm.map((p: number) => inputShape[p] ?? 1);
        break;
      }

      case "Concat": {
        const axis = node.getAttributes["axis"] ?? 0;
        const inputShapes = infos.map(i => i.shape);
        if (inputShapes.length === 0) break;
        const refShape = inputShapes.find(s => s.length) ?? [];

        outShape = [...refShape];
        outShape[axis] = inputShapes.reduce((sum, s) => sum + (s[axis] ?? 0), 0);
        break;
      }

      case "Flatten": {
        const inputShape = infos[0].shape;
        const axis = node.getAttributes["axis"] ?? 1;

        const d0 = inputShape.slice(0, axis).reduce((a, b) => a * b, 1);
        const d1 = inputShape.slice(axis).reduce((a, b) => a * b, 1);
        outShape = [d0, d1];
        break;
      }

      case "Expand": {
        const shapeInput = inputs[1]?.tryAs(TensorNode);
        const targetShape = shapeInput?.constantValue?.int64Data?.map(Number);
        if (targetShape && targetShape.length > 0) {
          outShape = targetShape;
        }
        break;
      }

      default:
        // Maintain input shape
        //console.log(node.type, "infos:", infos);
        const first = infos.find(i => i.shape !== undefined);
        if (first) {
          outShape = first.shape;
          outDtype = first.dtype;
        }
        //console.log(node.type, "outshape:", outShape);
    }

    // Get current output TensorNodes
    const outputs = node.getOutgoers.targets;
    const outputTensors = outputs.filter(t => t.is(TensorNode));

    // Clean old outgoing edges (to avoid duplicates)
    node.getOutgoers.forEach(e => graph.getEdgeById(e.id).remove());

    // Reconnect updated output edges with correct shape/dtype
    for (const output of outputs) {
      graph.addEdge(node, output).init(new OnnxEdge.Builder(outDtype, outShape));
    }

    // Also update the tensor node itself
    for (const outTensor of outputTensors) {
      const tensorNode = outTensor.tryAs(TensorNode);
      if (tensorNode) {
        // Optional: only override if intermediate or undefined
        if (tensorNode.type === "intermediate" || tensorNode.shape?.length === 0) {
          tensorNode.setShape(outShape);
          tensorNode.setLiteralType(outDtype);
        }
      }
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