export function convertCytoscapeGraphToOnnxModelProto(cytoGraph : any): any {
  const nodes = cytoGraph.elements.nodes;
  const edges = cytoGraph.elements.edges;
  
  console.log("Processing graph with direct implementation...");
  
  type nodeIdsMap = Record<string, any>;

  const modelNodes = [];
  
  const nodeIds: nodeIdsMap = {}; 
  
  const modelInputs = [];
  const modelOutputs = [];
  const modelInitializers = [];

  const specsOnnxNodeTypes = {
    Tensor: '__specs-onnx__tensor_node',
    Variable: '__specs-onnx__variable_node',
    Operation: '__specs-onnx__operation_node',
    Constant: '__specs-onnx__constant_node'
  };

  for (const node of nodes) {
    if (node.data[specsOnnxNodeTypes.Tensor] || (node.data[specsOnnxNodeTypes.Operation] && !node.data.parent)) {
      continue;
    }

    if (node.data[specsOnnxNodeTypes.Constant]) {
      nodeIds[node.data.id] = {
        value: node.data[specsOnnxNodeTypes.Constant].value
      };
    } else if (node.data[specsOnnxNodeTypes.Variable]) {
      nodeIds[node.data.id] = {
        literalType: node.data[specsOnnxNodeTypes.Variable].literalType,
        name: node.data[specsOnnxNodeTypes.Variable].name,
        type: node.data[specsOnnxNodeTypes.Variable].type
      };
    } else if (node.data[specsOnnxNodeTypes.Operation]) {
      nodeIds[node.data.id] = {
        type: node.data[specsOnnxNodeTypes.Operation].type
      };
    }
  }

  for (const [key, value] of Object.entries(nodeIds)) {
    if (value.type && value.type === 'input') {
      modelInputs.push({
        name: key,
        type: {
          tensorType: {
            elemType: value.literalType,
            shape: {
              dim: [
                {
                  dimValue: 1
                },
                {
                  dimValue: 1
                }
              ]
            }
          }
        } 
      });
    } else if (value.type && value.type === 'output') {
      modelOutputs.push({
        name: key,
        type: {
          tensorType: {
            elemType: value.literalType,
              shape: {
              dim: [
                {
                  dimValue: 1
                },
                {
                  dimValue: 1
                }
              ]
            }
          }
        }
      });
    }
  }

  for (const edge of edges) {
    const sourceNode = edge.data.source;
    const targetNode = edge.data.target;

    if (!nodeIds[sourceNode] || !nodeIds[targetNode]) {
      continue;
    }

    modelNodes.push({
      opType: nodeIds[targetNode].type || 'Unknown',
      input: [sourceNode],
      output: [targetNode]
    });

    if (nodeIds[sourceNode].value) {
      modelNodes.push({
        opType: 'Constant',
        input: [],
        output: [sourceNode],
        attribute: [
          {
            name: 'value_int',
            type: 2,
            i: nodeIds[sourceNode].value
          }
        ]
      })
    }
  }

  const tempModelNodes = [];
  let already = false;

  for (const node of modelNodes) {
    already = false;
    for (const tempNode of tempModelNodes) {
      if (tempNode.output[0] === node.output[0]) {
        tempNode.input.push(...node.input);
        tempNode.output = node.output;
        already = true;
        break;
      }
    }
    if (!already) {
      tempModelNodes.push(node);
      already = false;
    }
  }

  return {
    irVersion: 8,
    opsetImport: [{ version: 17 }],
    graph: {
      name: "Graph",
      input: modelInputs,
      output: modelOutputs,
      node: tempModelNodes,
      initializer: modelInitializers
    }
  };
}
