export function convertCytoscapeGraphToOnnxModelProto(cytoGraph : any): any {
  const nodes = cytoGraph.elements.nodes;
  const edges = cytoGraph.elements.edges;
  
  console.log("Processing graph with direct implementation...");
  
  type nodeInputsMap = Record<string, string[]>;
  type nodeOutputsMap = Record<string, string[]>;

  const modelNodes = [];
  
  const nodeInputs: nodeInputsMap = {};
  const nodeOutputs: nodeOutputsMap = {};
  
  const inputNodes = [];
  const outputNodes = [];
  
  const operationTypeMap = {
    'Load': 'Load',
    'Store': 'Store',
    'Addition': 'Add',
    'Multiplication': 'Mul',
    'Subtraction': 'Sub',
    'Division': 'Div'
  };
  
  const allNodeIds: Set<string> = new Set();
  const modelInputs = [];
  const modelOutputs = [];

  for (const node of nodes) {
    if (node.data['__specs-onnx__tensor_node']) {
      const tensorSpec = node.data['__specs-onnx__tensor_node'];
      const tensorId = node.data.id;

      if (tensorSpec.type === "input") {
        inputNodes.push(tensorId);
      } else if (tensorSpec.type === "output") {
        outputNodes.push(tensorId);
      }
    }
  }

  for (const edge of edges) {
    if (edge.data['__specs-onnx__onnx_edge']) {
      const sourceId = edge.data.source;
      const targetId = edge.data.target;

      allNodeIds.add(sourceId);
      allNodeIds.add(targetId);

      if (!nodeOutputs[sourceId]) {
        nodeOutputs[sourceId] = [];
      }
      if (!nodeInputs[targetId]) {
        nodeInputs[targetId] = [];
      }

      nodeOutputs[sourceId].push(targetId);
      nodeInputs[targetId].push(sourceId);
    }
  }

  for (const id of allNodeIds) {
    const originalNode = nodes.find(n => n.data.id === id);
    const inputIds: string[] = nodeInputs[id] || [];
    const outputIds: string[] = nodeOutputs[id] || [];

    if (originalNode) {
      const opSpec = originalNode.data['__specs-onnx__operation_node'];
      const tensorSpec = originalNode.data['__specs-onnx__tensor_node'];

      if (opSpec && !originalNode.data['parent']) {
        const opType = operationTypeMap[opSpec.type] || opSpec.type;

        if (outputNodes.includes(nodeOutputs[id][0])) {
          modelNodes.push({
            opType: opType,
            input: inputIds,
            output: outputIds
          });
        } else {
          modelNodes.push({
            opType: opType,
            input: inputIds,
            output: [id]
          });
        }
      } else if (tensorSpec) {
        const tensorName = tensorSpec.name || id;
        const tensorType = tensorSpec.literalType;
        const tensorShape = tensorSpec.shape || [1];

        if (tensorSpec.type === "input") {
          modelInputs.push({
            name: tensorName,
            type: {
              tensorType: {
                elemType: tensorType,
                shape: {
                  dim: tensorShape.map(dim => ({ dimValue: dim }))
                }
              }
            }
          });
        } else if (tensorSpec.type === "output") {
          modelOutputs.push({
            name: tensorName,
            type: {
              tensorType: {
                elemType: tensorType,
                shape: {
                  dim: tensorShape.map(dim => ({ dimValue: dim }))
                }
              }
            }
          });
        }
      }
    }
  }

  return {
    irVersion: 8,
    opsetImport: [{ version: 17 }],
    graph: {
      name: "Graph",
      input: modelInputs,
      output: modelOutputs,
      node: modelNodes
    }
  };
}
