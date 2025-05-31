import OnnxGraph from "./Onnx/OnnxGraph.js";
import TensorNode from "./Onnx/TensorNode.js";
import OperationNode from "./Onnx/OperationNode.js";
import ConstantNode from "./Onnx/ConstantNode.js";
import VariableNode from "./Onnx/VariableNode.js";

export function convertFlowGraphToOnnxJson(graph: OnnxGraph.Class): any {
  const nodeMap: Record<string, any> = {};
  const modelInputs: any[] = [];
  const modelOutputs: any[] = [];
  const modelInitializers: any[] = [];
  const modelNodes: any[] = [];

  /*
  // Phase 1: Register all nodes
  for (const node of graph.nodes) {
    const id = node.id;

    if (node instanceof TensorNode.Class) {
      nodeMap[id] = {
        type: "tensor",
        literalType: node.literalType,
        shape: node.shape,
      };
    } else if (node instanceof ConstantNode.Class) {
      nodeMap[id] = {
        type: "constant",
        value: node.value,
      };
    } else if (node instanceof VariableNode.Class) {
      nodeMap[id] = {
        type: "variable",
        literalType: node.literalType,
        name: node.name,
      };
    } else if (node instanceof OperationNode.Class) {
      console.log("HEREOP", node);
      nodeMap[id] = {
        type: "operation",
        opType: node.type ?? "UnknownOp",
        attributes: node.attributes ?? {},
      };
    }
  }
    */

  // Phase 1: Inputs & Outputs
  for (const node of graph.getInputTensorNodes()) {
    modelInputs.push({
      name: node.id,
      type: {
        tensorType: {
          elemType: node.literalType,
          shape: { dim: node.shape.map(d => ({ dimValue: d })) },
        },
      },
    });
  }

  for (const node of graph.getOutputTensorNodes()) {
    modelOutputs.push({
      name: node.id,
      type: {
        tensorType: {
          elemType: node.literalType,
          shape: { dim: node.shape.map(d => ({ dimValue: d })) },
        },
      },
    });
  }

  // Phase 2: Constant nodes
  for (const [id, meta] of Object.entries(nodeMap)) {
    if (meta.type === "constant") {
      modelNodes.push({
        opType: "Constant",
        input: [],
        output: [id],
        attribute: [
          {
            name: "value_int",
            type: 2,
            i: meta.value,
          },
        ],
      });
    }
  }

  // Phase 3: Operation nodes
  for (const opNode of graph.getOperationNodes()) {
    const opType = opNode.type ?? "UnknownOp";

    const inputs = opNode.getIncomers.sources.toArray().map(n => n.id);
    const outputs = opNode.getOutgoers.targets.toArray().map(n => n.id);

    let nodeEntry = {
      opType,
      input: inputs,
      output: outputs,
      attribute: Object.entries(opNode.attributes || {}).map(([name, value]) => {
        const attr: any = { name };
        if (Array.isArray(value)) {
          attr.ints = value;
          attr.type = 7;
        } else if (typeof value === "number") {
          attr.i = value;
          attr.type = 2;
        } else if (typeof value === "string") {
          attr.s = value;
          attr.type = 3;
        }
        return attr;
      }),
    };

    // Special handling for Loop
    if (opType === "Loop") {
      const bodyGraph = opNode.getBodySubgraph?.();
      const bodyGraphJson = bodyGraph ? convertFlowGraphToOnnxJson(bodyGraph).graph : null;

      const baseAttrs = Object.entries(opNode.attributes || {}).map(([name, value]) => {
        const attr: any = { name };
        if (Array.isArray(value)) {
          attr.ints = value;
          attr.type = 7;
        } else if (typeof value === "number") {
          attr.i = value;
          attr.type = 2;
        } else if (typeof value === "string") {
          attr.s = value;
          attr.type = 3;
        }
        return attr;
      });

      // Remove any attribute with name "body" to prevent duplicates
      const filteredAttrs = baseAttrs.filter(attr => attr.name !== "body");

      if (bodyGraphJson) {
        filteredAttrs.push({
          name: "body",
          type: 4, // GRAPH
          g: bodyGraphJson,
        });
      }

      nodeEntry = {
        opType,
        input: inputs,
        output: outputs,
        attribute: filteredAttrs,
      };
    }

    modelNodes.push(nodeEntry);
  }

  return {
    irVersion: 8,
    opsetImport: [{ version: 17 }],
    graph: {
      name: "Graph",
      input: modelInputs,
      output: modelOutputs,
      initializer: modelInitializers,
      node: modelNodes,
    },
  };
}