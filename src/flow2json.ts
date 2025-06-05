import OnnxGraph from "./Onnx/OnnxGraph.js";

export function convertFlowGraphToOnnxJson(graph: OnnxGraph.Class, name?: String): any {
  const modelInputs: any[] = [];
  const modelOutputs: any[] = [];
  const modelInitializers = convertInitializers(graph);
  const modelNodes: any[] = [];

  function sanitizeTensor(tensor: any): any {
    const allowedKeys = [
      "dataType", "dims", "floatData", "int32Data", "stringData", "int64Data",
      "doubleData", "uint64Data", "externalData"
    ];

    const sanitized: any = { name: tensor.name };

    for (const key of allowedKeys) {
      const value = tensor[key];
      if (value !== undefined && value !== null) {
        if (Array.isArray(value)) {
          sanitized[key] = value.map(v =>
            typeof v === "string" ? Number(v) : v
          );
        } else {
          sanitized[key] = value;
        }
      }
    }

    // Special handling for rawData
    if (tensor.rawData && Buffer.isBuffer(tensor.rawData) && tensor.rawData.length > 0) {
      sanitized.rawData = {
        type: "Buffer",
        data: Array.from(tensor.rawData),
      };
    }

    return sanitized;
  }


  function convertInitializers(graph: OnnxGraph.Class): any[] {
    const initializers: any[] = [];

    for (const node of graph.getTensorNodes()) {
      const meta = node.data["__specs-onnx__tensor_node"];
      if (meta?.type !== "initializer") continue;

      const original = node.data["__specs-onnx__initializer_data"];
      if (!original) {
        console.warn(`[Convert] Missing original initializer data for '${node.id}'`);
        continue;
      }

      const serialized = sanitizeTensor({ ...original, name: node.id });

      initializers.push(serialized);
    }

    return initializers;
  }



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

    for (const tensorNode of graph.getTensorNodes()) {
      if (tensorNode.data["__specs-onnx__constant_value"]) {
        const original = tensorNode.data["__specs-onnx__constant_value"];
        const serialized = sanitizeTensor({ ...original, name: tensorNode.id });

        const attrs = [{
          name: "value",
          type: 4, //TENSOR TYPE
          t: serialized
        }];

        // Include any other preserved attributes
        for (const key of Object.keys(tensorNode.data)) {
          if (key.startsWith("__specs-onnx__constant_attr_")) {
            const attr = tensorNode.data[key];
            attrs.push(attr);
          }
        }

        modelNodes.push({
          opType: "Constant",
          input: [],
          output: [tensorNode.id],
          attribute: attrs,
        });
      }
    }


  for (const opNode of graph.getOperationNodes()) {
    const opType = opNode.type ?? "UnknownOp";
    const inputs = opNode.getInputs().map(n => n.id);
    const outputs = opNode.getOutgoers.targets.toArray().map(n => n.id);


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

    if (opType === "Loop") {
      const bodyGraph = opNode.getBodySubgraph?.();
      const bodyGraphJson = bodyGraph ? convertFlowGraphToOnnxJson(bodyGraph, "loop_body").graph : null;

      const filteredAttrs = baseAttrs.filter(attr => attr.name !== "body");

      if (bodyGraphJson) {
        // Include valueInfo in the body subgraph if missing (helps Netron)
        const valueInfo = (bodyGraphJson.output ?? []).map((out: any) => ({
          name: out.name,
          type: out.type,
        }));

        filteredAttrs.push({
          name: "body",
          type: 5,
          g: {
            ...bodyGraphJson,
            valueInfo,
          },
        });
      }

      modelNodes.push({
        opType,
        input: inputs,
        output: outputs,
        attribute: filteredAttrs,
      });
    } else {
      modelNodes.push({
        opType,
        input: inputs,
        output: outputs,
        attribute: baseAttrs,
      });
    }
  }

  return {
    irVersion: 9,
    opsetImport: [{ version: 17 }],
    graph: {
      name: name ?? "Graph",
      initializer: modelInitializers,
      node: modelNodes,
      input: modelInputs,
      output: modelOutputs,
    },
  };
}