import OnnxGraph from "./Onnx/OnnxGraph.js";
import { AttributeProto, AttributeType, DataType } from "./Onnx/OnnxTypes.js";

const IR_VERSION = 9;
const OPSET_IMPORT = 17;

export function convertFlowGraphToOnnxJson(graph: OnnxGraph.Class, name?: String): any {
  const modelInputs: any[] = [];
  const modelOutputs: any[] = [];
  const modelInitializers = convertInitializers(graph);
  const modelNodes: any[] = [];

  function sanitizeTensor(tensor: any): any {
    // TensorProto keys except name and rawData (handled differently)
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
        } else if (
          key.endsWith("Data") &&
          tensor.rawData &&
          !value.length // only override if missing
        ) {
          // Try decoding rawData if value array is empty
          const dtype = tensor.dataType ?? DataType.INT64;
          const buffer = Buffer.from(tensor.rawData.data);
          if (dtype === DataType.INT64) { 
            sanitized.int64Data = [];
            for (let i = 0; i < buffer.length; i += 8) {
              sanitized.int64Data.push(buffer.readBigInt64LE(i));
            }
          } else if (dtype === DataType.INT32) {
            sanitized.int32Data = [];
            for (let i = 0; i < buffer.length; i += 4) {
              sanitized.int32Data.push(buffer.readInt32LE(i));
            }
          } else if (dtype === DataType.FLOAT) {
            sanitized.floatData = [];
            for (let i = 0; i < buffer.length; i += 4) {
              sanitized.floatData.push(buffer.readFloatLE(i));
            }
          }
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
      if (node.type !== "initializer") continue;
      const original = node.originalInitializer;

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
      if (tensorNode.isConstant()) {
        const original = tensorNode.constantValue!;
        const serialized = sanitizeTensor({ ...original, name: tensorNode.id });

        const attrs: AttributeProto[] = [{
          name: "value",
          type: AttributeType.TENSOR,
          t: serialized
        }];

        // Include any other preserved attributes
        for (const attr of tensorNode.extraAttrs ?? []) {
          attrs.push(attr);
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
        attr.type = AttributeType.INTS;
      } else if (typeof value === "number") {
        attr.i = value;
        attr.type = AttributeType.INT;
      } else if (typeof value === "string") {
        attr.s = value;
        attr.type = AttributeType.STRING;
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
          type: AttributeType.GRAPH,
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
    irVersion: IR_VERSION,
    opsetImport: [{ version: OPSET_IMPORT }],
    graph: {
      name: name ?? "Graph",
      initializer: modelInitializers,
      node: modelNodes,
      input: modelInputs,
      output: modelOutputs,
    },
  };
}