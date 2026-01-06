import OnnxGraph from "../../OnnxGraph.js";
import decomposeAdd from "./decomposers/Add.js";
import { decomposeMatMul } from "./decomposers/MatMul.js";
import decomposeRelu from "./decomposers/Relu.js";
import TensorSplitter from "./TensorSplitter.js";

const decomposers = {
  "MatMul": decomposeMatMul,
  "Add": decomposeAdd,
  "Relu": decomposeRelu,
};

export default function transformForCgra(g: OnnxGraph.Class): OnnxGraph.Class {
  let anyDivided = true;
  const tensorSplitter = new TensorSplitter(g);

  while (anyDivided) {
    anyDivided = false;
    const operationNodes = g.getOperationNodes();

    for (const node of operationNodes) {
      const decomposer = decomposers[node.type];
      if (decomposer !== undefined && decomposer(node, g, tensorSplitter)) {
        anyDivided = true;
      }
    }
  }

  tensorSplitter.clearTensors();

  return g;
}