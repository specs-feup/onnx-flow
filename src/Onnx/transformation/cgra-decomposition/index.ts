import OnnxGraph from "../../OnnxGraph.js";
import { divideMatMul } from "./CgraDecomposition.js";

export default function transformForCgra(g: OnnxGraph.Class): OnnxGraph.Class {
  let anyDivided = true;

  while (anyDivided) {
    anyDivided = false;
    const operationNodes = g.getOperationNodes().toArray();

    for (const node of operationNodes) {
      if (node.type === "MatMul" && divideMatMul(node, g)) {
        anyDivided = true;
      }
    }
  }

  return g;
}