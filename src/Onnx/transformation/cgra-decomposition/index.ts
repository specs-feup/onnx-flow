import OnnxGraph from "../../OnnxGraph.js";
import decomposeAdd from "./decomposers/Add.js";
import { decomposeMatMul } from "./decomposers/MatMul.js";
import decomposeRelu from "./decomposers/Relu.js";
import TensorSplitter from "./TensorSplitter.js";

const decomposers = {
    MatMul: decomposeMatMul,
    Add: decomposeAdd,
    Relu: decomposeRelu,
};

/**
 * @brief Decomposes the given ONNX graph for CGRA mapping.
 *
 * @attention This function does not ensure that the resulting graph is valid
 * for CGRA mapping, which will depend on the operations present in the graph
 * and the decompositions currently supported.
 *
 * @param g The ONNX graph to be transformed.
 * @returns The transformed ONNX graph.
 */
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
