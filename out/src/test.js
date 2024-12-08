import { createGraph } from "./initGraph.js";
import fs from "fs";
import OnnxDotFormatter from "./Onnx/dot/OnnxDotFormatter.js";
import OnnxGraphTransformer from "./Onnx/transformation/LowLevelTransformation/LowLevelConversion.js";
import OnnxGraphOptimizer from "./Onnx/transformation/OptimizeForDimensions/OptimizeForDimensions.js";
import { generateCode } from "./codeGeneration.js";
import { onnx2json } from "./onnx2json.js";
// Load ONNX JSON data from a file
//const inputFilePath = "../specs-onnx/json_examples/MatMulCase101.json";
const inputFilePath = "../specs-onnx/onnx_examples/MultiplyAndAdd.onnx";
let onnxObject;
if (inputFilePath.endsWith('.json')) {
    onnxObject = JSON.parse(fs.readFileSync(inputFilePath, 'utf8'));
}
else {
    onnxObject = await onnx2json(inputFilePath);
}
// Create the graph using the ONNX JSON data
const graph = createGraph(onnxObject).apply(new OnnxGraphTransformer()).apply(new OnnxGraphOptimizer());
// Generate code
console.log(generateCode(graph));
// Save the graph as a DOT file
const formatter = new OnnxDotFormatter();
graph.toFile(formatter, "graph.dot");
//# sourceMappingURL=test.js.map