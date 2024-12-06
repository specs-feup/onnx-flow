import { createGraph } from "./initGraph.js";
import fs from "fs";
import OnnxDotFormatter from "./Onnx/dot/OnnxDotFormatter.js";
import OnnxGraphTransformer from "./Onnx/transformation/LowLevelTransformation/LowLevelConversion.js";
import OnnxGraphOptimizer from "./Onnx/transformation/OptimizeForDimensions/OptimizeForDimensions.js";
import { generateCode } from "./codeGeneration.js";

// Load ONNX JSON data from a file
const onnxJsonData = JSON.parse(fs.readFileSync("../specs-onnx/json_examples/MatMulCase000.json", "utf-8"));

// Create the graph using the ONNX JSON data
const graph = createGraph(onnxJsonData).apply(new OnnxGraphTransformer()).apply(new OnnxGraphOptimizer());

// Generate code
console.log(generateCode(graph));

// Save the graph as a DOT file
const formatter = new OnnxDotFormatter();
graph.toFile(formatter, "graph.dot");   