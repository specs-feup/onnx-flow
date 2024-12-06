import { createGraph } from "./initGraph.js";
import fs from "fs";
import OnnxDotFormatter from "./Onnx/dot/OnnxDotFormatter.js";
import OnnxGraphTransformer from "./Onnx/transformation/LowLevelTransformation/LowLevelConversion.js";
import OnnxGraphOptimizer from "./Onnx/transformation/OptimizeForDimensions/OptimizeForDimensions.js";
import { generateCode } from "./codeGeneration.js";
// Load ONNX JSON data from a file
const onnxJsonData = JSON.parse(fs.readFileSync("../specs-onnx/json_examples/TestFail.json", "utf-8"));
// Create the graph using the ONNX JSON data
const graph = createGraph(onnxJsonData).apply(new OnnxGraphTransformer()).apply(new OnnxGraphOptimizer());
// Generate code
generateCode(graph);
/*
// Example: Print the graph nodes and edges
graph.nodes.forEach(node => {
    if (node.is(TensorNode)) {
        const valueNodeInstance = node.as(TensorNode);
        console.log(`Node ID: ${valueNodeInstance.id}`);
        console.log(`Type: ${valueNodeInstance.type}`);
        console.log(`Literal Type: ${valueNodeInstance.literalType}`);
        console.log(`Shape: ${valueNodeInstance.shape}`);
        console.log(`Element Type: ${valueNodeInstance.type}`);
        console.log(`Dimensions: ${valueNodeInstance.shape}`);
    } else {
        console.log(`Node ID: ${node.id}, ${node}`);
    }
});

// Print the graph edges
graph.edges.forEach(edge => {
    const edgeInstance = edge.tryAs(OnnxEdge);
    console.log(`Edge ID: ${edgeInstance?.id}`);
    console.log(`Source: ${edgeInstance?.source.id}`);
    console.log(`Target: ${edgeInstance?.target.id}`);
    console.log(`Literal Type: ${edgeInstance?.literalType}`);
    console.log(`Shape: ${edgeInstance?.shape}`);
});

*/
//rever el zezoca do matmul, fazer optimizações + codeGen
// Optionally, save the graph to a file using a formatter
const formatter = new OnnxDotFormatter();
graph.toFile(formatter, "graph.dot");
//# sourceMappingURL=test.js.map