import { InferenceSession, Tensor } from 'onnxruntime-web';
import { onnx2json } from './onnx2json.js';
import { createGraph } from "./initGraph.js";
import OnnxGraphTransformer from "./Onnx/transformation/LowLevelTransformation/LowLevelConversion.js";
import OnnxGraphOptimizer from "./Onnx/transformation/OptimizeForDimensions/OptimizeForDimensions.js";
import { generateCode } from "./codeGeneration.js";
import { typeSizeMap } from './Onnx/transformation/Utilities.js';
import fs from 'fs';
import path from 'path';
import JsonToOnnxConverter from './json2onnx.js';
//AddAdd, AddAddAdd
async function runModel() {
    // Load the ONNX model or convert JSON to ONNX
    const inputFilePath = '../specs-onnx/onnx_examples/AddAddAdd.onnx';
    let onnxObject;
    if (inputFilePath.endsWith('.json')) {
        const tempOnnxFilePath = path.join(path.dirname(inputFilePath), 'temp.onnx');
        await JsonToOnnxConverter.convert(inputFilePath, tempOnnxFilePath);
        onnxObject = await onnx2json(tempOnnxFilePath);
        fs.unlinkSync(tempOnnxFilePath); // Clean up temporary file
    }
    else {
        onnxObject = await onnx2json(inputFilePath);
    }
    // Create an inference session
    const session = await InferenceSession.create(inputFilePath);
    // Helper function to generate random arrays of given shape and data type
    function getRandomArray(shape, elemType) {
        const size = shape.reduce((a, b) => a * b, 1); // Ensure size is calculated correctly
        switch (elemType) {
            case 'float32':
                return Float32Array.from({ length: size }, () => Math.random());
            case 'int32':
                return Int32Array.from({ length: size }, () => Math.floor(Math.random() * 100));
            case 'int64':
                return BigInt64Array.from({ length: size }, () => BigInt(Math.floor(Math.random() * 100)));
            case 'uint8':
                return Uint8Array.from({ length: size }, () => Math.floor(Math.random() * 256));
            case 'int8':
                return Int8Array.from({ length: size }, () => Math.floor(Math.random() * 256) - 128);
            case 'uint16':
                return Uint16Array.from({ length: size }, () => Math.floor(Math.random() * 65536));
            case 'int16':
                return Int16Array.from({ length: size }, () => Math.floor(Math.random() * 65536) - 32768);
            case 'uint32':
                return Uint32Array.from({ length: size }, () => Math.floor(Math.random() * 4294967296));
            case 'float64':
                return Float64Array.from({ length: size }, () => Math.random());
            default:
                throw new Error(`Unsupported data type: ${elemType}`);
        }
    }
    // Map ONNX tensor element type to corresponding array type
    function getArrayType(elemType) {
        switch (elemType) {
            case 1:
                return 'float32';
            case 6:
                return 'int32';
            case 7:
                return 'int64';
            case 2:
                return 'uint8';
            case 3:
                return 'int8';
            case 4:
                return 'uint16';
            case 5:
                return 'int16';
            case 12:
                return 'uint32';
            case 11:
                return 'float64';
            default:
                throw new Error(`Unsupported tensor element type: ${elemType}`);
        }
    }
    // Create a list of inputs dynamically based on the model's input specifications
    const listOfInputs = {};
    onnxObject.graph.input.forEach((input) => {
        const shape = input.type.tensorType.shape.dim.map((dim) => parseInt(dim.dimValue, 10));
        const elemType = getArrayType(input.type.tensorType.elemType);
        listOfInputs[input.name] = new Tensor(elemType, getRandomArray(shape, elemType), shape);
    });
    console.log('Generated random inputs:', listOfInputs);
    // Run the model with the generated inputs
    const output = await session.run(listOfInputs);
    console.log('Model output:', output);
    const outputTensorName = Object.keys(output)[0];
    const outputTensor = output[outputTensorName];
    // Create the OnnxGraph with transformations applied
    const graph = createGraph(onnxObject).apply(new OnnxGraphTransformer()).apply(new OnnxGraphOptimizer());
    // Generate code from the graph
    const generatedCode = generateCode(graph);
    console.log('Generated Code:', generatedCode);
    // Convert the randomly generated inputs to a format accepted by the generated code
    const formattedInputs = {};
    for (const [key, tensor] of Object.entries(listOfInputs)) {
        const elemType = onnxObject.graph.input.find((input) => input.name === key).type.tensorType.elemType;
        const displacement = typeSizeMap[elemType];
        formattedInputs[`tensor_${key}`] = {};
        for (let i = 0; i < tensor.dims.reduce((a, b) => a * b, 1); i++) {
            formattedInputs[`tensor_${key}`][i * displacement] = tensor.data[i];
        }
    }
    console.log('Formatted inputs for generated code:', formattedInputs);
    // Prepare the arguments for the generated function
    const inputKeys = Object.keys(formattedInputs);
    const inputValues = inputKeys.map(key => formattedInputs[key]);
    // Log the inputs being sent to the generated function
    console.log('Inputs being sent to the generated function:', inputKeys, inputValues);
    // Run the generated code
    const generatedFunction = new Function(...inputKeys, generatedCode + ` return onnxGraph(${inputKeys.join(', ')});`);
    const generatedOutput = generatedFunction(...inputValues);
    console.log('Generated Output:', generatedOutput);
    // Convert values to strings for comparison
    const outputData = Array.from(outputTensor.data).map(value => value.toString());
    const generatedOutputValues = Object.values(generatedOutput).map((value) => value.toString());
    // Compare the results
    const tolerance = 1e-6;
    const areEqual = outputData.every((value, index) => Math.abs(parseFloat(value) - parseFloat(generatedOutputValues[index])) < tolerance);
    console.log('Are the outputs equal?', areEqual);
}
runModel().catch(err => console.error(err));
//# sourceMappingURL=testing.js.map