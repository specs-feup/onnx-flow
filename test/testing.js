import { InferenceSession, Tensor } from 'onnxruntime-web';
import { onnx2json } from '../onnx2json.js';


async function runModel() {
  // Load the ONNX model
  const modelPath = '../onnx_examples/MatMul.onnx';
  const session = await InferenceSession.create(modelPath);

  // Load the model specification JSON file
  const data = await onnx2json(modelPath);
  
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
  data.graph.input.forEach(input => {
    const shape = input.type.tensorType.shape.dim.map(dim => parseInt(dim.dimValue, 10));
    const elemType = getArrayType(input.type.tensorType.elemType);
    listOfInputs[input.name] = new Tensor(elemType, getRandomArray(shape, elemType), shape);
  });
  console.log('Generated random inputs:', listOfInputs);

  // Run the model with the generated inputs
  const output = await session.run(listOfInputs);
  console.log('Model output:', output);
}

runModel().catch(err => console.error(err));