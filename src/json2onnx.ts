import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';
import { fileURLToPath } from 'url';

export async function json2onnx(jsonFilePath: string, outputOnnxPath: string): Promise<void> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const protoPath = path.join(__dirname, '../../out/src/Onnx/onnx.proto');

  try {
    // Load the ONNX protobuf definition
    const root = await protobuf.load(protoPath);
    const ModelProto = root.lookupType('onnx.ModelProto');

    if (path.extname(jsonFilePath) !== '.json') {
      throw new Error('The specified file is not a JSON file. Please provide a valid .json file.');
    }

    const jsonText = fs.readFileSync(jsonFilePath, 'utf-8');
    const jsonData = JSON.parse(jsonText);

    // Validate essential fields
    const defaultFields = {
      ir_version: 8,                    // Use current minimum ONNX IR version
      opset_import: [{ domain: '', version: 13 }],
      producer_name: 'onnx-flow',
      producer_version: '0.1.0',
      model_version: 1,
    };

    // Fill in any missing required fields
    const completeJson = {
      ...defaultFields,
      ...jsonData,
      graph: {
        name: jsonData.graph?.name ?? 'default_graph',
        ...jsonData.graph,
      }
    };

    // Validate against the protobuf schema
    const errMsg = ModelProto.verify(completeJson);
    if (errMsg) {
      throw new Error('Validation error: ' + errMsg);
    }

    // Create the ONNX model
    const message = ModelProto.create(completeJson);
    const buffer = ModelProto.encode(message).finish();

    fs.writeFileSync(outputOnnxPath, buffer);
    console.log(`ONNX model successfully written to ${outputOnnxPath}`);
  } catch (error) {
    console.error('Failed to convert JSON to ONNX:');
    if (error instanceof Error) {
      console.error('Message:', error.message);
    } else {
      console.error(error);
    }
    throw error;
  }
}
