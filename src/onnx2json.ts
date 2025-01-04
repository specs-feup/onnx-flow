import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';
import { fileURLToPath } from 'url';

export function onnx2json(onnxFilePath: string): Promise<any> {
    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const isProduction = process.env.NODE_ENV === 'production';
    const protoPath = isProduction
    ? path.join(__dirname, './Onnx/onnx.proto')  // For built files
    : path.join(__dirname, '../../src/Onnx/onnx.proto');  // For source files

    return new Promise((resolve, reject) => {
        // Load the ONNX protobuf definition
        protobuf.load(protoPath, (err, root) => {
            if (err) {
                return reject('Error loading ONNX protobuf definition: ' + err);
            }

            // Get the ModelProto message type
            if (!root) {
                return reject('Error: ONNX protobuf root is undefined.');
            }
            const ModelProto = root.lookupType('onnx.ModelProto');

            // Function to load and inspect the ONNX model
            function loadAndInspectModel(filePath: string) {
                try {
                    // Check if the file exists and is a valid ONNX file
                    if (path.extname(filePath) !== '.onnx') {
                        return reject('The specified file is not an ONNX file. Please provide a valid .onnx file.');
                    }

                    // Read the ONNX model file
                    const buffer = fs.readFileSync(filePath);
                    const model = ModelProto.decode(buffer);
                    const modelJson = ModelProto.toObject(model, {
                        longs: String,
                        enums: String,
                        defaults: true,
                        arrays: true,
                    });

                    // Resolve with the model JSON
                    resolve(modelJson);
                } catch (error) {
                    if (error instanceof Error) {
                        reject('Error loading ONNX model: ' + error.message);
                    } else {
                        reject('Error loading ONNX model: ' + String(error));
                    }
                }
            }

            // Load and inspect the ONNX model
            loadAndInspectModel(onnxFilePath);
        });
    });
}