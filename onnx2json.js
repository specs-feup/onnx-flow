import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';

export function onnx2json(onnxFilePath) {
    return new Promise((resolve, reject) => {
        // Load the ONNX protobuf definition
        protobuf.load('onnx.proto', (err, root) => {
            if (err) {
                return reject('Error loading ONNX protobuf definition: ' + err);
            }

            // Get the ModelProto message type
            const ModelProto = root.lookupType('onnx.ModelProto');

            // Function to load and inspect the ONNX model
            function loadAndInspectModel(filePath) {
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
                    reject('Error loading ONNX model: ' + error.message);
                }
            }

            // Load and inspect the ONNX model
            loadAndInspectModel(onnxFilePath);
        });
    });
}

