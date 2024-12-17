import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';
import { fileURLToPath } from 'url';

export function json2onnx(jsonFilePath : string, outputOnnxPath : string) {
    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const protoPath = path.join(__dirname, '../../src/Onnx', 'onnx.proto');

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

            // Function to encode and save the ONNX model
            function encodeAndSaveModel(jsonPath : string, outputPath : string) {
                try {
                    // Check if the input file exists and is a valid JSON file
                    if (path.extname(jsonPath) !== '.json') {
                        return reject('The specified file is not a JSON file. Please provide a valid .json file.');
                    }

                    // Read and parse the JSON file
                    const jsonData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

                    // Verify that the JSON data matches the ONNX schema
                    const errMsg = ModelProto.verify(jsonData);
                    if (errMsg) {
                        return reject('Error verifying JSON model data: ' + errMsg);
                    }

                    // Create a ModelProto message from the JSON data
                    const message = ModelProto.create(jsonData);

                    // Encode the message to a buffer
                    const buffer = ModelProto.encode(message).finish();

                    // Write the buffer to the output ONNX file
                    fs.writeFileSync(outputPath, buffer);

                    resolve(`ONNX model successfully written to ${outputPath}`);
                } catch (error) {
                    if (error instanceof Error) {
                        reject('Error converting JSON to ONNX: ' + error.message);
                    } else {
                        reject('Error converting JSON to ONNX: ' + String(error));
                    }
                }
            }

            // Encode and save the ONNX model
            encodeAndSaveModel(jsonFilePath, outputOnnxPath);
        });
    });
}