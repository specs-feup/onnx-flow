import fs from "fs";
import path from "path";
import protobuf from "protobufjs";
import { fileURLToPath } from "url";
import type { RawOnnxModel } from "../onnx/OnnxTypes.js";

export function onnx2json(onnxFilePath: string): Promise<RawOnnxModel> {
    const __dirname = path.dirname(fileURLToPath(import.meta.url));

    // Try possible locations for onnx.proto:
    const possiblePaths = [
        path.join(__dirname, "../../../out/src/onnx/onnx.proto"),
        path.join(__dirname, "../../src/onnx/onnx.proto"),
        path.join(__dirname, "../onnx/onnx.proto"), // Next to this file in src/ or out/src/
    ];

    const protoPath = possiblePaths.find((p) => fs.existsSync(p));

    if (protoPath === undefined) {
        return Promise.reject(
            `Error: Could not find 'onnx.proto'. Searched in: \n${possiblePaths.join("\n")}`,
        );
    }

    return new Promise((resolve, reject) => {
        // Load the ONNX protobuf definition
        protobuf.load(protoPath!, (err, root) => {
            if (err) {
                return reject("Error loading ONNX protobuf definition: " + err);
            }

            // Get the ModelProto message type
            if (!root) {
                return reject("Error: ONNX protobuf root is undefined.");
            }
            try {
                const ModelProto = root.lookupType("onnx.ModelProto");

                // Function to load and inspect the ONNX model
                function loadAndInspectModel(filePath: string) {
                    try {
                        // Check if the file exists and is a valid ONNX file
                        if (path.extname(filePath) !== ".onnx") {
                            return reject(
                                "The specified file is not an ONNX file. Please provide a valid .onnx file.",
                            );
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
                            reject("Error decoding ONNX model: " + error.message);
                        } else {
                            reject("Error decoding ONNX model: " + String(error));
                        }
                    }
                }

                // Load and inspect the ONNX model
                loadAndInspectModel(onnxFilePath);
            } catch (e) {
                reject("Error looking up ModelProto: " + e);
            }
        });
    });
}
