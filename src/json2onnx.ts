import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';
import { fileURLToPath } from 'url';

class Color {
    static BLACK = '\x1b[30m';
    static RED = '\x1b[31m';
    static GREEN = '\x1b[32m';
    static YELLOW = '\x1b[33m';
    static BLUE = '\x1b[34m';
    static MAGENTA = '\x1b[35m';
    static CYAN = '\x1b[36m';
    static WHITE = '\x1b[37m';
    static COLOR_DEFAULT = '\x1b[39m';
    static BOLD = '\x1b[1m';
    static UNDERLINE = '\x1b[4m';
    static INVISIBLE = '\x1b[8m';
    static REVERSE = '\x1b[7m';
    static BG_BLACK = '\x1b[40m';
    static BG_RED = '\x1b[41m';
    static BG_GREEN = '\x1b[42m';
    static BG_YELLOW = '\x1b[43m';
    static BG_BLUE = '\x1b[44m';
    static BG_MAGENTA = '\x1b[45m';
    static BG_CYAN = '\x1b[46m';
    static BG_WHITE = '\x1b[47m';
    static BG_DEFAULT = '\x1b[49m';
    static RESET = '\x1b[0m';
}

class JsonToOnnxConverter {
    static convert(jsonFilePath: string, outputOnnxFilePath: string): Promise<void> {
        const __dirname = path.dirname(fileURLToPath(import.meta.url));
        const protoPath = path.join(__dirname, '../../public', 'onnx.proto'); // Adjusted path

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

                // Function to load and convert the JSON model
                function loadAndConvertJson(filePath: string) {
                    try {
                        // Read the JSON model file
                        const jsonContent = fs.readFileSync(filePath, 'utf-8');
                        const jsonObject = JSON.parse(jsonContent);

                        // Ensure irVersion and other similar fields are integers
                        if (jsonObject.irVersion) {
                            jsonObject.irVersion = parseInt(jsonObject.irVersion, 10);
                        }
                        if (jsonObject.opsetImport) {
                            jsonObject.opsetImport = jsonObject.opsetImport.map((opset: any) => ({
                                ...opset,
                                version: parseInt(opset.version, 10)
                            }));
                        }

                        // Ensure dimValue fields are integers for input and output
                        const ensureDimValueIntegers = (tensor: any) => {
                            if (tensor.type && tensor.type.tensorType && tensor.type.tensorType.shape && tensor.type.tensorType.shape.dim) {
                                tensor.type.tensorType.shape.dim.forEach((dim: any) => {
                                    if (dim.dimValue) {
                                        dim.dimValue = parseInt(dim.dimValue, 10);
                                    }
                                });
                            }
                        };

                        if (jsonObject.graph) {
                            if (jsonObject.graph.input) {
                                jsonObject.graph.input.forEach(ensureDimValueIntegers);
                            }
                            if (jsonObject.graph.output) {
                                jsonObject.graph.output.forEach(ensureDimValueIntegers);
                            }
                            if (jsonObject.graph.node) {
                                jsonObject.graph.node.forEach((node: any) => {
                                    if (node.attribute) {
                                        node.attribute.forEach((attr: any) => {
                                            if (attr.ints) {
                                                attr.ints = attr.ints.map((val: any) => parseInt(val, 10));
                                            }
                                            if (attr.floats) {
                                                attr.floats = attr.floats.map((val: any) => parseFloat(val));
                                            }
                                        });
                                    }
                                });
                            }
                        }

                        // Convert JSON to ONNX model
                        const errMsg = ModelProto.verify(jsonObject);
                        if (errMsg) {
                            throw new Error(errMsg);
                        }

                        const model = ModelProto.fromObject(jsonObject);
                        const buffer = ModelProto.encode(model).finish();

                        // Write the ONNX model to the output file
                        fs.writeFileSync(outputOnnxFilePath, buffer);

                        // Resolve the promise
                        resolve();
                    } catch (error) {
                        if (error instanceof Error) {
                            reject('Error converting JSON to ONNX model: ' + error.message);
                        } else {
                            reject('Error converting JSON to ONNX model: ' + String(error));
                        }
                    }
                }

                // Load and convert the JSON model
                loadAndConvertJson(jsonFilePath);
            });
        });
    }
}

export default JsonToOnnxConverter;