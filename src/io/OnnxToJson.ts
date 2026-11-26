import fs from "fs";
import path from "path";
import type { RawOnnxModel } from "../onnx/OnnxTypes.js";
import { OnnxProtobufCodec } from "./OnnxProtobufCodec.js";

/**
 * Decodes a binary `.onnx` protobuf file into a plain JSON-compatible `RawOnnxModel`.
 */
export async function onnx2json(onnxFilePath: string): Promise<RawOnnxModel> {
    if (path.extname(onnxFilePath) !== ".onnx") {
        return Promise.reject(
            "The specified file is not an ONNX file. Please provide a valid .onnx file.",
        );
    }

    let modelProto;
    try {
        modelProto = await OnnxProtobufCodec.loadModelProto();
    } catch (error) {
        return Promise.reject("Error loading ONNX protobuf definition: " + String(error));
    }

    try {
        const buffer = fs.readFileSync(onnxFilePath);
        const model = modelProto.decode(buffer);
        return modelProto.toObject(model, {
            longs: String,
            enums: String,
            defaults: true,
            arrays: true,
        }) as RawOnnxModel;
    } catch (error) {
        if (error instanceof Error) {
            return Promise.reject("Error decoding ONNX model: " + error.message);
        }
        return Promise.reject("Error decoding ONNX model: " + String(error));
    }
}
