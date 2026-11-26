import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import protobuf from "protobufjs";
import Long from "long";

/**
 * Shared ONNX protobuf schema resolver and loader used by both `OnnxToJson`
 * and `JsonToOnnx`. Works seamlessly whether executed from uncompiled `src/io/`
 * or compiled `out/src/io/`.
 */
export class OnnxProtobufCodec {
    private static cachedModelProto: protobuf.Type | undefined;
    private static longConfigured = false;

    /**
     * Resolves the filesystem path to `onnx.proto` by checking standard sibling
     * and build output locations relative to this module.
     */
    public static resolveProtoPath(): string {
        const currentDir = path.dirname(fileURLToPath(import.meta.url));
        const possiblePaths = [
            path.join(currentDir, "../onnx/onnx.proto"),
            path.join(currentDir, "../../src/onnx/onnx.proto"),
            path.join(currentDir, "../../../out/src/onnx/onnx.proto"),
            path.join(currentDir, "../../../src/onnx/onnx.proto"),
        ];

        const found = possiblePaths.find((candidate) => fs.existsSync(candidate));
        if (found === undefined) {
            throw new Error(
                `Error: Could not find 'onnx.proto'. Searched in: \n${possiblePaths.join("\n")}`,
            );
        }
        return found;
    }

    /**
     * Loads and caches the `onnx.ModelProto` protobufjs message type, ensuring
     * 64-bit `Long` support is registered once.
     */
    public static async loadModelProto(): Promise<protobuf.Type> {
        if (!OnnxProtobufCodec.longConfigured) {
            (protobuf.util as Record<string, unknown>)["Long"] = Long;
            protobuf.configure();
            OnnxProtobufCodec.longConfigured = true;
        }

        if (OnnxProtobufCodec.cachedModelProto !== undefined) {
            return OnnxProtobufCodec.cachedModelProto;
        }

        const protoPath = OnnxProtobufCodec.resolveProtoPath();
        const root = await protobuf.load(protoPath);
        const modelProto = root.lookupType("onnx.ModelProto");
        OnnxProtobufCodec.cachedModelProto = modelProto;
        return modelProto;
    }
}
