import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function findProjectRoot(startDir: string = __dirname): string {
    let cur = startDir;
    while (cur !== path.dirname(cur)) {
        if (fs.existsSync(path.join(cur, "package.json"))) {
            return cur;
        }
        cur = path.dirname(cur);
    }
    return startDir;
}

export function ensureProtoExists(): void {
    const root = findProjectRoot();
    const srcProto = path.resolve(root, "src/Onnx/onnx.proto");
    if (!fs.existsSync(srcProto)) return;

    // 1. Where compiled index.js looks (<root>/out/src/Onnx/onnx.proto)
    // 2. Where uncompiled src/json2onnx.ts looks when called in-process (<parent>/out/src/Onnx/onnx.proto)
    const targets = [
        path.resolve(root, "out/src/Onnx/onnx.proto"),
        path.resolve(root, "../out/src/Onnx/onnx.proto"),
    ];

    for (const target of targets) {
        if (!fs.existsSync(target)) {
            try {
                fs.mkdirSync(path.dirname(target), { recursive: true });
                fs.copyFileSync(srcProto, target);
            } catch (err: unknown) {
                // If another worker thread created or locked the file concurrently, ignore race condition
                const code = (err as { code?: string }).code;
                if (code !== "EBUSY" && code !== "EPERM" && code !== "EEXIST") {
                    throw err;
                }
            }
        }
    }
}

// Ensure protos exist before any worker thread runs
ensureProtoExists();
