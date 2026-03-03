import fs from "fs";
import path from "path";
import protobuf from "protobufjs";
import Long from "long";
import { fileURLToPath } from "url";
import type {
    RawOnnxAttribute,
    RawOnnxGraph,
    RawOnnxModel,
    RawOnnxNode,
} from "./Onnx/OnnxTypes.js";
import { BASE_TEN } from "./Onnx/Utils.js";

/**
 * Toggle strict behavior for Reshape shape constants:
 * - false (default): first null -> -1; additional nulls -> 0 (copy dim)
 * - true: throw when there are 2+ nulls
 */
const STRICT_RESHAPE_NULLS = false as boolean;

const OPSET = 19;
const IR_VERSION = 9;

const MODEL_VERSION = 1;

/**
 * Recursively traverses an object and converts any { type: 'Buffer', data: [...] }
 * back into actual Node.js Buffers for protobuf compatibility.
 */
function fixBuffers(obj: unknown): unknown {
    if (Array.isArray(obj)) {
        return obj.map(fixBuffers);
    }

    if (obj !== undefined && obj !== null && typeof obj === "object") {
        const record = obj as Record<string, unknown>;
        if (
            "type" in record &&
            record["type"] === "Buffer" &&
            "data" in record &&
            Array.isArray(record["data"])
        ) {
            return Buffer.from(record["data"]);
        }
        for (const key of Object.keys(record)) {
            record[key] = fixBuffers(record[key]);
        }
    }

    return obj;
}

/**
 * Resilient pre-pass for Reshape shapes produced by Constant with INT64 tensors.
 *
 * Policy:
 * - If the int64Data contains:
 *   - 0 nulls: leave as-is.
 *   - 1 null: set it to -1 (infer that dimension).
 *   - 2+ nulls:
 *      * STRICT_RESHAPE_NULLS === true  -> throw (ask user to build shapes dynamically).
 *      * STRICT_RESHAPE_NULLS === false -> first null -> -1, remaining nulls -> 0 (copy-dim).
 *
 * Notes:
 * - We only touch Constant-fed shapes with explicit int64Data arrays (not rawData).
 * - This keeps within ONNX Reshape semantics: one -1 allowed, 0 means "copy from input".
 */
function fixSingleNullReshapeShapesInGraph(graph: RawOnnxGraph): void {
    const nodes: RawOnnxNode[] = graph.node ?? [];
    // Map output name → producer node
    const byOutput: Record<string, RawOnnxNode> = {};
    for (const n of nodes) for (const o of n.output ?? []) byOutput[o] = n;

    for (const n of nodes) {
        // --- Recurse into subgraphs first
        for (const a of n.attribute ?? []) {
            if (a.g) fixSingleNullReshapeShapesInGraph(a.g);
            if (Array.isArray(a.graphs))
                for (const sg of a.graphs) fixSingleNullReshapeShapesInGraph(sg);
        }

        if (n.opType !== "Reshape") continue;

        const shapeInput = n.input?.[1];
        if (shapeInput === undefined || !(shapeInput in byOutput)) continue;

        const shapeProducer = byOutput[shapeInput];
        if (shapeProducer.opType !== "Constant") continue;

        // Find Constant’s tensor attribute (commonly "value" or unnamed)
        const attrs = shapeProducer.attribute ?? [];
        const tensorAttr = attrs.find((a: RawOnnxAttribute) => a.t?.dataType === /* INT64 */ 7);
        const t = tensorAttr?.t;
        if (!t) continue;

        // Only handle explicit int64Data
        if (!Array.isArray(t.int64Data)) continue;

        const data = t.int64Data.map((v: unknown) => (v === "null" || v === undefined ? null : v));
        const nullIdxs: number[] = [];
        for (let i = 0; i < data.length; i++) if (data[i] == null) nullIdxs.push(i);

        if (nullIdxs.length === 0) continue;

        if (nullIdxs.length === 1) {
            const out = data.slice();
            out[nullIdxs[0]] = -1; // ONNX Reshape: one unknown → -1
            t.int64Data = out as (number | bigint)[];
            continue;
        }

        if (STRICT_RESHAPE_NULLS) {
            const cname = shapeProducer.name !== undefined ? shapeProducer.name : shapeInput;
            throw new Error(
                `Reshape shape Constant(${cname}) has ${nullIdxs.length} unknown dims. ` +
                    `ONNX allows only one -1. Build shape dynamically with Shape/Gather/Concat.`,
            );
        } else {
            // Heuristic: first unknown → -1, the rest → 0 (copy-dim)
            const out = data.slice();
            out[nullIdxs[0]] = -1;
            for (let k = 1; k < nullIdxs.length; k++) out[nullIdxs[k]] = 0;
            const cname = shapeProducer.name !== undefined ? shapeProducer.name : shapeInput;
            console.warn(
                `[json2onnx] Reshape(${n.name !== undefined ? n.name : ""}) shape Constant(${cname}) had ${nullIdxs.length} unknown dims; ` +
                    `converted first -> -1, others -> 0 (copy dim).`,
            );
            t.int64Data = out as (number | bigint)[];
        }
    }
}

// OLD entry point now delegates to the recursive walker
function fixSingleNullReshapeShapes(model: RawOnnxModel): void {
    const graph = model.graph;
    if (!graph) return;
    fixSingleNullReshapeShapesInGraph(graph);
}

// Coerce numeric-like strings to numbers for fields protobuf expects as ints/floats.
// Also normalizes common ONNX numeric array fields (ints, floats, dims, etc.).
export function coerceNumericFields(obj: unknown): unknown {
    if (obj == null) return obj;

    if (Array.isArray(obj)) {
        for (let i = 0; i < obj.length; i++) coerceNumericFields(obj[i]);
        return obj;
    }

    if (typeof obj !== "object") return obj;

    const intArrayKeys = new Set(["ints", "axes", "perm", "pads", "dims", "int64s"]);
    const floatArrayKeys = new Set(["floats"]);
    const intScalarKeys = new Set(["i", "axis", "group", "value", "size"]);
    const floatScalarKeys = new Set(["f"]);

    const tensorIntArrays = new Set(["int32Data", "int64Data", "uint64Data"]);
    const tensorFloatArrays = new Set(["floatData", "doubleData"]);

    // Helpers
    const toInt = (x: unknown) => {
        if (x == null) return 0;
        if (typeof x === "string") {
            const s = x.trim().toLowerCase();
            if (s === "" || s === "null" || s === "nan" || s === "undefined") return 0;
            const n = parseInt(x, BASE_TEN);
            return Number.isFinite(n) ? n : 0;
        }
        if (typeof x === "number") {
            return Number.isFinite(x) ? Math.trunc(x) : 0;
        }
        if (typeof x === "bigint") {
            const n = Number(x);
            return Number.isFinite(n) ? n : parseInt(x.toString(), BASE_TEN);
        }
        return 0;
    };

    const toFloat = (x: unknown) => {
        if (x == null) return 0;
        if (typeof x === "string") {
            const s = x.trim().toLowerCase();
            if (s === "" || s === "null" || s === "nan" || s === "undefined") return 0;
            const n = parseFloat(x);
            return Number.isFinite(n) ? n : 0;
        }
        if (typeof x === "number") return Number.isFinite(x) ? x : 0;
        if (typeof x === "bigint") return Number(x);
        return 0;
    };

    // Normalize scalar → array for tensor payloads
    const ensureArray = (v: unknown) => (Array.isArray(v) ? v : [v]);

    const record = obj as Record<string, unknown>;
    for (const k of Object.keys(record)) {
        const v = record[k];
        if (v == null) continue;

        // Known int[] fields
        if (intArrayKeys.has(k) && Array.isArray(v)) {
            record[k] = v.map(toInt);
            continue;
        }

        // Known float[] fields
        if (floatArrayKeys.has(k) && Array.isArray(v)) {
            record[k] = v.map(toFloat);
            continue;
        }

        // Known int scalar fields
        if (
            intScalarKeys.has(k) &&
            (typeof v === "string" || typeof v === "number" || typeof v === "bigint")
        ) {
            record[k] = toInt(v);
            continue;
        }

        // Known float scalar fields
        if (
            floatScalarKeys.has(k) &&
            (typeof v === "string" || typeof v === "number" || typeof v === "bigint")
        ) {
            record[k] = toFloat(v);
            continue;
        }

        // TensorProto payloads (accept scalar or array)
        if (tensorIntArrays.has(k)) {
            const arr = ensureArray(v);
            record[k] = arr.map(toInt);
            continue;
        }
        if (tensorFloatArrays.has(k)) {
            const arr = ensureArray(v);
            record[k] = arr.map(toFloat);
            continue;
        }

        // Recurse into nested objects (attributes, tensors, graphs, etc.)
        coerceNumericFields(v);
    }

    return obj;
}

export async function json2onnx(jsonFilePath: string, outputOnnxPath: string): Promise<void> {
    const __dirname = path.dirname(fileURLToPath(import.meta.url));
    const protoPath = path.join(__dirname, "../../out/src/Onnx/onnx.proto");

    try {
        // Make protobufjs accept Longs for int64/uint64 fields
        (protobuf.util as Record<string, unknown>)["Long"] = Long;
        protobuf.configure();

        // Load the ONNX protobuf definition
        const root = await protobuf.load(protoPath);
        const ModelProto = root.lookupType("onnx.ModelProto");

        if (path.extname(jsonFilePath) !== ".json") {
            throw new Error(
                "The specified file is not a JSON file. Please provide a valid .json file.",
            );
        }

        const jsonText = fs.readFileSync(jsonFilePath, "utf-8");
        const jsonData = JSON.parse(jsonText) as RawOnnxModel;

        const defaultFields = {
            ir_version: IR_VERSION,
            opset_import: [{ domain: "", version: OPSET }],
            producer_name: "onnx-flow",
            producer_version: "0.1.0",
            model_version: MODEL_VERSION,
        };

        const completeJson = {
            ...defaultFields,
            ...jsonData,
            graph: {
                name: jsonData.graph?.name ?? "default_graph",
                ...jsonData.graph,
            },
        };

        const fixedJson = fixBuffers(completeJson);

        // Resilient Reshape shape fix runs BEFORE numeric coercion
        fixSingleNullReshapeShapes(fixedJson!);

        const normalizedJson = coerceNumericFields(fixedJson);

        const errMsg = ModelProto.verify(normalizedJson!);
        if (errMsg !== null) {
            throw new Error("Validation error: " + errMsg);
        }

        const message = ModelProto.create(normalizedJson!);
        const buffer = ModelProto.encode(message).finish();

        fs.writeFileSync(outputOnnxPath, buffer);
        console.log(`ONNX model successfully written to ${outputOnnxPath}`);
    } catch (error) {
        console.error("Failed to convert JSON to ONNX:");
        if (error instanceof Error) {
            console.error("Message:", error.message);
        } else {
            console.error(error);
        }
        throw error;
    }
}
