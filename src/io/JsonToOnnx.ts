import fs from "fs";
import path from "path";
import type {
    RawOnnxAttribute,
    RawOnnxGraph,
    RawOnnxModel,
    RawOnnxNode,
} from "../onnx/ir/OnnxTypes.js";
import { DataType } from "../onnx/ir/OnnxTypes.js";
import { BASE_TEN } from "../onnx/utils/index.js";
import {
    DEFAULT_COMPAT_IR_VERSION,
    DEFAULT_COMPAT_OPSET,
    getOpsetMinIrVersion,
} from "../onnx/schema/Versioning.js";
import { OnnxProtobufCodec } from "./OnnxProtobufCodec.js";

/**
 * Toggle strict behavior for Reshape shape constants:
 * - false (default): first null -> -1; additional nulls -> 0 (copy dim)
 * - true: throw when there are 2+ nulls
 */
const STRICT_RESHAPE_NULLS = false as boolean;
const MODEL_VERSION = 1;

const INT_ARRAY_KEYS = new Set(["ints", "axes", "perm", "pads", "dims", "int64s"]);
const FLOAT_ARRAY_KEYS = new Set(["floats"]);
const INT_SCALAR_KEYS = new Set(["i", "axis", "group", "value", "size"]);
const FLOAT_SCALAR_KEYS = new Set(["f"]);
const TENSOR_INT_ARRAYS = new Set(["int32Data", "int64Data", "uint64Data"]);
const TENSOR_FLOAT_ARRAYS = new Set(["floatData", "doubleData"]);

/**
 * Recursively converts JSON-serialized `{ type: "Buffer", data: [...] }` objects
 * back into Node.js `Buffer` instances required by protobufjs encoding.
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
            return Buffer.from(record["data"] as number[]);
        }
        for (const key of Object.keys(record)) {
            record[key] = fixBuffers(record[key]);
        }
    }

    return obj;
}

/**
 * Pre-pass that repairs null entries inside Constant-fed `Reshape` shape tensors
 * (replacing the first unknown dim with `-1` and remaining ones with `0`).
 */
function fixSingleNullReshapeShapesInGraph(graph: RawOnnxGraph): void {
    const nodes: RawOnnxNode[] = graph.node ?? [];
    const byOutput: Record<string, RawOnnxNode> = {};
    for (const n of nodes) {
        for (const o of n.output ?? []) {
            byOutput[o] = n;
        }
    }

    for (const n of nodes) {
        for (const a of n.attribute ?? []) {
            if (a.g != null) fixSingleNullReshapeShapesInGraph(a.g);
            if (Array.isArray(a.graphs)) {
                for (const sg of a.graphs) fixSingleNullReshapeShapesInGraph(sg);
            }
        }

        if (n.opType !== "Reshape") continue;

        const shapeInput = n.input?.[1];
        if (shapeInput === undefined || !(shapeInput in byOutput)) continue;

        const shapeProducer = byOutput[shapeInput];
        if (shapeProducer.opType !== "Constant") continue;

        const attrs = shapeProducer.attribute ?? [];
        const tensorAttr = attrs.find((a: RawOnnxAttribute) => a.t?.dataType === DataType.INT64);
        const t = tensorAttr?.t;
        if (!t || !Array.isArray(t.int64Data)) continue;

        const data = t.int64Data.map((v: unknown) => (v === "null" || v === undefined ? null : v));
        const nullIdxs: number[] = [];
        for (let i = 0; i < data.length; i++) {
            if (data[i] == null) nullIdxs.push(i);
        }

        if (nullIdxs.length === 0) continue;

        if (nullIdxs.length === 1) {
            const out = data.slice();
            out[nullIdxs[0]] = -1;
            t.int64Data = out as (number | bigint)[];
            continue;
        }

        const cname = shapeProducer.name !== undefined ? shapeProducer.name : shapeInput;
        if (STRICT_RESHAPE_NULLS) {
            throw new Error(
                `Reshape shape Constant(${cname}) has ${nullIdxs.length} unknown dims. ` +
                    `ONNX allows only one -1. Build shape dynamically with Shape/Gather/Concat.`,
            );
        }

        const out = data.slice();
        out[nullIdxs[0]] = -1;
        for (let k = 1; k < nullIdxs.length; k++) out[nullIdxs[k]] = 0;
        console.warn(
            `[json2onnx] Reshape(${n.name !== undefined ? n.name : ""}) shape Constant(${cname}) had ${nullIdxs.length} unknown dims; ` +
                `converted first -> -1, others -> 0 (copy dim).`,
        );
        t.int64Data = out as (number | bigint)[];
    }
}

function toInt(x: unknown): number {
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
}

function toFloat(x: unknown): number {
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
}

/**
 * Recursively coerces numeric-like strings and bigints into JS numbers for fields
 * that protobufjs expects as ints/floats, normalizing scalar tensor payloads to arrays.
 */
export function coerceNumericFields(obj: unknown): unknown {
    if (obj == null) return obj;

    if (Array.isArray(obj)) {
        for (let i = 0; i < obj.length; i++) coerceNumericFields(obj[i]);
        return obj;
    }

    if (typeof obj !== "object") return obj;

    const record = obj as Record<string, unknown>;
    for (const k of Object.keys(record)) {
        const v = record[k];
        if (v == null) continue;

        if (INT_ARRAY_KEYS.has(k) && Array.isArray(v)) {
            record[k] = v.map(toInt);
            continue;
        }
        if (FLOAT_ARRAY_KEYS.has(k) && Array.isArray(v)) {
            record[k] = v.map(toFloat);
            continue;
        }
        if (
            INT_SCALAR_KEYS.has(k) &&
            (typeof v === "string" || typeof v === "number" || typeof v === "bigint")
        ) {
            record[k] = toInt(v);
            continue;
        }
        if (
            FLOAT_SCALAR_KEYS.has(k) &&
            (typeof v === "string" || typeof v === "number" || typeof v === "bigint")
        ) {
            record[k] = toFloat(v);
            continue;
        }
        if (TENSOR_INT_ARRAYS.has(k)) {
            record[k] = (Array.isArray(v) ? v : [v]).map(toInt);
            continue;
        }
        if (TENSOR_FLOAT_ARRAYS.has(k)) {
            record[k] = (Array.isArray(v) ? v : [v]).map(toFloat);
            continue;
        }

        coerceNumericFields(v);
    }

    return obj;
}

/**
 * Encodes a JSON ONNX model file back into binary `.onnx` protobuf format.
 */
export async function json2onnx(
    jsonFilePath: string,
    outputOnnxPath: string,
    options?: { targetOpset?: number; irVersion?: number; verbose?: boolean },
): Promise<void> {
    try {
        const modelProto = await OnnxProtobufCodec.loadModelProto();

        if (path.extname(jsonFilePath) !== ".json") {
            throw new Error(
                "The specified file is not a JSON file. Please provide a valid .json file.",
            );
        }

        const jsonText = fs.readFileSync(jsonFilePath, "utf-8");
        const jsonData = JSON.parse(jsonText) as RawOnnxModel;

        const defaultOpset = options?.targetOpset ?? DEFAULT_COMPAT_OPSET;
        const defaultIr =
            options?.irVersion ??
            Math.max(DEFAULT_COMPAT_IR_VERSION, getOpsetMinIrVersion(defaultOpset));
        const defaultFields = {
            ir_version: defaultIr,
            opset_import: [{ domain: "", version: defaultOpset }],
            producer_name: "onnx-flow",
            producer_version: "0.1.0",
            model_version: MODEL_VERSION,
        };

        const rawIr = jsonData.irVersion ?? jsonData.ir_version;
        const resolvedIr = options?.irVersion ?? (rawIr !== undefined ? Number(rawIr) : defaultIr);
        const rawOpsets = jsonData.opsetImport ?? jsonData.opset_import;
        const resolvedOpsets =
            options?.targetOpset !== undefined
                ? [{ domain: "", version: options.targetOpset }]
                : rawOpsets;

        const completeJson: RawOnnxModel = {
            ...defaultFields,
            ...jsonData,
            irVersion: resolvedIr,
            ...(resolvedOpsets !== undefined ? { opsetImport: resolvedOpsets } : {}),
            graph: {
                name: jsonData.graph?.name ?? "default_graph",
                ...jsonData.graph,
            },
        };

        const fixedJson = fixBuffers(completeJson) as RawOnnxModel;
        if (fixedJson.graph !== undefined) {
            fixSingleNullReshapeShapesInGraph(fixedJson.graph);
        }

        const normalizedJson = coerceNumericFields(fixedJson) as Record<string, unknown>;

        const errMsg = modelProto.verify(normalizedJson);
        if (errMsg !== null) {
            throw new Error("Validation error: " + errMsg);
        }

        const message = modelProto.create(normalizedJson);
        const buffer = modelProto.encode(message).finish();

        fs.writeFileSync(outputOnnxPath, buffer);
        if (options?.verbose === true) {
            console.log(`ONNX model successfully written to ${outputOnnxPath}`);
        }
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
