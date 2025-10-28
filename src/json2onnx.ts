import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';
import { fileURLToPath } from 'url';
import Long from 'long';

/**
 * Toggle strict behavior for Reshape shape constants:
 * - false (default): first null -> -1; additional nulls -> 0 (copy dim)
 * - true: throw when there are 2+ nulls
 */
const STRICT_RESHAPE_NULLS = false;

/**
 * Recursively traverses an object and converts any { type: 'Buffer', data: [...] }
 * back into actual Node.js Buffers for protobuf compatibility.
 */
function fixBuffers(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(fixBuffers);
  }

  if (obj && typeof obj === 'object') {
    if (obj.type === 'Buffer' && Array.isArray(obj.data)) {
      return Buffer.from(obj.data);
    }

    for (const key of Object.keys(obj)) {
      obj[key] = fixBuffers(obj[key]);
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
function fixSingleNullReshapeShapes(model: any): void {
  const graph = model?.graph;
  if (!graph) return;

  const nodes: any[] = graph.node ?? [];
  // Map output name → producer node
  const byOutput: Record<string, any> = {};
  for (const n of nodes) {
    for (const o of n.output ?? []) byOutput[o] = n;
  }

  for (const n of nodes) {
    if (n.opType !== 'Reshape') continue;

    const shapeInput = n.input?.[1];
    if (!shapeInput) continue;

    const shapeProducer = byOutput[shapeInput];
    if (!shapeProducer || shapeProducer.opType !== 'Constant') continue;

    // Find a tensor attribute on the Constant (usually 'value' or unnamed)
    const attrs = shapeProducer.attribute ?? [];
    const tensorAttr = attrs.find((a: any) => a?.t && a.t.dataType === /* INT64 */ 7);
    const t = tensorAttr?.t;
    if (!t) continue;

    // Only handle explicit int64Data (we won't decode rawData here)
    if (!Array.isArray(t.int64Data)) continue;

    // Normalize "null"/undefined to null for counting
    const data = t.int64Data.map((v: any) => (v === 'null' || v === undefined) ? null : v);
    const nullIdxs: number[] = [];
    for (let i = 0; i < data.length; i++) if (data[i] == null) nullIdxs.push(i);

    if (nullIdxs.length === 0) continue;

    if (nullIdxs.length === 1) {
      // Exactly one unknown -> infer (-1)
      const out = data.slice();
      out[nullIdxs[0]] = -1;
      t.int64Data = out;
      continue;
    }

    // 2+ nulls
    if (STRICT_RESHAPE_NULLS) {
      const cname = shapeProducer.name || shapeInput;
      throw new Error(
        `Reshape shape Constant(${cname}) has ${nullIdxs.length} unknown dims. ` +
        `ONNX allows only one -1. Build the shape dynamically with Shape/Gather/Concat.`
      );
    } else {
      // Heuristic fallback: first -> -1, rest -> 0 (copy from input)
      const out = data.slice();
      out[nullIdxs[0]] = -1;
      for (let k = 1; k < nullIdxs.length; k++) out[nullIdxs[k]] = 0;
      // Optional: warn (non-fatal)
      const cname = shapeProducer.name || shapeInput;
      // eslint-disable-next-line no-console
      console.warn(
        `[json2onnx] Reshape(${n.name || ''}) shape Constant(${cname}) had ${nullIdxs.length} unknown dims; ` +
        `converted first -> -1, others -> 0 (copy dim).`
      );
      t.int64Data = out;
    }
  }
}

// Coerce numeric-like strings to numbers for fields protobuf expects as ints/floats.
// Also normalizes common ONNX numeric array fields (ints, floats, dims, etc.).
export function coerceNumericFields(obj: any): any {
  if (obj == null) return obj;

  if (Array.isArray(obj)) {
    for (let i = 0; i < obj.length; i++) coerceNumericFields(obj[i]);
    return obj;
  }

  if (typeof obj !== 'object') return obj;

  const intArrayKeys = new Set(['ints', 'axes', 'perm', 'pads', 'dims', 'int64s']);
  const floatArrayKeys = new Set(['floats']);
  const intScalarKeys = new Set(['i', 'axis', 'group', 'value', 'size']);
  const floatScalarKeys = new Set(['f']);

  const tensorIntArrays = new Set(['int32Data', 'int64Data', 'uint64Data']);
  const tensorFloatArrays = new Set(['floatData', 'doubleData']);

  // Helpers
  const toInt = (x: any) => {
    if (typeof x === 'string') return x.trim() === '' ? 0 : parseInt(x, 10);
    if (typeof x === 'bigint') {
      const n = Number(x);
      return Number.isFinite(n) ? n : parseInt(x.toString(), 10);
    }
    return x;
  };
  const toFloat = (x: any) => {
    if (typeof x === 'string') return x.trim() === '' ? 0 : parseFloat(x);
    if (typeof x === 'bigint') return Number(x);
    return x;
  };

  // Normalize scalar → array for tensor payloads
  const ensureArray = (v: any) => (Array.isArray(v) ? v : [v]);

  for (const k of Object.keys(obj)) {
    const v = (obj as any)[k];
    if (v == null) continue;

    // Known int[] fields
    if (intArrayKeys.has(k) && Array.isArray(v)) {
      (obj as any)[k] = v.map(toInt);
      continue;
    }

    // Known float[] fields
    if (floatArrayKeys.has(k) && Array.isArray(v)) {
      (obj as any)[k] = v.map(toFloat);
      continue;
    }

    // Known int scalar fields
    if (intScalarKeys.has(k) && (typeof v === 'string' || typeof v === 'number' || typeof v === 'bigint')) {
      (obj as any)[k] = toInt(v);
      continue;
    }

    // Known float scalar fields
    if (floatScalarKeys.has(k) && (typeof v === 'string' || typeof v === 'number' || typeof v === 'bigint')) {
      (obj as any)[k] = toFloat(v);
      continue;
    }

    // TensorProto payloads (accept scalar or array)
    if (tensorIntArrays.has(k)) {
      const arr = ensureArray(v);
      (obj as any)[k] = arr.map(toInt);
      continue;
    }
    if (tensorFloatArrays.has(k)) {
      const arr = ensureArray(v);
      (obj as any)[k] = arr.map(toFloat);
      continue;
    }

    // Recurse into nested objects (attributes, tensors, graphs, etc.)
    coerceNumericFields(v);
  }

  return obj;
}

export async function json2onnx(jsonFilePath: string, outputOnnxPath: string): Promise<void> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const protoPath = path.join(__dirname, '../../out/src/Onnx/onnx.proto');

  try {
    // Make protobufjs accept Longs for int64/uint64 fields
    (protobuf.util as any).Long = Long;
    protobuf.configure();

    // Load the ONNX protobuf definition
    const root = await protobuf.load(protoPath);
    const ModelProto = root.lookupType('onnx.ModelProto');

    if (path.extname(jsonFilePath) !== '.json') {
      throw new Error('The specified file is not a JSON file. Please provide a valid .json file.');
    }

    const jsonText = fs.readFileSync(jsonFilePath, 'utf-8');
    const jsonData = JSON.parse(jsonText);

    const defaultFields = {
      ir_version: 9,
      opset_import: [{ domain: '', version: 17 }],
      producer_name: 'onnx-flow',
      producer_version: '0.1.0',
      model_version: 1,
    };

    const completeJson = {
      ...defaultFields,
      ...jsonData,
      graph: {
        name: jsonData.graph?.name ?? 'default_graph',
        ...jsonData.graph,
      }
    };

    const fixedJson = fixBuffers(completeJson);

    // Resilient Reshape shape fix runs BEFORE numeric coercion
    fixSingleNullReshapeShapes(fixedJson);

    const normalizedJson = coerceNumericFields(fixedJson);

    const errMsg = ModelProto.verify(normalizedJson);
    if (errMsg) {
      throw new Error('Validation error: ' + errMsg);
    }

    const message = ModelProto.create(normalizedJson);
    const buffer = ModelProto.encode(message).finish();

    fs.writeFileSync(outputOnnxPath, buffer);
    console.log(`ONNX model successfully written to ${outputOnnxPath}`);
  } catch (error) {
    console.error('Failed to convert JSON to ONNX:');
    if (error instanceof Error) {
      console.error('Message:', error.message);
    } else {
      console.error(error);
    }
    throw error;
  }
}
