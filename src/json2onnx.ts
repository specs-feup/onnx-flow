import fs from 'fs';
import path from 'path';
import protobuf from 'protobufjs';
import { fileURLToPath } from 'url';


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

// Coerce numeric-like strings to numbers for fields protobuf expects as ints/floats.
// Also normalizes common ONNX numeric array fields (ints, floats, dims, etc.).
function coerceNumericFields(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(coerceNumericFields);
  }
  if (obj && typeof obj === 'object') {
    for (const k of Object.keys(obj)) {
      const v = obj[k];

      // Recurse first
      obj[k] = coerceNumericFields(v);

      // Now fix known numeric fields
      if (k === 'ints' && Array.isArray(obj[k])) {
        obj[k] = obj[k].map((x: any) =>
          typeof x === 'string' ? parseInt(x, 10) : x
        );
      } else if (k === 'floats' && Array.isArray(obj[k])) {
        obj[k] = obj[k].map((x: any) =>
          typeof x === 'string' ? parseFloat(x) : x
        );
      } else if ((k === 'i' || k === 'f') && (typeof obj[k] === 'string')) {
        obj[k] = k === 'i' ? parseInt(obj[k], 10) : parseFloat(obj[k]);
      } else if (k === 'dims' && Array.isArray(obj[k])) {
        // Tensor/shape dims (accept -1, etc.)
        obj[k] = obj[k].map((x: any) =>
          typeof x === 'string' ? parseInt(x, 10) : x
        );
      }
    }
  }
  return obj;
}

export async function json2onnx(jsonFilePath: string, outputOnnxPath: string): Promise<void> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const protoPath = path.join(__dirname, '../../out/src/Onnx/onnx.proto');

  try {
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
