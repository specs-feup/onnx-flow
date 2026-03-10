/*
import path from "path";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { onnx2json } from "../src/onnx2json.js";
import { createGraph } from "../src/initGraph.js";
import { fileURLToPath } from "url";
import { TYPE_SIZE_MAP } from "@specs-feup/onnx-flow/Onnx/Utils";
import OnnxGraphTransformer from "@specs-feup/onnx-flow/Onnx/transformation/index";

// Dynamically determine the project root directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, "../../"); // Adjust based on directory structure

//AddAdd, AddAddAdd
async function runTests() {
    // Load the ONNX model or convert JSON to ONNX
    const inputFilePath = path.resolve(PROJECT_ROOT, "examples/onnx/AddAddAdd.onnx");
    const onnxObject = await onnx2json(inputFilePath);

    // Create an inference session
    const session = await InferenceSession.create(inputFilePath);

    // Helper function to generate random arrays of given shape and data type
    function getRandomArray(
        shape: number[],
        elemType: string,
    ):
        | Float32Array
        | Int32Array
        | BigInt64Array
        | Uint8Array
        | Int8Array
        | Uint16Array
        | Int16Array
        | Uint32Array
        | BigUint64Array
        | Float64Array {
        const size = shape.reduce((a, b) => a * b, 1); // Ensure size is calculated correctly
        switch (elemType) {
            case "float32":
                return Float32Array.from({ length: size }, () => Math.random());
            case "int32":
                return Int32Array.from({ length: size }, () => Math.floor(Math.random() * 100));
            case "int64":
                return BigInt64Array.from({ length: size }, () =>
                    BigInt(Math.floor(Math.random() * 100)),
                );
            case "uint8":
                return Uint8Array.from({ length: size }, () => Math.floor(Math.random() * 256));
            case "int8":
                return Int8Array.from(
                    { length: size },
                    () => Math.floor(Math.random() * 256) - 128,
                );
            case "uint16":
                return Uint16Array.from({ length: size }, () => Math.floor(Math.random() * 65536));
            case "int16":
                return Int16Array.from(
                    { length: size },
                    () => Math.floor(Math.random() * 65536) - 32768,
                );
            case "uint32":
                return Uint32Array.from({ length: size }, () =>
                    Math.floor(Math.random() * 4294967296),
                );
            case "uint64":
                return BigUint64Array.from({ length: size }, () =>
                    BigInt(Math.floor(Math.random() * 100)),
                );
            case "float64":
                return Float64Array.from({ length: size }, () => Math.random());
            case "uint4":
                return Uint8Array.from({ length: size }, () => Math.floor(Math.random() * 16)); // Simulate 4-bit unsigned integer
            case "int4":
                return Int8Array.from({ length: size }, () => Math.floor(Math.random() * 16) - 8); // Simulate 4-bit signed integer
            case "float4":
                return Float32Array.from({ length: size }, () => Math.random() * 16); // Simulate 4-bit float
            default:
                throw new Error(`Unsupported data type: ${elemType}`);
        }
    }

    // Map ONNX tensor element type to corresponding array type
    function getArrayType(elemType: number): string {
        switch (elemType) {
            case 1:
                return "float32";
            case 2:
                return "uint8";
            case 3:
                return "int8";
            case 4:
                return "uint16";
            case 5:
                return "int16";
            case 6:
                return "int32";
            case 7:
                return "int64";
            case 11:
                return "float64";
            case 12:
                return "uint32";
            case 13:
                return "uint64";
            case 21:
                return "uint4";
            case 22:
                return "int4";
            case 23:
                return "float4";
            default:
                throw new Error(`Unsupported tensor element type: ${elemType}`);
        }
    }

    // Create a list of inputs dynamically based on the model's input specifications
    const listOfInputs: Record<string, Tensor> = {};
    onnxObject.graph?.input?.forEach((input) => {
        // 1. Safely resolve tensorType (handling both camelCase and snake_case)
        const tensorType = input.type?.tensorType ?? input.type?.tensor_type;

        // 2. Map dimensions safely, using Number() to handle string|number unions
        const shape = (tensorType?.shape?.dim ?? []).map(
            (dim) => Number(dim.dimValue ?? dim.dim_value ?? 1), // Default to 1 if unknown
        );

        // 3. Extract elemType strictly as a number
        const rawElemType = Number(tensorType?.elemType ?? tensorType?.elem_type ?? 0);
        const elemType = getArrayType(rawElemType);

        if (input.name) {
            listOfInputs[input.name] = new Tensor(
                elemType as keyof Tensor.DataTypeMap,
                getRandomArray(shape, elemType),
                shape,
            );
        }
    });
    console.log("Generated random inputs:", listOfInputs, "\n\n");

    // Run the model with the generated inputs
    const output = await session.run(listOfInputs);
    console.log("Model output:", output, "\n\n");

    const outputTensorName = Object.keys(output)[0];
    const outputTensor = output[outputTensorName];

    // Create the OnnxGraph with no optimizations applied
    const graph1 = createGraph(onnxObject).apply(new OnnxGraphTransformer());
    console.log("Created OnnxGraph with no optimizations", "\n\n");

    // Generate code from the graph with no optimizationstransformations
    const generatedCode1 = generateCode(graph1);
    console.log("Generated Code (no optimizations):", generatedCode1, "\n\n");

    // Create the OnnxGraph with optimizations applied
    const graph2 = graph1.apply(new OnnxGraphOptimizer());
    console.log("Created OnnxGraph with optimizations", "\n\n");

    // Generate code from the graph with optimizations
    const generatedCode2 = generateCode(graph2);
    console.log("Generated Code (with optimizations):", generatedCode2, "\n\n");

    // Convert the randomly generated inputs to a format accepted by the generated code
    const formattedInputs: Record<string, unknown> = {};
    for (const [key, tensor] of Object.entries(listOfInputs)) {
        // 1. Safely find the input definition
        const inputDef = onnxObject.graph?.input?.find((input) => input.name === key);

        // 2. Safely drill down to the tensor type (handling snake_case fallbacks just in case)
        const tensorType = inputDef?.type?.tensorType ?? inputDef?.type?.tensor_type;
        const rawElemType = tensorType?.elemType ?? tensorType?.elem_type;

        // 3. Stop if we couldn't resolve the type, rather than crashing
        if (rawElemType === undefined) {
            throw new Error(`Could not resolve elemType for input '${key}'`);
        }

        // 4. Safely coerce to number for your TYPE_SIZE_MAP lookup
        const elemType = Number(rawElemType);
        const displacement = TYPE_SIZE_MAP[elemType];
        formattedInputs[`tensor_${key}`] = {};
        // 1. Create a strictly typed temporary map for the offsets
        const tensorDataMap: Record<number, number> = {};

        // 2. Calculate the total number of elements
        const totalElements = tensor.dims.reduce((a, b) => a * b, 1);

        // 3. Populate the map safely
        for (let i = 0; i < totalElements; i++) {
            tensorDataMap[i * displacement] = tensor.data[i] as number;
        }

        // 4. Assign the fully populated map back to your formatted inputs
        formattedInputs[`tensor_${key}`] = tensorDataMap;
    }
    console.log("Formatted inputs for generated code:", formattedInputs, "\n\n");

    // Prepare the arguments for the generated function
    const inputKeys = Object.keys(formattedInputs);
    const inputValues = inputKeys.map((key) => formattedInputs[key]);

    // Log the inputs being sent to the generated function
    console.log("Inputs being sent to the generated function:", inputKeys, inputValues);

    // Run the generated code (no optimizations)
    const generatedFunction1 = new Function(
        ...inputKeys,
        generatedCode1 + ` return onnxGraph(${inputKeys.join(", ")});`,
    );
    const generatedOutput1 = generatedFunction1(...inputValues);
    console.log("Generated Output (no transformations):", generatedOutput1);

    // Run the generated code (with optimizations)
    const generatedFunction2 = new Function(
        ...inputKeys,
        generatedCode2 + ` return onnxGraph(${inputKeys.join(", ")});`,
    );
    const generatedOutput2 = generatedFunction2(...inputValues);
    console.log("Generated Output (with transformations):", generatedOutput2);

    // Convert values to strings for comparison
    const outputData = Array.from(outputTensor.data as unknown[]).map((value) => value?.toString());
    const generatedOutputValues1 = Object.values(generatedOutput1).map((value) =>
        value?.toString(),
    );
    const generatedOutputValues2 = Object.values(generatedOutput2).map((value) =>
        value?.toString(),
    );

    // Compare the results (no optimizations)
    const tolerance = 1e-6;
    const areEqual1 = outputData.every(
        (value, index) =>
            Math.abs(parseFloat(value!) - parseFloat(generatedOutputValues1[index]!)) < tolerance,
    );
    console.log("Test (no optimizations) passed?", areEqual1);

    // Compare the results (with optimizations)
    const areEqual2 = outputData.every(
        (value, index) =>
            Math.abs(parseFloat(value!) - parseFloat(generatedOutputValues2[index]!)) < tolerance,
    );
    console.log("Test (with optimizations) passed?", areEqual2);
}

runTests().catch((err) => console.error(err));
*/
