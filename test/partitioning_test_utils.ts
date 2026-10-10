import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";
import { createGraph } from "../src/initGraph.js";
import { onnx2json } from "../src/onnx2json.js";
import { convertFlowGraphToOnnxJson } from "../src/flow2json.js";
import { json2onnx } from "../src/json2onnx.js";
import { splitByAncestor } from "../src/Onnx/partitioning/Strategies.js";
import { partitionGraph } from "../src/Onnx/partitioning/Partition.js";
import type OnnxGraph from "../src/Onnx/OnnxGraph.js";

export interface InputSpec {
    name: string;
    dtype: string;
    shape: number[];
    data?: number[] | bigint[];
}

export interface PartitionTestCase {
    label: string;
    originalPath: string;
    splitNodeId?: string;
    specs: InputSpec[];
    tol?: number;
}

export function generateTensorFromSpec(spec: InputSpec): ort.Tensor {
    const size = spec.shape.reduce((a, b) => a * b, 1);
    let data;
    if (spec.data !== undefined) {
        if (spec.data.length !== size) {
            throw new Error(`Provided data length does not match shape size for '${spec.name}'.`);
        }
        if (spec.dtype === "float32") data = new Float32Array(spec.data as number[]);
        else if (spec.dtype === "int64") {
            const bigIntData = (spec.data as (number | bigint)[]).map((v) => BigInt(v));
            data = new BigInt64Array(bigIntData);
        } else if (spec.dtype === "int32") data = new Int32Array(spec.data as number[]);
        else if (spec.dtype === "bool") data = new Uint8Array(spec.data as number[]);
        else if (spec.dtype === "uint8") data = new Uint8Array(spec.data as number[]);
        else if (spec.dtype === "int8") data = new Int8Array(spec.data as number[]);
        else throw new Error(`Unsupported dtype: ${spec.dtype}`);
    } else {
        if (spec.dtype === "float32") {
            data = new Float32Array(size).map(() => Math.random());
        } else if (spec.dtype === "int64") {
            data = new BigInt64Array(size).map(() => BigInt(Math.floor(Math.random() * 100)));
        } else if (spec.dtype === "int32") {
            data = new Int32Array(size).map(() => Math.floor(Math.random() * 100));
        } else if (spec.dtype === "bool") {
            data = new Uint8Array(size).map(() => (Math.random() > 0.5 ? 1 : 0));
        } else if (spec.dtype === "uint8") {
            data = new Uint8Array(size).map(() => Math.floor(Math.random() * 255));
        } else if (spec.dtype === "int8") {
            data = new Int8Array(size).map(() => Math.floor(Math.random() * 256) - 128);
        } else {
            throw new Error(`Unsupported dtype: ${spec.dtype}`);
        }
    }
    return new ort.Tensor(spec.dtype, data, spec.shape);
}

// Generates unique temporary files to prevent parallel test collisions
async function saveGraphToTempOnnx(graph: OnnxGraph.Class, prefix: string): Promise<string> {
    const json = convertFlowGraphToOnnxJson(graph);
    const uniqueToken = `${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    const tmpJson = path.resolve(`temp_${prefix}_${uniqueToken}.json`);
    const tmpOnnx = path.resolve(`temp_${prefix}_${uniqueToken}.onnx`);

    fs.writeFileSync(tmpJson, JSON.stringify(json, null, 2));
    await json2onnx(tmpJson, tmpOnnx);
    if (fs.existsSync(tmpJson)) {
        try {
            fs.unlinkSync(tmpJson);
        } catch {
            /* ignore */
        }
    }
    return tmpOnnx;
}

export async function runPartitionTest(testCase: PartitionTestCase): Promise<void> {
    console.log(`\n[Test: ${testCase.label}] Initializing...`);
    const tol = testCase.tol ?? 1e-4;

    // 1. Prepare Inputs
    const feeds: Record<string, ort.Tensor> = {};
    for (const spec of testCase.specs) {
        feeds[spec.name] = generateTensorFromSpec(spec);
    }

    // 2. Run Original Model
    const sessionOrig = await ort.InferenceSession.create(testCase.originalPath);
    const resultsOrig = await sessionOrig.run(feeds);

    // 3. Load Graph IR
    const modelJson = await onnx2json(testCase.originalPath);
    const irGraph = createGraph(modelJson);

    // 4. Select Split Node
    let splitId = testCase.splitNodeId;
    if (splitId === undefined) {
        const ops = irGraph.getOperationNodes().toArray();
        const eligibleOps = ops.filter((op) => op.type !== "Constant");
        if (eligibleOps.length === 0) throw new Error(`No eligible operation nodes found.`);
        const randomOp = eligibleOps[Math.floor(Math.random() * eligibleOps.length)];
        splitId = randomOp.id;
        console.log(`     Randomly selected split node: '${splitId}' (${randomOp.type})`);
    } else {
        console.log(`     Using specified split node: '${splitId}'`);
    }

    // A. Determine Sets
    const sets = splitByAncestor(irGraph, splitId);
    console.log(`   Split stats: Head=${sets.head.size} nodes, Tail=${sets.tail.size} nodes.`);

    // B. Construct Graphs
    const { head, tail } = partitionGraph(irGraph, sets);
    let headPath = "";
    let tailPath = "";

    try {
        let resultsHead: Record<string, ort.Tensor> = {};

        // --- 5. Run Head ---
        if (head.nodes.length > 0) {
            headPath = await saveGraphToTempOnnx(head, `head_${testCase.label}`);
            const sessionHead = await ort.InferenceSession.create(headPath);
            const headFeeds: Record<string, ort.Tensor> = {};
            sessionHead.inputNames.forEach((name) => {
                if (name in feeds) headFeeds[name] = feeds[name];
                else throw new Error(`Head model input '${name}' missing from specs.`);
            });
            resultsHead = await sessionHead.run(headFeeds);
        } else {
            console.warn("     Head partition is empty. Skipping execution.");
        }

        // --- 6. Run Tail ---
        let resultsTail: Record<string, ort.Tensor> = {};
        if (tail.nodes.length > 0) {
            tailPath = await saveGraphToTempOnnx(tail, `tail_${testCase.label}`);
            const sessionTail = await ort.InferenceSession.create(tailPath);
            const tailFeeds: Record<string, ort.Tensor> = {};
            sessionTail.inputNames.forEach((name) => {
                if (name in resultsHead) tailFeeds[name] = resultsHead[name];
                else if (name in feeds) tailFeeds[name] = feeds[name];
                else throw new Error(`Tail model input '${name}' missing.`);
            });
            resultsTail = await sessionTail.run(tailFeeds);
        } else {
            console.warn("     Tail partition is empty. Using Head results as final.");
            resultsTail = resultsHead;
        }

        // 7. Validate Results
        for (const outName of sessionOrig.outputNames) {
            const valOrig = resultsOrig[outName];
            const valTail = resultsTail[outName];
            const d1 = valOrig.data as Float32Array;
            const d2 = valTail.data as Float32Array;
            if (d1.length !== d2.length) {
                throw new Error(`Shape mismatch for '${outName}'`);
            }
            let maxDiff = 0;
            for (let i = 0; i < d1.length; i++) {
                const diff = Math.abs(d1[i] - d2[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            if (maxDiff > tol) {
                throw new Error(
                    `Value mismatch for '${outName}': max diff ${maxDiff} exceeds tolerance ${tol}`,
                );
            }
        }
        console.log(`     Success! Max diff within tolerance ${tol}`);
    } finally {
        if (headPath && fs.existsSync(headPath)) {
            try {
                fs.unlinkSync(headPath);
            } catch {
                /* ignore */
            }
        }
        if (tailPath && fs.existsSync(tailPath)) {
            try {
                fs.unlinkSync(tailPath);
            } catch {
                /* ignore */
            }
        }
    }
}
