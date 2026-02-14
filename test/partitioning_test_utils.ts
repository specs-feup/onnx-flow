import fs from "fs";
import path from "path";
import * as ort from "onnxruntime-web";
import { createGraph } from "../src/initGraph.js";
import { onnx2json } from "../src/onnx2json.js";
import { convertFlowGraphToOnnxJson } from "../src/flow2json.js";
import { json2onnx } from "../src/json2onnx.js";
import { splitByAncestor } from "../src/Onnx/partitioning/Strategies.js";
import { partitionGraph } from "../src/Onnx/partitioning/Partition.js";

export interface InputSpec {
    name: string;
    dtype: string;
    shape: number[];
}

export interface PartitionTestCase {
    label: string;
    originalPath: string;
    splitNodeId?: string;
    specs: InputSpec[];
    tol?: number;
}

function generateTensorFromSpec(spec: InputSpec): ort.Tensor {
    const size = spec.shape.reduce((a, b) => a * b, 1);
    let data: any;

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
        data = new Int8Array(size).map(() => Math.floor(Math.random() * 255 - 128));
    } else {
        throw new Error(`Unsupported dtype: ${spec.dtype}`);
    }

    return new ort.Tensor(spec.dtype, data, spec.shape);
}

async function saveGraphToTempOnnx(graph: any, prefix: string): Promise<string> {
    const json = convertFlowGraphToOnnxJson(graph);
    const tmpJson = path.resolve(`temp_${prefix}.json`);
    const tmpOnnx = path.resolve(`temp_${prefix}.onnx`);

    fs.writeFileSync(tmpJson, JSON.stringify(json, null, 2));
    await json2onnx(tmpJson, tmpOnnx);
    fs.unlinkSync(tmpJson);
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
    if (!splitId) {
        const ops = irGraph.getOperationNodes().toArray();
        const eligibleOps = ops.filter((op) => op.type !== "Constant");
        if (eligibleOps.length === 0) throw new Error(`No eligible operation nodes found.`);
        const randomOp = eligibleOps[Math.floor(Math.random() * eligibleOps.length)];
        splitId = randomOp.id;
        console.log(`   🎲 Randomly selected split node: '${splitId}' (${randomOp.type})`);
    } else {
        console.log(`   📍 Using specified split node: '${splitId}'`);
    }

    // A. Determine Sets
    const sets = splitByAncestor(irGraph, splitId);
    const headCount = sets.head.size;
    const tailCount = sets.tail.size;
    console.log(`   Split stats: Head=${headCount} nodes, Tail=${tailCount} nodes.`);

    // B. Construct Graphs
    const { head, tail } = partitionGraph(irGraph, sets);

    const headPath = path.resolve(`temp_head_${testCase.label}.onnx`);
    const tailPath = path.resolve(`temp_tail_${testCase.label}.onnx`);

    try {
        let resultsHead: Record<string, ort.Tensor> = {};

        // --- 5. Run Head (Only if not empty) ---
        if (head.nodes.length > 0) {
            await saveGraphToTempOnnx(head, `head_${testCase.label}`);
            const sessionHead = await ort.InferenceSession.create(headPath);

            const headFeeds: Record<string, ort.Tensor> = {};
            sessionHead.inputNames.forEach((name) => {
                if (feeds[name]) headFeeds[name] = feeds[name];
                else throw new Error(`Head model input '${name}' missing from specs.`);
            });

            resultsHead = await sessionHead.run(headFeeds);
        } else {
            console.warn("   ⚠️ Head partition is empty. Skipping execution.");
        }

        // --- 6. Run Tail (Only if not empty) ---
        let resultsTail: Record<string, ort.Tensor> = {};

        if (tail.nodes.length > 0) {
            await saveGraphToTempOnnx(tail, `tail_${testCase.label}`);
            const sessionTail = await ort.InferenceSession.create(tailPath);

            const tailFeeds: Record<string, ort.Tensor> = {};
            sessionTail.inputNames.forEach((name) => {
                if (resultsHead[name]) tailFeeds[name] = resultsHead[name];
                else if (feeds[name]) tailFeeds[name] = feeds[name];
                else throw new Error(`Tail model input '${name}' missing.`);
            });

            resultsTail = await sessionTail.run(tailFeeds);
        } else {
            console.warn("   ⚠️ Tail partition is empty. Using Head results as final.");
            // If Tail is empty, Head outputs MUST match Original outputs
            resultsTail = resultsHead;
        }

        // 7. Validate Results
        for (const outName of sessionOrig.outputNames) {
            const valOrig = resultsOrig[outName];
            const valTail = resultsTail[outName];

            if (!valTail) {
                throw new Error(`Output '${outName}' missing from final results (Tail/Head).`);
            }

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

        console.log(`   ✅ Success! Max diff: ${0} (or within ${tol})`);
    } catch (e: any) {
        console.error(`   ❌ Failed: ${e.message ?? e}`);
        if (e.stack) console.error(e.stack);
        throw e;
    } finally {
        if (fs.existsSync(headPath)) fs.unlinkSync(headPath);
        if (fs.existsSync(tailPath)) fs.unlinkSync(tailPath);
    }
}
