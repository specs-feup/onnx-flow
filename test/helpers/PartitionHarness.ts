import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import * as ort from "onnxruntime-web";
import { createGraph } from "../../src/io/OnnxGraphLoader.js";
import { onnx2json } from "../../src/io/OnnxToJson.js";
import { convertFlowGraphToOnnxJson } from "../../src/io/OnnxGraphExporter.js";
import { json2onnx } from "../../src/io/JsonToOnnx.js";
import { splitByAncestor } from "../../src/onnx/partitioning/Strategies.js";
import { partitionGraph } from "../../src/onnx/partitioning/Partition.js";
import type OnnxGraph from "../../src/onnx/ir/OnnxGraph.js";
import { buildFeeds, compareOrtOutputs, ensureProtoExists, type FeedSpec } from "./ortHarness.js";

export interface PartitionTestCase {
    label: string;
    originalPath: string;
    splitNodeId?: string;
    specs: FeedSpec[];
    tol?: number;
}

async function saveGraphToTempOnnx(
    graph: OnnxGraph.Class,
    tmpDir: string,
    prefix: string,
): Promise<string> {
    const json = convertFlowGraphToOnnxJson(graph);
    const tmpJson = path.join(tmpDir, `${prefix}.json`);
    const tmpOnnx = path.join(tmpDir, `${prefix}.onnx`);

    fs.writeFileSync(tmpJson, JSON.stringify(json, null, 2));
    await json2onnx(tmpJson, tmpOnnx);
    return tmpOnnx;
}

export async function runPartitionTest(testCase: PartitionTestCase): Promise<void> {
    ensureProtoExists();
    const tol = testCase.tol ?? 1e-4;
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "onnx-flow-part-"));

    try {
        // 1. Prepare deterministic Inputs
        const feeds = buildFeeds(testCase.specs, `part_${testCase.label}`);

        // 2. Run Original Model
        const sessionOrig = await ort.InferenceSession.create(testCase.originalPath);
        const resultsOrig = await sessionOrig.run(feeds);

        // 3. Load Graph IR
        const modelJson = await onnx2json(testCase.originalPath);
        const irGraph = createGraph(modelJson);

        // 4. Select Split Node deterministically (middle non-Constant op if unspecified)
        let splitId = testCase.splitNodeId;
        if (splitId === undefined) {
            const ops = irGraph.getOperationNodes().toArray();
            const eligibleOps = ops.filter((op) => op.type !== "Constant");
            if (eligibleOps.length === 0) throw new Error(`No eligible operation nodes found.`);
            const targetOp = eligibleOps[Math.floor(eligibleOps.length / 2)];
            splitId = targetOp.id;
        }

        // A. Determine Sets
        const sets = splitByAncestor(irGraph, splitId);

        // B. Construct Graphs
        const { head, tail } = partitionGraph(irGraph, sets);

        let resultsHead: Record<string, ort.Tensor> = {};

        // 5. Run Head
        if (head.nodes.length > 0) {
            const headPath = await saveGraphToTempOnnx(head, tmpDir, "head");
            const sessionHead = await ort.InferenceSession.create(headPath);
            const headFeeds: Record<string, ort.Tensor> = {};
            for (const name of sessionHead.inputNames) {
                if (name in feeds) headFeeds[name] = feeds[name];
                else throw new Error(`Head model input '${name}' missing from specs.`);
            }
            resultsHead = await sessionHead.run(headFeeds);
        }

        // 6. Run Tail
        let resultsTail: Record<string, ort.Tensor> = {};
        if (tail.nodes.length > 0) {
            const tailPath = await saveGraphToTempOnnx(tail, tmpDir, "tail");
            const sessionTail = await ort.InferenceSession.create(tailPath);
            const tailFeeds: Record<string, ort.Tensor> = {};
            for (const name of sessionTail.inputNames) {
                if (name in resultsHead) tailFeeds[name] = resultsHead[name];
                else if (name in feeds) tailFeeds[name] = feeds[name];
                else throw new Error(`Tail model input '${name}' missing.`);
            }
            resultsTail = await sessionTail.run(tailFeeds);
        } else {
            resultsTail = resultsHead;
        }

        // 7. Validate Results (supports float32, int32, int64, bool, uint8, int8, NaN/Inf)
        const cmp = compareOrtOutputs(resultsOrig, resultsTail, { tol });
        if (!cmp.passed) {
            throw new Error(cmp.message ?? `Partition output mismatch for '${testCase.label}'`);
        }
    } finally {
        if (fs.existsSync(tmpDir)) {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        }
    }
}
