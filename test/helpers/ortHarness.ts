import * as ort from "onnxruntime-web";
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { onnx2json } from "../../src/onnx2json.js";
import { createGraph } from "../../src/initGraph.js";
import { convertFlowGraphToOnnxJson, prepareGraphForExport } from "../../src/flow2json.js";
import { json2onnx } from "../../src/json2onnx.js";
import { safeWriteJson } from "../../src/Onnx/Utils.js";
import type OnnxGraph from "../../src/Onnx/OnnxGraph.js";

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

export const PROJECT_ROOT = findProjectRoot(__dirname);

export function ensureProtoExists(): void {
    const srcProto = path.resolve(PROJECT_ROOT, "src/Onnx/onnx.proto");
    if (!fs.existsSync(srcProto)) return;
    const targets = [
        path.resolve(PROJECT_ROOT, "out/src/Onnx/onnx.proto"),
        path.resolve(PROJECT_ROOT, "../out/src/Onnx/onnx.proto"),
    ];
    for (const target of targets) {
        if (!fs.existsSync(target)) {
            try {
                fs.mkdirSync(path.dirname(target), { recursive: true });
                fs.copyFileSync(srcProto, target);
            } catch (err: unknown) {
                const code = (err as { code?: string }).code;
                if (code !== "EBUSY" && code !== "EPERM" && code !== "EEXIST") {
                    throw err;
                }
            }
        }
    }
}

/* ============================== TYPES & CONFIG ================================== */
export type DType = "uint8" | "int8" | "float32" | "int32" | "int64" | "bool";

export type FeedSpec = {
    name: string;
    dtype: DType;
    shape: number[];
    init?: number[] | bigint[] | boolean[];
    gen?: "random" | "negmix" | "zeros" | "ones" | "range" | "greaterThanOne";
    value?: number;
};

export interface EquivalenceTestConfig {
    label: string;
    originalPath: string;
    tol?: number;
    rtol?: number;
    exact?: boolean;
    cliArgs: string | ((p: string) => string);
    specs: FeedSpec[];
}

export interface EquivalenceOptions {
    skipCli?: boolean;
    showStats?: boolean;
    verboseOnFailure?: boolean;
    /** Optional directory for staging input copies and CLI outputs without mutating examples/ */
    workDir?: string;
}

export type SingleEquivResult = "pass" | "warn" | "error" | "no-config";

export interface EquivalenceRunResult {
    status: SingleEquivResult;
    message?: string;
    maxDiff?: number;
    originalStats?: GraphStats | null;
    decomposedStats?: GraphStats | null;
}

export interface GraphStats {
    total: number;
    ops: Record<string, number>;
}

/* ============================== STATS HELPERS ================================== */
export function collectStatsRecursive(graph: OnnxGraph.Class, stats: GraphStats): void {
    const opNodes = graph.getOperationNodes().toArray();
    stats.total += opNodes.length;
    for (const node of opNodes) {
        stats.ops[node.type] = (stats.ops[node.type] ?? 0) + 1;
        for (const region of node.regions) {
            collectStatsRecursive(region, stats);
        }
    }
}

export async function getGraphStats(modelPath: string): Promise<GraphStats | null> {
    if (!fs.existsSync(modelPath)) return null;
    try {
        const json = await onnx2json(modelPath);
        const graph = createGraph(json);
        const stats: GraphStats = { total: 0, ops: {} };
        collectStatsRecursive(graph, stats);
        return stats;
    } catch {
        return null;
    }
}

export function printStatsComparison(
    label: string,
    original: GraphStats | null,
    decomposed: GraphStats | null,
): void {
    console.log(`\n--- Stats for ${label} (Recursive) ---`);
    if (original) console.log(`Original Nodes: ${original.total}`);
    if (decomposed) console.log(`Decomposed Nodes: ${decomposed.total}`);
    if (original && decomposed)
        console.log(`Node Count Change: ${original.total} -> ${decomposed.total}`);

    console.log("Operations:");
    const opsSet = new Set<string>();
    if (original) Object.keys(original.ops).forEach((k) => opsSet.add(k));
    if (decomposed) Object.keys(decomposed.ops).forEach((k) => opsSet.add(k));
    const sorted = Array.from(opsSet).sort();
    for (const op of sorted) {
        const from = original?.ops[op] ?? 0;
        const to = decomposed?.ops[op] ?? 0;
        if (original && decomposed) {
            if (from !== to) console.log(`   ${op.padEnd(25)}: ${from} -> ${to}`);
        } else {
            const val = original ? from : to;
            if (val > 0) console.log(`   ${op.padEnd(25)}: ${val}`);
        }
    }
    console.log("--------------------------\n");
}

/* ============================== ORT HELPERS ================================== */
export async function rerunWithOrtVerbose(
    modelPath: string,
    feeds: Record<string, ort.Tensor>,
): Promise<void> {
    console.log("\n  [Diagnostic] Re-running model with ORT verbose WebAssembly logging...");
    const prevDebug = ort.env.debug;
    const prevLogLevel = ort.env.logLevel;
    try {
        ort.env.debug = true;
        ort.env.logLevel = "verbose";
        const so: ort.InferenceSession.SessionOptions = {
            executionProviders: ["wasm"],
            logSeverityLevel: 0,
            logVerbosityLevel: 4,
        };
        const sess = await ort.InferenceSession.create(modelPath, so);
        const outNames = sess.outputNames;
        const outs = await sess.run(feeds, outNames);
        console.log(
            "     Diagnostic run outputs:",
            outNames.map((k) => `${k}[${outs[k].dims}] ${outs[k].type}`),
        );
    } catch (e) {
        console.error("  Diagnostic re-run failed with error:", e);
    } finally {
        if (prevDebug !== undefined) ort.env.debug = prevDebug;
        if (prevLogLevel !== undefined) ort.env.logLevel = prevLogLevel;
    }
}

export async function runOrtChecked(
    stage: string,
    modelPath: string,
    feeds: Record<string, ort.Tensor>,
    sessionOptions?: ort.InferenceSession.SessionOptions,
): Promise<Record<string, ort.Tensor>> {
    try {
        const session = sessionOptions
            ? await ort.InferenceSession.create(modelPath, sessionOptions)
            : await ort.InferenceSession.create(modelPath);
        return await session.run(feeds);
    } catch (e) {
        console.error(
            `\n[ORT Diagnostic] Re-running ${stage} (${modelPath}) with verbose logging...`,
        );
        const jsonPath = `${modelPath}.json`;
        if (fs.existsSync(jsonPath)) {
            const m = JSON.parse(fs.readFileSync(jsonPath, "utf8")) as {
                opsetImport?: unknown;
                irVersion?: unknown;
                graph?: {
                    output?: unknown;
                    node?: {
                        opType: string;
                        input: string[];
                        output: string[];
                        attribute?: {
                            name: string;
                            g?: {
                                input?: unknown;
                                output?: unknown;
                                node?: { opType: string; input: string[]; output: string[] }[];
                            };
                        }[];
                    }[];
                };
            };
            console.error(`[Model Summary ${stage}] opset=`, m.opsetImport, "ir=", m.irVersion);
            console.error(`[Model Summary ${stage}] outputs=`, m.graph?.output);
            console.error(
                `[Model Summary ${stage}] nodes=`,
                (m.graph?.node ?? []).map(
                    (n) => `${n.opType}(${n.input.join(",")})->(${n.output.join(",")})`,
                ),
            );
            const lastLoop = [...(m.graph?.node ?? [])].reverse().find((n) => n.opType === "Loop");
            const bodyGraph = lastLoop?.attribute?.find((a) => a.name === "body")?.g;
            if (bodyGraph !== undefined) {
                console.error(`[Loop Body ${stage}] inputs=`, JSON.stringify(bodyGraph.input));
                console.error(`[Loop Body ${stage}] outputs=`, JSON.stringify(bodyGraph.output));
                console.error(
                    `[Loop Body ${stage}] nodes=`,
                    (bodyGraph.node ?? []).map(
                        (n) => `${n.opType}(${n.input.join(",")})->(${n.output.join(",")})`,
                    ),
                );
            }
        }
        await rerunWithOrtVerbose(modelPath, feeds);
        throw new Error(`[${stage}] ORT failed: ${e instanceof Error ? e.message : String(e)}`);
    }
}

export async function exportGraphToOnnx(
    graph: OnnxGraph.Class,
    onnxPath: string,
    minOpset?: number,
    exactOpset?: number,
): Promise<void> {
    prepareGraphForExport(graph);
    const json = convertFlowGraphToOnnxJson(graph) as {
        irVersion?: number;
        opsetImport?: { domain?: string; version?: number }[];
    };
    if (json.opsetImport?.[0]) {
        if (exactOpset !== undefined) {
            json.opsetImport[0].version = exactOpset;
            json.irVersion = exactOpset >= 20 ? 9 : 7;
        } else if (minOpset !== undefined) {
            json.opsetImport[0].version = Math.max(json.opsetImport[0].version ?? 13, minOpset);
            if (minOpset >= 20) {
                json.irVersion = Math.max(json.irVersion ?? 7, 9);
            }
        }
    }
    const tmpJson = `${onnxPath}.json`;
    safeWriteJson(tmpJson, json);
    await json2onnx(tmpJson, onnxPath);
}

/* ============================== FEED GENERATORS ================================== */
function createSeededRng(seedStr: string): () => number {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < seedStr.length; i++) {
        h ^= seedStr.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return () => {
        h += 0x6d2b79f5;
        let t = Math.imul(h ^ (h >>> 15), 1 | h);
        t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function makeArray(
    dtype: DType,
    len: number,
    gen: FeedSpec["gen"],
    rng: () => number = Math.random,
): number[] | bigint[] | boolean[] {
    const randFloat = () => rng() * 10;
    const randNegMix = () => rng() * 2 - 1;
    const randInt = (max = 10) => Math.floor(rng() * max);
    const randBool = () => rng() < 0.5;

    switch (dtype) {
        case "float32":
            if (gen === "negmix") return Array.from({ length: len }, randNegMix);
            if (gen === "zeros") return Array.from({ length: len }, () => 0);
            if (gen === "ones") return Array.from({ length: len }, () => 1);
            if (gen === "greaterThanOne") return Array.from({ length: len }, () => randFloat() + 1);
            return Array.from({ length: len }, randFloat);
        case "int32":
            if (gen === "zeros") return Array.from({ length: len }, () => 0);
            if (gen === "ones") return Array.from({ length: len }, () => 1);
            return Array.from({ length: len }, () => randInt(10));
        case "int64":
            if (gen === "zeros") return Array.from({ length: len }, () => 0n);
            if (gen === "ones") return Array.from({ length: len }, () => 1n);
            return Array.from({ length: len }, () => BigInt(randInt(10)));
        case "int8":
            return Array.from({ length: len }, () => randInt(256) - 128);
        case "uint8":
            return Array.from({ length: len }, () => randInt(256));
        case "bool":
            return Array.from({ length: len }, randBool);
    }
}

export function buildFeeds(
    specs: FeedSpec[],
    seedKey: string = "default",
): Record<string, ort.Tensor> {
    const rng = createSeededRng(seedKey);
    const out: Record<string, ort.Tensor> = {};
    for (const s of specs) {
        const len = s.shape.reduce((a, b) => a * b, 1) || 1;
        const data =
            s.init ??
            (s.gen === "range" && s.shape.length === 0 && typeof s.value === "number"
                ? [s.value]
                : makeArray(s.dtype, len, s.gen ?? "random", rng));

        let typed: ort.Tensor.DataTypeMap[ort.Tensor.Type];
        if (s.dtype === "float32") typed = new Float32Array(data as number[]);
        else if (s.dtype === "int32") typed = new Int32Array(data as number[]);
        else if (s.dtype === "int64")
            typed = BigInt64Array.from((data as (number | bigint)[]).map(BigInt));
        else if (s.dtype === "uint8") typed = Uint8Array.from(data as number[]);
        else if (s.dtype === "int8") typed = Int8Array.from(data as number[]);
        else typed = new Uint8Array((data as boolean[]).map((v) => (v ? 1 : 0)));

        out[s.name] = new ort.Tensor(s.dtype, typed, s.shape);
    }
    return out;
}

export function resolveModelPath(rel: string): string {
    if (path.isAbsolute(rel)) return rel;
    const fromCwd = path.resolve(process.cwd(), rel);
    if (fs.existsSync(fromCwd)) return fromCwd;

    const fromRoot = path.resolve(PROJECT_ROOT, rel);
    if (fs.existsSync(fromRoot)) return fromRoot;

    return fromRoot;
}

export function getReconvertedPath(originalPath: string): string {
    const extIndex = originalPath.lastIndexOf(".");
    const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
    return `${base}_reconverted.onnx`;
}

export interface CompareOutputsResult {
    passed: boolean;
    maxDiff: number;
    maxRelDiff: number;
    message?: string;
}

export function compareOrtOutputs(
    origOut: Record<string, ort.Tensor>,
    recOut: Record<string, ort.Tensor>,
    options: {
        tol?: number | Record<string, number>;
        rtol?: number;
        exact?: boolean;
    } = {},
): CompareOutputsResult {
    const origKeys = Object.keys(origOut);
    const recKeys = Object.keys(recOut);
    if (origKeys.length !== recKeys.length) {
        return {
            passed: false,
            maxDiff: Infinity,
            maxRelDiff: Infinity,
            message: `Output count mismatch: expected ${origKeys.length} (${origKeys.join(", ")}), got ${recKeys.length} (${recKeys.join(", ")})`,
        };
    }

    const rtol = options.rtol ?? 0;
    let globalMaxDiff = 0;
    let globalMaxRelDiff = 0;

    for (const key of origKeys) {
        const origTensor = origOut[key];
        if (key in recOut === false) {
            return {
                passed: false,
                maxDiff: Infinity,
                maxRelDiff: Infinity,
                message: `Missing output '${key}' in reconverted model`,
            };
        }
        const recTensor = recOut[key];

        const origDims = Array.from(origTensor.dims).join(",");
        const recDims = Array.from(recTensor.dims).join(",");
        if (origDims !== recDims) {
            return {
                passed: false,
                maxDiff: Infinity,
                maxRelDiff: Infinity,
                message: `Shape mismatch on '${key}': [${origDims}] vs [${recDims}]`,
            };
        }

        const origArr = Array.from(origTensor.data as Iterable<number | bigint | boolean>);
        const recArr = Array.from(recTensor.data as Iterable<number | bigint | boolean>);

        if (options.exact === true) {
            const badIdx = origArr.findIndex((v, i) => v !== recArr[i]);
            if (badIdx !== -1) {
                return {
                    passed: false,
                    maxDiff: Infinity,
                    maxRelDiff: Infinity,
                    message: `Exact mismatch on '${key}'[${badIdx}]: ${String(origArr[badIdx])} vs ${String(recArr[badIdx])}`,
                };
            }
            continue;
        }

        const atol =
            typeof options.tol === "object" ? (options.tol[key] ?? 1e-4) : (options.tol ?? 1e-4);

        for (let i = 0; i < origArr.length; i++) {
            const o = Number(origArr[i]);
            const r = Number(recArr[i]);

            if (Number.isNaN(o) && Number.isNaN(r)) continue;
            if (Number.isNaN(o) !== Number.isNaN(r)) {
                return {
                    passed: false,
                    maxDiff: NaN,
                    maxRelDiff: NaN,
                    message: `NaN mismatch on '${key}'[${i}]: original=${o}, reconverted=${r}`,
                };
            }
            if (!Number.isFinite(o) || !Number.isFinite(r)) {
                if (o !== r) {
                    return {
                        passed: false,
                        maxDiff: Infinity,
                        maxRelDiff: Infinity,
                        message: `Infinity mismatch on '${key}'[${i}]: original=${o}, reconverted=${r}`,
                    };
                }
                continue;
            }

            const diff = Math.abs(o - r);
            const relDiff = o === 0 ? diff : diff / Math.abs(o);
            if (diff > globalMaxDiff) globalMaxDiff = diff;
            if (relDiff > globalMaxRelDiff) globalMaxRelDiff = relDiff;

            const allowed = atol + rtol * Math.abs(o);
            if (diff > allowed) {
                return {
                    passed: false,
                    maxDiff: diff,
                    maxRelDiff: relDiff,
                    message: `Mismatch on '${key}'[${i}]: orig=${o}, rec=${r}, diff=${diff} > allowed=${allowed} (atol=${atol}, rtol=${rtol})`,
                };
            }
        }
    }

    return { passed: true, maxDiff: globalMaxDiff, maxRelDiff: globalMaxRelDiff };
}

/* ============================== CORE HARNESS EXECUTION ================================== */
export async function executeEquivalenceTest(
    config: EquivalenceTestConfig,
    options: EquivalenceOptions = {},
): Promise<EquivalenceRunResult> {
    ensureProtoExists();
    const absOrig = resolveModelPath(config.originalPath);
    const safeLabel = config.label.replace(/[^a-zA-Z0-9_-]+/g, "_");
    const cliInputPath =
        options.workDir !== undefined && options.skipCli !== true
            ? path.join(options.workDir, `${safeLabel}_${path.basename(absOrig)}`)
            : absOrig;

    if (cliInputPath !== absOrig) {
        fs.mkdirSync(path.dirname(cliInputPath), { recursive: true });
        fs.copyFileSync(absOrig, cliInputPath);
    }

    const reconvertedPath = getReconvertedPath(cliInputPath);
    const cliArgs =
        typeof config.cliArgs === "function" ? config.cliArgs(cliInputPath) : config.cliArgs;
    const feeds = buildFeeds(config.specs, config.label);
    const tol = config.tol ?? 1e-4;

    // 1. CLI Execution
    if (options.skipCli !== true) {
        if (fs.existsSync(reconvertedPath)) {
            try {
                fs.unlinkSync(reconvertedPath);
            } catch {
                /* ignore */
            }
        }
        try {
            execSync(`node ./out/src/index.js "${cliInputPath}" ${cliArgs}`, {
                cwd: PROJECT_ROOT,
                stdio: "pipe",
            });
        } catch (e: unknown) {
            const errObj = e as { message?: string; stderr?: Buffer | string };
            const stderrText =
                errObj.stderr !== undefined ? `\n${errObj.stderr.toString().trim()}` : "";
            return {
                status: "error",
                message: `CLI generation failed: ${errObj.message ?? String(e)}${stderrText}`,
            };
        }
    }

    // 2. Optional Stats
    let origStats: GraphStats | null = null;
    let decompStats: GraphStats | null = null;
    if (options.showStats === true) {
        origStats = await getGraphStats(absOrig);
        decompStats = await getGraphStats(reconvertedPath);
        printStatsComparison(config.label, origStats, decompStats);
    }

    // 3. Inference
    let origOut: Record<string, ort.Tensor>;
    try {
        const origSess = await ort.InferenceSession.create(absOrig);
        origOut = await origSess.run(feeds);
    } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return { status: "error", message: `Original ORT inference failed: ${msg}` };
    }

    let recOut: Record<string, ort.Tensor>;
    try {
        const recSess = await ort.InferenceSession.create(reconvertedPath);
        recOut = await recSess.run(feeds);
    } catch (e: unknown) {
        if (options.verboseOnFailure !== false) {
            await rerunWithOrtVerbose(reconvertedPath, feeds);
        }
        const msg = e instanceof Error ? e.message : String(e);
        return { status: "error", message: `Reconverted ORT inference failed: ${msg}` };
    }

    // 4. Compare ALL Outputs (names, shapes, NaN/Inf, exact or tolerance)
    const cmp = compareOrtOutputs(origOut, recOut, {
        tol,
        rtol: config.rtol ?? 0,
        exact: config.exact ?? false,
    });

    const result: EquivalenceRunResult = {
        status: cmp.passed ? "pass" : "warn",
        maxDiff: cmp.maxDiff,
        originalStats: origStats,
        decomposedStats: decompStats,
    };
    if (cmp.message !== undefined) {
        result.message = cmp.message;
    }
    return result;
}
