#!/usr/bin/env node
import * as ort from "onnxruntime-web";
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { onnx2json } from "../src/onnx2json.js";
import { createGraph } from "../src/initGraph.js";
import type OnnxGraph from "../src/Onnx/OnnxGraph.js";

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
    exact?: boolean;
    cliArgs: string | ((p: string) => string);
    specs: FeedSpec[];
}

export interface EquivalenceOptions {
    skipCli?: boolean;
    showStats?: boolean;
    verboseOnFailure?: boolean;
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
        stats.ops[node.type] = (stats.ops[node.type] || 0) + 1;
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

/* ============================== ORT VERBOSE DIAGNOSTICS ================================== */
export async function rerunWithOrtVerbose(
    modelPath: string,
    feeds: Record<string, ort.Tensor>,
): Promise<void> {
    console.log("\n  [Diagnostic] Re-running model with ORT verbose WebAssembly logging...");
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
    }
}

/* ============================== FEED GENERATORS ================================== */
function randFloat() {
    return Math.random() * 10;
}
function randNegMix() {
    return Math.random() * 2 - 1;
}
function randInt(max = 10) {
    return Math.floor(Math.random() * max);
}
function randBool() {
    return Math.random() < 0.5;
}

export function makeArray(
    dtype: DType,
    len: number,
    gen: FeedSpec["gen"],
): number[] | bigint[] | boolean[] {
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

export function buildFeeds(specs: FeedSpec[]): Record<string, ort.Tensor> {
    const out: Record<string, ort.Tensor> = {};
    for (const s of specs) {
        const len = s.shape.reduce((a, b) => a * b, 1) || 1;
        const data =
            s.init ??
            (s.gen === "range" && s.shape.length === 0 && typeof s.value === "number"
                ? [s.value]
                : makeArray(s.dtype, len, s.gen ?? "random"));

        let typed: ort.Tensor.DataTypeMap[ort.Tensor.Type];
        if (s.dtype === "float32") typed = new Float32Array(data as number[]);
        else if (s.dtype === "int32") typed = new Int32Array(data as number[]);
        else if (s.dtype === "int64") typed = BigInt64Array.from(data as bigint[]);
        else if (s.dtype === "uint8") typed = Uint8Array.from(data as number[]);
        else if (s.dtype === "int8") typed = Int8Array.from(data as number[]);
        else typed = new Uint8Array((data as boolean[]).map((v) => (v ? 1 : 0)));

        out[s.name] = new ort.Tensor(s.dtype, typed, s.shape);
    }
    return out;
}

export function resolveModelPath(rel: string): string {
    if (path.isAbsolute(rel)) return rel;
    // 1. Check relative to process working directory (where npm test executes)
    const fromCwd = path.resolve(process.cwd(), rel);
    if (fs.existsSync(fromCwd)) return fromCwd;

    // 2. Check relative to this file's root directory
    const repoRoot = __dirname.includes("out")
        ? path.resolve(__dirname, "../..")
        : path.resolve(__dirname, "..");
    const fromRepo = path.resolve(repoRoot, rel);
    if (fs.existsSync(fromRepo)) return fromRepo;

    return fromCwd;
}

export function getReconvertedPath(originalPath: string): string {
    const extIndex = originalPath.lastIndexOf(".");
    const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
    return `${base}_reconverted.onnx`;
}

/* ============================== CORE HARNESS EXECUTION ================================== */
export async function executeEquivalenceTest(
    config: EquivalenceTestConfig,
    options: EquivalenceOptions = {},
): Promise<EquivalenceRunResult> {
    ensureProtoExists();
    const absOrig = resolveModelPath(config.originalPath);
    const reconvertedPath = getReconvertedPath(absOrig);
    const cliArgs =
        typeof config.cliArgs === "function" ? config.cliArgs(config.originalPath) : config.cliArgs;
    const feeds = buildFeeds(config.specs);
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
            execSync(`node ./out/src/index.js "${absOrig}" ${cliArgs}`, {
                cwd: PROJECT_ROOT,
                stdio: "pipe",
            });
        } catch (e: unknown) {
            const isError = e instanceof Error;
            const msg = isError ? e.message : String(e);
            return { status: "error", message: `CLI generation failed: ${msg}` };
        }
    }

    if (!fs.existsSync(reconvertedPath)) {
        return { status: "error", message: `Reconverted model missing at ${reconvertedPath}` };
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
        const isError = e instanceof Error;
        const msg = isError ? e.message : String(e);
        return { status: "error", message: `CLI generation failed: ${msg}` };
    }

    let recOut: Record<string, ort.Tensor>;
    try {
        const recSess = await ort.InferenceSession.create(reconvertedPath);
        recOut = await recSess.run(feeds);
    } catch (e: unknown) {
        if (options.verboseOnFailure !== false) {
            await rerunWithOrtVerbose(reconvertedPath, feeds);
        }
        const isError = e instanceof Error;
        const msg = isError ? e.message : String(e);
        return { status: "error", message: `CLI generation failed: ${msg}` };
    }

    // 4. Compare Outputs
    const origArr = Array.from(
        Object.values(origOut)[0].data as Iterable<number | bigint | boolean>,
    );
    const recArr = Array.from(Object.values(recOut)[0].data as Iterable<number | bigint | boolean>);

    if (origArr.length !== recArr.length) {
        return {
            status: "warn",
            message: `Length mismatch: ${origArr.length} vs ${recArr.length}`,
        };
    }

    // Exact comparison branch (handles bigints, ints, and booleans without Math.abs)
    if (config.exact === true) {
        const passed = origArr.every((v, i) => v === recArr[i]);
        return {
            status: passed ? "pass" : "warn",
            message: passed ? undefined : "Outputs do not match exactly",
            originalStats: origStats,
            decomposedStats: decompStats,
        } as EquivalenceRunResult;
    }

    // Tolerance comparison branch (coerces to Number safely)
    let maxDiff = 0;
    for (let i = 0; i < origArr.length; i++) {
        const o = Number(origArr[i]);
        const r = Number(recArr[i]);
        const diff = Math.abs(o - r);
        if (diff > maxDiff) maxDiff = diff;
    }

    const passed = maxDiff <= tol;
    return {
        status: passed ? "pass" : "warn",
        maxDiff,
        message: passed ? undefined : `Max difference ${maxDiff} exceeds tolerance ${tol}`,
        originalStats: origStats,
        decomposedStats: decompStats,
    } as EquivalenceRunResult;
}

/* ============================== CLI HOOK FOR index.ts ================================== */
export async function runEquivalenceForOriginalPath(
    originalPath: string,
    options?: { labelHint?: string; skipCli?: boolean },
): Promise<SingleEquivResult> {
    const target = path.resolve(originalPath).replace(/\\/g, "/");
    const all = [...TESTS, ...CORE_OP_TESTS];
    const config =
        all.find(
            (t) => path.resolve(resolveModelPath(t.originalPath)).replace(/\\/g, "/") === target,
        ) ||
        (options?.labelHint !== undefined && options.labelHint !== ""
            ? all.find((t) => t.label === options.labelHint)
            : undefined);

    if (!config) return "no-config";

    const res = await executeEquivalenceTest(config, {
        skipCli: options?.skipCli ?? false,
        showStats: true,
        verboseOnFailure: true,
    });
    return res.status;
}

/* ============================== TEST MODELS REGISTRY ================================== */
export const jsonFullArgs = "--format json -vz 0 -v 0";
export const jsonNoLowLevelArgs = "--format json --noLowLevel -vz 0 -v 0";
export const dotFullArgs = (p: string): string =>
    `--format dot -vz 0 -v 0 -o ${p.replace(/\.onnx$/, "")}.dot`;
export const dotNoLowLevelArgs = (p: string): string =>
    `--noLowLevel --format dot -vz 0 -v 0 -o ${p.replace(/\.onnx$/, "")}.dot`;

export const TESTS: EquivalenceTestConfig[] = [
    // ===== Standard Reconversions (JSON, noLowLevel) =====
    {
        label: "vector_add_standard",
        originalPath: "examples/onnx/vector_add_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonNoLowLevelArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "add_chain_standard",
        originalPath: "examples/onnx/add_chain_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonNoLowLevelArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
            { name: "C", dtype: "float32", shape: [4] },
            { name: "D", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "matmul_standard",
        originalPath: "examples/onnx/matmul_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonNoLowLevelArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 2] },
            { name: "B", dtype: "float32", shape: [2, 2] },
        ],
    },
    {
        label: "matmul_add_standard",
        originalPath: "examples/onnx/matmul_add_standard.onnx",
        exact: true,
        cliArgs: jsonNoLowLevelArgs,
        specs: [
            { name: "X", dtype: "int32", shape: [3, 1] },
            { name: "A", dtype: "int32", shape: [1, 3] },
            { name: "B", dtype: "int32", shape: [3, 3] },
        ],
    },

    // ===== Decomposed Reconversions (DOT, noLowLevel, need loop controls) =====
    {
        label: "vector_add_decomposed",
        originalPath: "examples/onnx/vector_add_decomposed.onnx",
        tol: 1e-5,
        cliArgs: (p) => dotNoLowLevelArgs(p),
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
            { name: "trip_count", dtype: "int64", shape: [], init: [BigInt(4)] },
            { name: "cond", dtype: "bool", shape: [], init: [true] },
        ],
    },
    {
        label: "add_chain_decomposed",
        originalPath: "examples/onnx/add_chain_decomposed.onnx",
        tol: 1e-5,
        cliArgs: (p) => dotNoLowLevelArgs(p),
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
            { name: "C", dtype: "float32", shape: [4] },
            { name: "D", dtype: "float32", shape: [4] },
            { name: "trip_count", dtype: "int64", shape: [], init: [BigInt(4)] },
            { name: "cond", dtype: "bool", shape: [], init: [true] },
        ],
    },
    {
        label: "matmul_decomposed",
        originalPath: "examples/onnx/matmul_decomposed.onnx",
        tol: 1e-5,
        cliArgs: (p) => dotNoLowLevelArgs(p),
        specs: [
            { name: "A", dtype: "float32", shape: [2, 2] },
            { name: "B", dtype: "float32", shape: [2, 2] },
            { name: "trip_count", dtype: "int64", shape: [], init: [BigInt(4)] },
            { name: "cond", dtype: "bool", shape: [], init: [true] },
        ],
    },
    {
        label: "matmul_add_decomposed",
        originalPath: "examples/onnx/matmul_add_decomposed.onnx",
        exact: true,
        cliArgs: (p) => dotNoLowLevelArgs(p),
        specs: [
            { name: "X", dtype: "int32", shape: [3, 1] },
            { name: "A", dtype: "int32", shape: [1, 3] },
            { name: "B", dtype: "int32", shape: [3, 3] },
            { name: "trip_count", dtype: "int64", shape: [], init: [BigInt(9)] },
            { name: "cond", dtype: "bool", shape: [], init: [true] },
        ],
    },

    // ===== Complete Suite (Full Decomposition with DOTs) =====
    {
        label: "vectoradd_test",
        originalPath: "examples/onnx/vectoradd_test.onnx",
        tol: 1e-5,
        cliArgs: (p) => dotFullArgs(p),
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "addchain_test",
        originalPath: "examples/onnx/addchain_test.onnx",
        tol: 1e-5,
        cliArgs: (p) => dotFullArgs(p),
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
            { name: "C", dtype: "float32", shape: [4] },
            { name: "D", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "matmul_test",
        originalPath: "examples/onnx/matmul_test.onnx",
        tol: 1e-5,
        cliArgs: (p) => dotFullArgs(p),
        specs: [
            { name: "A", dtype: "float32", shape: [2, 2] },
            { name: "B", dtype: "float32", shape: [2, 2] },
        ],
    },
    {
        label: "matmuladd_test",
        originalPath: "examples/onnx/matmuladd_test.onnx",
        exact: true,
        cliArgs: (p) => dotFullArgs(p),
        specs: [
            { name: "X", dtype: "int32", shape: [3, 3] },
            { name: "A", dtype: "int32", shape: [3, 3] },
            { name: "B", dtype: "int32", shape: [3, 3] },
        ],
    },

    // ===== Range / Transpose / MatMul+Transpose (JSON) =====
    {
        label: "range_standard",
        originalPath: "examples/onnx/range_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "start", dtype: "float32", shape: [], gen: "range", value: 0 },
            { name: "limit", dtype: "float32", shape: [], gen: "range", value: 5 },
            { name: "delta", dtype: "float32", shape: [], gen: "range", value: 1 },
        ],
    },
    {
        label: "range_add_standard",
        originalPath: "examples/onnx/range_add_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "start", dtype: "float32", shape: [], gen: "range", value: 1 },
            { name: "limit", dtype: "float32", shape: [], gen: "range", value: 6 },
            { name: "delta", dtype: "float32", shape: [], gen: "range", value: 1.5 },
            { name: "V", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "transpose_standard",
        originalPath: "examples/onnx/transpose_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3] }],
    },
    {
        label: "transpose_add_standard",
        originalPath: "examples/onnx/transpose_add_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 3] },
            { name: "Y", dtype: "float32", shape: [3, 2] },
        ],
    },
    {
        label: "matmul_transpose_standard",
        originalPath: "examples/onnx/matmul_transpose_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [3, 4] },
        ],
    },

    // ===== Activations (JSON) =====
    {
        label: "relu_standard",
        originalPath: "examples/onnx/relu_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [6], gen: "negmix" }],
    },
    {
        label: "sigmoid_standard",
        originalPath: "examples/onnx/sigmoid_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [6], gen: "negmix" }],
    },
    {
        label: "tanh_standard",
        originalPath: "examples/onnx/tanh_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [6], gen: "negmix" }],
    },
    {
        label: "exp_standard",
        originalPath: "examples/onnx/exp_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [6] }],
    },

    // ===== Unary & Binary Combinations =====
    {
        label: "unary_binary_combo",
        originalPath: "examples/onnx/unary_binary_combo.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [6] },
            { name: "A", dtype: "float32", shape: [6] },
            { name: "B", dtype: "float32", shape: [6] },
            { name: "Y", dtype: "float32", shape: [6] },
            { name: "S", dtype: "float32", shape: [6] },
        ],
    },
    {
        label: "sum_standard",
        originalPath: "examples/onnx/sum_variadic.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [1, 3] },
            { name: "C", dtype: "float32", shape: [] },
        ],
    },
    {
        label: "add_scalar_vector_broadcast",
        originalPath: "examples/onnx/add_scalar_vector.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [6] },
            { name: "S", dtype: "float32", shape: [] },
        ],
    },
    {
        label: "add_row_vector_to_matrix_broadcast",
        originalPath: "examples/onnx/add_row_vector_to_matrix.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [3] },
        ],
    },
    {
        label: "add_col_vector_to_matrix_broadcast",
        originalPath: "examples/onnx/add_col_vector_to_matrix.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "C", dtype: "float32", shape: [2, 1] },
        ],
    },
    {
        label: "mul_3d_channel",
        originalPath: "examples/onnx/mul_3d_channel.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 3, 4] },
            { name: "W", dtype: "float32", shape: [1, 3, 1] },
        ],
    },
    {
        label: "chain_broadcast",
        originalPath: "examples/onnx/chain_broadcast.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "b_row", dtype: "float32", shape: [3] },
            { name: "c_col", dtype: "float32", shape: [2, 1] },
            { name: "s_sub", dtype: "float32", shape: [] },
            { name: "s_div", dtype: "float32", shape: [] },
        ],
    },

    // ===== Transpose + Broadcast (2D..5D) =====
    {
        label: "transpose_broadcast_2d",
        originalPath: "examples/onnx/transpose_broadcast_2d.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 3] },
            { name: "Y", dtype: "float32", shape: [3] },
        ],
    },
    {
        label: "transpose_broadcast_3d",
        originalPath: "examples/onnx/transpose_broadcast_3d.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 1, 3] },
            { name: "Zin", dtype: "float32", shape: [1, 3, 1] },
        ],
    },
    {
        label: "transpose_broadcast_4d",
        originalPath: "examples/onnx/transpose_broadcast_4d.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 1, 3, 1] },
            { name: "B", dtype: "float32", shape: [3, 1, 1, 1] },
        ],
    },
    {
        label: "transpose_broadcast_5d",
        originalPath: "examples/onnx/transpose_broadcast_5d.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 2, 1, 3, 1] },
            { name: "C", dtype: "float32", shape: [1, 3, 1, 1, 1] },
        ],
    },

    // ===== MatMul Broadcast =====
    {
        label: "matmul_bcast_left_unbatched",
        originalPath: "examples/onnx/matmul_bcast_left_unbatched.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3, 4] },
            { name: "B", dtype: "float32", shape: [4, 5] },
        ],
    },
    {
        label: "matmul_bcast_both_sides",
        originalPath: "examples/onnx/matmul_bcast_both_sides.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [1, 3, 4] },
            { name: "B", dtype: "float32", shape: [2, 4, 5] },
        ],
    },
    {
        label: "matmul_bcast_highrank",
        originalPath: "examples/onnx/matmul_bcast_highrank.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 1, 3, 4] },
            { name: "B", dtype: "float32", shape: [1, 5, 4, 6] },
        ],
    },

    // ===== Slice, Pad, Clip, Conv =====
    {
        label: "slice_decomposition",
        originalPath: "examples/onnx/slice.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [1, 2, 5, 6] }],
    },
    {
        label: "pad_decomposition",
        originalPath: "examples/onnx/pad_normal.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [1, 2, 3, 4] }],
    },
    {
        label: "clip_scalar",
        originalPath: "examples/onnx/clip_scalar.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 3] },
            { name: "Min", dtype: "float32", shape: [] },
            { name: "Max", dtype: "float32", shape: [] },
        ],
    },
    {
        label: "conv_normal",
        originalPath: "examples/onnx/conv_normal.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 1, 4, 4] },
            { name: "W", dtype: "float32", shape: [1, 1, 3, 3] },
            { name: "B", dtype: "float32", shape: [1] },
        ],
    },
    {
        label: "conv_simple",
        originalPath: "examples/onnx/conv_simple.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 1, 4, 4] },
            { name: "W", dtype: "float32", shape: [1, 1, 3, 3] },
            { name: "B", dtype: "float32", shape: [1] },
        ],
    },

    // ===== Gemm, Concat, Dequantize, AvgPool =====
    {
        label: "gemm_standard",
        originalPath: "examples/onnx/gemm_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [3, 4] },
            { name: "C", dtype: "float32", shape: [2, 4] },
        ],
    },
    {
        label: "concat_standard",
        originalPath: "examples/onnx/concat_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X0", dtype: "float32", shape: [2, 3] },
            { name: "X1", dtype: "float32", shape: [2, 4] },
            { name: "X2", dtype: "float32", shape: [2, 2] },
        ],
    },
    {
        label: "dequantize_standard",
        originalPath: "examples/onnx/dequantize_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "uint8", shape: [2, 3, 4] },
            { name: "S", dtype: "float32", shape: [3] },
            { name: "Z", dtype: "uint8", shape: [3] },
        ],
    },
    {
        label: "averagepool_standard",
        originalPath: "examples/onnx/avgpool_standard.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [1, 2, 5, 6] }],
    },

    // ===== Reduce Operators =====
    {
        label: "reducesum_standard",
        originalPath: "examples/onnx/reducesum_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
    },
    {
        label: "reducemax_standard",
        originalPath: "examples/onnx/reducemax_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
    },
    {
        label: "reducemin_standard",
        originalPath: "examples/onnx/reducemin_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "negmix" }],
    },
    {
        label: "reduceprod_standard",
        originalPath: "examples/onnx/reduceprod_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        label: "reducemean_standard",
        originalPath: "examples/onnx/reducemean_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "random" }],
    },
    {
        label: "reducesumsquare_standard",
        originalPath: "examples/onnx/reducesumsquare_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "random" }],
    },
    {
        label: "reducel1_standard",
        originalPath: "examples/onnx/reducel1_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "random" }],
    },
    {
        label: "reducel2_standard",
        originalPath: "examples/onnx/reducel2_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "random" }],
    },
    {
        label: "reducelogsum_standard",
        originalPath: "examples/onnx/reducelogsum_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "random" }],
    },
    {
        label: "reducelogsumexp_standard",
        originalPath: "examples/onnx/reducelogsumexp_standard.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4], gen: "random" }],
    },

    // ===== Softmax & Expand =====
    {
        label: "softmax_standard",
        originalPath: "examples/onnx/softmax_standard.onnx",
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [8, 3], gen: "random" }],
        tol: 1e-5,
    },
    {
        label: "expand_scalar_to_2x3",
        originalPath: "examples/onnx/expand_scalar_to_2x3.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [], gen: "random" }],
    },
    {
        label: "expand_vec_to_2x3",
        originalPath: "examples/onnx/expand_vec_to_2x3.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [3], gen: "random" }],
    },
    {
        label: "expand_batch",
        originalPath: "examples/onnx/expand_batch.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [1, 4, 5], gen: "random" }],
    },
    {
        label: "expand_middle_dim",
        originalPath: "examples/onnx/expand_middle_dim.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [2, 1, 4], gen: "random" }],
    },
    {
        label: "expand_highrank",
        originalPath: "examples/onnx/expand_highrank.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [1, 3, 1, 5], gen: "random" }],
    },

    // ===== TinyML & Pipeline Models =====
    {
        label: "ad01_fp32_standard",
        originalPath: "examples/onnx/ad01_fp32.onnx",
        tol: 1e-3,
        cliArgs: jsonFullArgs,
        specs: [{ name: "input_1", dtype: "float32", shape: [1, 640] }],
    },
    {
        label: "ad01_fp32_gemm_relu_standard",
        originalPath: "examples/onnx/ad01_fp32_gemm_relu.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "input_1", dtype: "float32", shape: [1, 640] }],
    },
    {
        label: "averagepool_kws_like",
        originalPath: "examples/onnx/avgpool_kws_like.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "X", dtype: "float32", shape: [1, 64, 25, 5] }],
    },
    {
        label: "SC2_X",
        originalPath: "examples/onnx/SC2_X_toy.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "input", dtype: "float32", shape: [2, 1] }],
    },
    {
        label: "SC2_Y",
        originalPath: "examples/onnx/SC2_Y_toy.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "input", dtype: "float32", shape: [2, 1] }],
    },
    {
        label: "SC2_Z",
        originalPath: "examples/onnx/SC2_Z_toy.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "input", dtype: "float32", shape: [2, 1] }],
    },

    // ===== Canonicalization Elemental Ops =====
    {
        label: "Sub",
        originalPath: "examples/onnx/sub.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2] },
            { name: "B", dtype: "float32", shape: [2] },
        ],
    },
    {
        label: "GreaterOrEqual",
        originalPath: "examples/onnx/greater_or_equal.onnx",
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2] },
            { name: "B", dtype: "float32", shape: [2] },
        ],
    },
    {
        label: "Less",
        originalPath: "examples/onnx/less.onnx",
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2], init: [10, -10] },
            { name: "B", dtype: "float32", shape: [2], init: [3, 3] },
        ],
    },
    {
        label: "LessOrEqual",
        originalPath: "examples/onnx/less_or_equal.onnx",
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2] },
            { name: "B", dtype: "float32", shape: [2] },
        ],
    },
    {
        label: "Min",
        originalPath: "examples/onnx/min.onnx",
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2] },
            { name: "B", dtype: "float32", shape: [2] },
        ],
    },
    {
        label: "Max",
        originalPath: "examples/onnx/max.onnx",
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2] },
            { name: "B", dtype: "float32", shape: [2] },
        ],
    },
    {
        label: "Abs",
        originalPath: "examples/onnx/abs.onnx",
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Sign",
        originalPath: "examples/onnx/sign.onnx",
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "LeakyRelu",
        originalPath: "examples/onnx/leaky_relu.onnx",
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "HardSigmoid",
        originalPath: "examples/onnx/hard_sigmoid.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Softplus",
        originalPath: "examples/onnx/softplus.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Mish",
        originalPath: "examples/onnx/mish.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Elu",
        originalPath: "examples/onnx/elu.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Celu",
        originalPath: "examples/onnx/celu.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Selu",
        originalPath: "examples/onnx/selu.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Or",
        originalPath: "examples/onnx/or.onnx",
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "bool", shape: [2] },
            { name: "B", dtype: "bool", shape: [2] },
        ],
    },
    {
        label: "Div",
        originalPath: "examples/onnx/Div.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2] },
            { name: "B", dtype: "float32", shape: [2] },
        ],
    },
    {
        label: "Mod fmod0",
        originalPath: "examples/onnx/mod_fmod0.onnx",
        exact: true,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "int64", shape: [6], init: [10n, 10n, -10n, -10n, 5n, -5n] },
            { name: "B", dtype: "int64", shape: [6], init: [3n, -3n, 3n, -3n, 2n, 2n] },
        ],
    },
    {
        label: "Mod fmod1",
        originalPath: "examples/onnx/mod_fmod1.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [6], gen: "negmix" },
            { name: "B", dtype: "float32", shape: [6], gen: "negmix" },
        ],
    },
    {
        label: "Pow",
        originalPath: "examples/onnx/pow.onnx",
        tol: 1e-2,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "A", dtype: "float32", shape: [2], init: [2.0, 3.0] },
            { name: "B", dtype: "float32", shape: [2], init: [2.0, 3.0] },
        ],
    },
    {
        label: "Not",
        originalPath: "examples/onnx/not.onnx",
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "bool", shape: [2] }],
    },
    {
        label: "Sigmoid",
        originalPath: "examples/onnx/sigmoid.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Softsign",
        originalPath: "examples/onnx/softsign.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Tan",
        originalPath: "examples/onnx/tan.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Sinh",
        originalPath: "examples/onnx/sinh.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Cosh",
        originalPath: "examples/onnx/cosh.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "TanhSigmoid",
        originalPath: "examples/onnx/tanh_sigmoid.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "TanhDiv",
        originalPath: "examples/onnx/tanh_div.onnx",
        tol: 1e-6,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Asinh",
        originalPath: "examples/onnx/asinh.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Acosh",
        originalPath: "examples/onnx/acosh.onnx",
        tol: 1e-4,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "greaterThanOne" }],
    },
    {
        label: "Atanh",
        originalPath: "examples/onnx/atanh.onnx",
        tol: 1e-1,
        cliArgs: jsonFullArgs,
        specs: [{ name: "A", dtype: "float32", shape: [2], gen: "negmix" }],
    },
    {
        label: "Group Normalization",
        originalPath: "examples/onnx/group_normalization.onnx",
        tol: 1e-3,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 4, 4, 4] },
            { name: "scale", dtype: "float32", shape: [4] },
            { name: "bias", dtype: "float32", shape: [4] },
        ],
    },
];

export const CORE_OP_TESTS: EquivalenceTestConfig[] = [
    {
        label: "Group Normalization",
        originalPath: "examples/onnx/group_normalization.onnx",
        tol: 1e-5,
        cliArgs: jsonFullArgs,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 4, 4, 4] },
            { name: "scale", dtype: "float32", shape: [4] },
            { name: "bias", dtype: "float32", shape: [4] },
        ],
    },
];
