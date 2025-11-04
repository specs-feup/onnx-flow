import * as ort from 'onnxruntime-web';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { performance } from 'perf_hooks';


const RUN_TOTAL = { passed: 0, failed: 0 };
const CURRENT = { test: '', reconverted: '' } as { test: string; reconverted: string };

// Failure buckets we’ll print at the end
const FAILURES = {
  warn: [] as Array<{ test: string; path: string }>,
  error: [] as Array<{ test: string; path: string }>,
};

/* ============================== HELPERS ================================== */

function printInputs(label: string, inputs: Record<string, ort.Tensor>) {
  console.log(`Inputs for ${label}:`);
  for (const [key, tensor] of Object.entries(inputs)) {
    const dataArray = Array.from(tensor.data as Iterable<number>);
    console.log(`  ${key}:`, dataArray);
  }
}

function logErrorDetails(context: string, e: any) {
  console.error(`❌ Error in ${context}:`, e?.message || e);
  if (e?.stack) console.error(e.stack);
  if (e && typeof e === 'object') {
    const props = Object.getOwnPropertyNames(e);
    for (const prop of props) {
      if (prop !== 'message' && prop !== 'stack') {
        console.error(`  ${prop}:`, (e as any)[prop]);
      }
    }
  }
}

function getReconvertedPath(originalPath: string): string {
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  return `${base}_reconverted.onnx`;
}

function generateReconvertedNow(originalPath: string, cliArgs: string) {
  const reconvertedPath = getReconvertedPath(originalPath);
  const tStart = Date.now();

  // If an old reconverted exists, remove it so a failed generation can't be masked.
  if (fs.existsSync(reconvertedPath)) {
    try { fs.unlinkSync(reconvertedPath); } catch {}
  }

  // Run the CLI (will throw on non-zero exit)
  execSync(`node ./out/src/index.js ${originalPath} ${cliArgs}`, { stdio: 'inherit' });

  // Confirm the file exists and was (re)created after we started.
  if (!fs.existsSync(reconvertedPath)) return { generatedNow: false, reconvertedPath };
  const stat = fs.statSync(reconvertedPath);
  return { generatedNow: stat.mtimeMs >= tStart, reconvertedPath };
}

function reportOutcome(status: 'pass' | 'warn' | 'error', label = 'Outputs equivalent', value?: boolean) {
  const icon = status === 'pass' ? '✅' : status === 'warn' ? '⚠️' : '❌';
  console.log(`${icon} ${label}:`, value ?? (status === 'pass'));
  if (status === 'pass') RUN_TOTAL.passed++; else RUN_TOTAL.failed++;

  if (status === 'warn') recordFailure('warn');
  if (status === 'error') recordFailure('error');

  if (status !== 'pass') {
    if (CURRENT.test)        console.log(`   test: ${CURRENT.test}`);
    if (CURRENT.reconverted) console.log(`   reconverted: ${CURRENT.reconverted}`);
  }
}

type SafeTimedRun<T = Record<string, ort.Tensor>> = {
  out: T | undefined;
  ms: number;
  error?: unknown;
};

async function safeTimedRun(
  session: ort.InferenceSession,
  feeds: Record<string, ort.Tensor>
): Promise<SafeTimedRun> {
  try {
    const r = await timedRun(session, feeds);
    return { out: r.out, ms: r.ms };          // success: no error field
  } catch (error) {
    return { out: undefined, ms: 0, error };  // failure: include error
  }
}

async function timedRun(session: ort.InferenceSession, feeds: Record<string, ort.Tensor>) {
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  return { out, ms };
}

async function rerunWithOrtVerbose(modelPath: string, feeds: Record<string, ort.Tensor>): Promise<void> {
  console.log('🔁 Re-running reconverted model with ORT verbose logging…');

  try {
    ort.env.debug = true;
    ort.env.logLevel = 'verbose';

    const so: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      logSeverityLevel: 0,
      logVerbosityLevel: 4,
    };
    const sess = await ort.InferenceSession.create(modelPath, so);
    const outNames = sess.outputNames;
    const outs = await sess.run(feeds, outNames);
    const shapes = outNames.map(n => outs[n]?.dims ?? []);
    const dtypes = outNames.map(n => outs[n]?.type ?? 'unknown');
    console.log('   ✓ ORT (web) run OK. Outputs:', outNames.map((k, i) => `${k}[${shapes[i]}] ${dtypes[i]}`));
  } catch (e) {
    console.error('❌ Verbose re-run (web) failed:', e);
  }
}

// Helper to record a reconverted-path failure once
function recordFailure(kind: 'warn' | 'error') {
  if (!CURRENT.reconverted) return; // skip non-reconversion tests
  const arr = FAILURES[kind];
  // dedup by path+test to avoid spam across retries
  if (!arr.some(e => e.path === CURRENT.reconverted && e.test === CURRENT.test)) {
    arr.push({ test: CURRENT.test || '(unnamed test)', path: CURRENT.reconverted });
  }
}

/* ============================== FEED HELPERS ================================== */

type DType = 'uint8' | 'float32' | 'int32' | 'int64' | 'bool';
type FeedSpec = {
  name: string;
  dtype: DType;
  shape: number[];               // [] for scalars
  // optional fixed initialization (overrides generator)
  init?: number[] | bigint[] | boolean[];
  // optional generator hint
  gen?: 'random' | 'negmix' | 'zeros' | 'ones' | 'range';
  // for 'range' convenience (scalar-float specs)
  value?: number;                 // when shape === []
};

// random helpers
function randFloat() { return Math.random() * 10; }
function randNegMix() { return Math.random() * 2 - 1; }
function randInt(max = 10) { return Math.floor(Math.random() * max); }
function randBool() { return Math.random() < 0.5; }

function makeArray(dtype: DType, len: number, gen: FeedSpec['gen']): number[] | bigint[] | boolean[] {
  switch (dtype) {
    case 'float32':
      if (gen === 'negmix') return Array.from({ length: len }, randNegMix);
      if (gen === 'zeros')  return Array.from({ length: len }, () => 0);
      if (gen === 'ones')   return Array.from({ length: len }, () => 1);
      return Array.from({ length: len }, randFloat);
    case 'int32':
      if (gen === 'zeros')  return Array.from({ length: len }, () => 0);
      if (gen === 'ones')   return Array.from({ length: len }, () => 1);
      return Array.from({ length: len }, () => randInt(10));
    case 'int64':
      if (gen === 'zeros')  return Array.from({ length: len }, () => BigInt(0));
      if (gen === 'ones')   return Array.from({ length: len }, () => BigInt(1));
      return Array.from({ length: len }, () => BigInt(randInt(10)));
    case 'uint8': // <-- add this
      if (gen === 'zeros')  return Array.from({ length: len }, () => 0);
      if (gen === 'ones')   return Array.from({ length: len }, () => 1);
      return Array.from({ length: len }, () => randInt(256));
    case 'bool':
      if (gen === 'zeros')  return Array.from({ length: len }, () => false);
      if (gen === 'ones')   return Array.from({ length: len }, () => true);
      return Array.from({ length: len }, randBool);
  }
}

function buildFeeds(specs: FeedSpec[]): Record<string, ort.Tensor> {
  const out: Record<string, ort.Tensor> = {};
  for (const s of specs) {
    const len = s.shape.reduce((a, b) => a * b, 1) || 1;
    let data: number[] | bigint[] | boolean[];

    if (s.init) {
      data = s.init;
    } else if (s.gen === 'range' && s.shape.length === 0 && typeof s.value === 'number') {
      // scalar “range” convenience for start/limit/delta
      data = [s.value];
    } else {
      data = makeArray(s.dtype, len, s.gen ?? 'random');
    }

    // typed array materialization
    let typed: any;
    if (s.dtype === 'float32') typed = new Float32Array(data as number[]);
    else if (s.dtype === 'int32') typed = new Int32Array(data as number[]);
    else if (s.dtype === 'int64') typed = BigInt64Array.from(data as bigint[]);
    else if (s.dtype === 'uint8')   typed = Uint8Array.from(data as number[]);  // <-- add this
    else /* bool */ typed = new Uint8Array((data as boolean[]).map(v => (v ? 1 : 0)));

    out[s.name] = new ort.Tensor(s.dtype, typed, s.shape);
  }
  return out;
}

function jsonFullArgs() { return `--format json -vz 0 -v 0`; }
function dotFullArgs(originalPath: string, extra = ``) {
  const base = originalPath.replace(/\.onnx$/, '');
  return `--format dot -vz 0 -v 0 -o ${base}.dot ${extra}`.trim();
}
function dotNoLowLevelArgs(originalPath: string, extra = ``) {
  const base = originalPath.replace(/\.onnx$/, '');
  return `--noLowLevel --format dot -vz 0 -v 0 -o ${base}.dot ${extra}`.trim();
}
function jsonNoLowLevelArgs() { return `--format json --noLowLevel -vz 0 -v 0`; }


/* ============================== TEST ================================== */

/**
 * Unified reconversion test
 * - Builds reconverted model from `originalPath` (CLI args configurable)
 * - Runs & times original and reconverted with your existing helpers
 * - Compares outputs (exact or with tolerance), prints tri-state outcome
 * - On warn: auto verbose re-run; On error: counted and listed at the end
 */
async function testReconversion(opts: {
  label: string;
  originalPath: string;
  feeds: Record<string, ort.Tensor>;
  tol?: number;
  exact?: boolean;
  cliArgs?: string;
  skipCli?: boolean;
}) {
  const {
    label,
    originalPath,
    feeds,
    tol = 1e-5,
    exact = false,
    cliArgs = "--format json -vz 0 -v 0",
    skipCli = false,
  } = opts;

  const reconvertedPath = getReconvertedPath(originalPath);
  CURRENT.test = label;
  CURRENT.reconverted = reconvertedPath;

  try {
    let generatedNow = false;

    if (!skipCli) {
      console.log(`\n=== Running CLI to generate reconverted ${path.basename(originalPath)} model ===`);
      try {
        const r = generateReconvertedNow(originalPath, cliArgs);
        generatedNow = r.generatedNow;
      } catch (cliErr) {
        logErrorDetails('reconversion CLI', cliErr);
        if (fs.existsSync(reconvertedPath)) {
          try { fs.unlinkSync(reconvertedPath); } catch {}
        }
        reportOutcome('error', 'Reconversion CLI failed');
        return;
      }
    }

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      reportOutcome('error', 'Reconverted file missing');
      return;
    }

    console.log(`\n=== Comparing ${label} and its reconverted version ===`);
    printInputs(label, feeds);

    // Original
    const originalSession = await ort.InferenceSession.create(originalPath);
    const { out: originalOut, ms: __origMs } = await timedRun(originalSession, feeds);
    console.log(`⏱️ original: ${__origMs.toFixed(2)} ms`);

    // Reconverted: create session with explicit error handling
    let recSession: ort.InferenceSession;
    try {
      recSession = await ort.InferenceSession.create(reconvertedPath);
    } catch (createErr) {
      logErrorDetails('reconverted session create', createErr);
      reportOutcome('error', 'Reconverted session creation failed');

      // If we actually regenerated this file, do a verbose re-run attempt
      if (generatedNow) {
        try { await rerunWithOrtVerbose(reconvertedPath, feeds); } catch {}
      }
      return;
    }

    // Reconverted (safe run)
    const recRun = await safeTimedRun(recSession, feeds);
    if (recRun.error) {
      logErrorDetails('reconverted run', recRun.error);
      reportOutcome('error', 'Reconverted run failed');

      // Only verbose-rerun if this reconverted file was generated now
      if (generatedNow) {
        try { await rerunWithOrtVerbose(reconvertedPath, feeds); } catch {}
      }
      return;
    }

    const { out: reconvOut, ms: __recMs } = recRun;
    console.log(`⏱️ reconverted: ${__recMs.toFixed(2)} ms`);

    const originalArr = Array.from(Object.values(originalOut)[0].data as Iterable<number>);
    const reconvArr   = Array.from(Object.values(reconvOut)[0].data as Iterable<number>);

    console.log('→ original:', originalArr);
    console.log('→ reconverted:', reconvArr);

    let equivalent = false;
    if (exact) {
      equivalent = originalArr.length === reconvArr.length &&
                   originalArr.every((v, i) => v === reconvArr[i]);
    } else {
      equivalent = originalArr.length === reconvArr.length &&
                   originalArr.every((v, i) => Math.abs(v - (reconvArr[i] as number)) < tol);
    }

    if (equivalent) {
      reportOutcome('pass');
    } else {
      // Mismatch is only a warning; do NOT verbose re-run here.
      reportOutcome('warn');
    }
  } catch (err) {
    logErrorDetails(`${label} reconversion test`, err);
    reportOutcome('error', `${label} reconversion test failed`);
  }
}


/* ============================== TESTS ================================== */

const TESTS: Array<{
  label: string;
  originalPath: string;
  tol?: number;
  exact?: boolean;
  cliArgs: string | ((p: string) => string);
  specs: FeedSpec[];
}> = [
  // ===== Standard Reconversions (JSON, noLowLevel) =====
  /*
  {
    label: 'vector_add_standard',
    originalPath: 'examples/onnx/vector_add_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonNoLowLevelArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [4] },
      { name: 'B', dtype: 'float32', shape: [4] },
    ],
  },
  {
    label: 'add_chain_standard',
    originalPath: 'examples/onnx/add_chain_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonNoLowLevelArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [4] },
      { name: 'B', dtype: 'float32', shape: [4] },
      { name: 'C', dtype: 'float32', shape: [4] },
      { name: 'D', dtype: 'float32', shape: [4] },
    ],
  },
  {
    label: 'matmul_standard',
    originalPath: 'examples/onnx/matmul_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonNoLowLevelArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 2] },
      { name: 'B', dtype: 'float32', shape: [2, 2] },
    ],
  },
  {
    label: 'matmul_add_standard',
    originalPath: 'examples/onnx/matmul_add_standard.onnx',
    exact: true,
    cliArgs: jsonNoLowLevelArgs,
    specs: [
      { name: 'X', dtype: 'int32', shape: [3, 1] },
      { name: 'A', dtype: 'int32', shape: [1, 3] },
      { name: 'B', dtype: 'int32', shape: [3, 3] },
    ],
  },

  // ===== Decomposed Reconversions (DOT, noLowLevel, need loop controls) =====
  {
    label: 'vector_add_decomposed',
    originalPath: 'examples/onnx/vector_add_decomposed.onnx',
    tol: 1e-5,
    cliArgs: (p) => dotNoLowLevelArgs(p),
    specs: [
      { name: 'A', dtype: 'float32', shape: [4] },
      { name: 'B', dtype: 'float32', shape: [4] },
      { name: 'trip_count', dtype: 'int64', shape: [], init: [BigInt(4)] as any },
      { name: 'cond', dtype: 'bool', shape: [], init: [true] as any },
    ],
  },
  {
    label: 'add_chain_decomposed',
    originalPath: 'examples/onnx/add_chain_decomposed.onnx',
    tol: 1e-5,
    cliArgs: (p) => dotNoLowLevelArgs(p),
    specs: [
      { name: 'A', dtype: 'float32', shape: [4] },
      { name: 'B', dtype: 'float32', shape: [4] },
      { name: 'C', dtype: 'float32', shape: [4] },
      { name: 'D', dtype: 'float32', shape: [4] },
      { name: 'trip_count', dtype: 'int64', shape: [], init: [BigInt(4)] as any },
      { name: 'cond', dtype: 'bool', shape: [], init: [true] as any },
    ],
  },
  {
    label: 'matmul_decomposed',
    originalPath: 'examples/onnx/matmul_decomposed.onnx',
    tol: 1e-5,
    cliArgs: (p) => dotNoLowLevelArgs(p),
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 2] },
      { name: 'B', dtype: 'float32', shape: [2, 2] },
      { name: 'trip_count', dtype: 'int64', shape: [], init: [BigInt(4)] as any },
      { name: 'cond', dtype: 'bool', shape: [], init: [true] as any },
    ],
  },
  {
    label: 'matmul_add_decomposed',
    originalPath: 'examples/onnx/matmul_add_decomposed.onnx',
    exact: true,
    cliArgs: (p) => dotNoLowLevelArgs(p),
    specs: [
      { name: 'X', dtype: 'int32', shape: [3, 1] },
      { name: 'A', dtype: 'int32', shape: [1, 3] },
      { name: 'B', dtype: 'int32', shape: [3, 3] },
      { name: 'trip_count', dtype: 'int64', shape: [], init: [BigInt(9)] as any },
      { name: 'cond', dtype: 'bool', shape: [], init: [true] as any },
    ],
  },
  

  // ===== “Complete” suite (Full Decomposition with DOTs) =====
  {
    label: 'vectoradd_test',
    originalPath: 'examples/onnx/vectoradd_test.onnx',
    tol: 1e-5,
    cliArgs: (p) => dotFullArgs(p),
    specs: [
      { name: 'A', dtype: 'float32', shape: [4] },
      { name: 'B', dtype: 'float32', shape: [4] },
    ],
  },
  {
    label: 'addchain_test',
    originalPath: 'examples/onnx/addchain_test.onnx',
    tol: 1e-5,
    cliArgs: (p) => dotFullArgs(p),
    specs: [
      { name: 'A', dtype: 'float32', shape: [4] },
      { name: 'B', dtype: 'float32', shape: [4] },
      { name: 'C', dtype: 'float32', shape: [4] },
      { name: 'D', dtype: 'float32', shape: [4] },
    ],
  },
  {
    label: 'matmul_test',
    originalPath: 'examples/onnx/matmul_test.onnx',
    tol: 1e-5,
    cliArgs: (p) => dotFullArgs(p),
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 2] },
      { name: 'B', dtype: 'float32', shape: [2, 2] },
    ],
  },
  {
    label: 'matmuladd_test',
    originalPath: 'examples/onnx/matmuladd_test.onnx',
    exact: true,
    cliArgs: (p) => dotFullArgs(p),
    specs: [
      { name: 'X', dtype: 'int32', shape: [3, 3] },
      { name: 'A', dtype: 'int32', shape: [3, 3] },
      { name: 'B', dtype: 'int32', shape: [3, 3] },
    ],
  },

  // ===== Range / Transpose / Matmul+Transpose (JSON) =====
  {
    label: 'range_standard',
    originalPath: 'examples/onnx/range_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'start', dtype: 'float32', shape: [], gen: 'range', value: 0 },
      { name: 'limit', dtype: 'float32', shape: [], gen: 'range', value: 5 },
      { name: 'delta', dtype: 'float32', shape: [], gen: 'range', value: 1 },
    ],
  },
  {
    label: 'range_add_standard',
    originalPath: 'examples/onnx/range_add_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'start', dtype: 'float32', shape: [], gen: 'range', value: 1 },
      { name: 'limit', dtype: 'float32', shape: [], gen: 'range', value: 6 },
      { name: 'delta', dtype: 'float32', shape: [], gen: 'range', value: 1.5 },
      // L = ceil((6-1)/1.5) = 4
      { name: 'V', dtype: 'float32', shape: [4] },
    ],
  },
  {
    label: 'transpose_standard',
    originalPath: 'examples/onnx/transpose_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [2, 3] },
    ],
  },
  {
    label: 'transpose_add_standard',
    originalPath: 'examples/onnx/transpose_add_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [2, 3] },
      { name: 'Y', dtype: 'float32', shape: [3, 2] },
    ],
  },
  {
    label: 'matmul_transpose_standard',
    originalPath: 'examples/onnx/matmul_transpose_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'B', dtype: 'float32', shape: [3, 4] },
    ],
  },

  // ===== Activations (JSON) =====
  {
    label: 'relu_standard',
    originalPath: 'examples/onnx/relu_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [6], gen: 'negmix' }],
  },
  {
    label: 'sigmoid_standard',
    originalPath: 'examples/onnx/sigmoid_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [6], gen: 'negmix' }],
  },
  {
    label: 'tanh_standard',
    originalPath: 'examples/onnx/tanh_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [6], gen: 'negmix' }],
  },
  {
    label: 'exp_standard',
    originalPath: 'examples/onnx/exp_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [6] }],
  },

  // ── unary/binary combo
  {
    label: 'unary_binary_combo',
    originalPath: 'examples/onnx/unary_binary_combo.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [6] },
      { name: 'A', dtype: 'float32', shape: [6] },
      { name: 'B', dtype: 'float32', shape: [6] },
      { name: 'Y', dtype: 'float32', shape: [6] },
      { name: 'S', dtype: 'float32', shape: [6] },
    ],
  },

  // ── sum
  {
    label: 'sum_standard',
    originalPath: 'examples/onnx/sum_variadic.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'B', dtype: 'float32', shape: [1, 3] },
      { name: 'C', dtype: 'float32', shape: [] },
    ],
  },

  // ── broadcast add/mul
  {
    label: 'add_scalar_vector_broadcast',
    originalPath: 'examples/onnx/add_scalar_vector.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [6] },  // vector
      { name: 'S', dtype: 'float32', shape: [] },   // scalar
    ],
  },
  {
    label: 'add_row_vector_to_matrix_broadcast',
    originalPath: 'examples/onnx/add_row_vector_to_matrix.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] }, // matrix
      { name: 'B', dtype: 'float32', shape: [3] },    // row vector
    ],
  },
  {
    label: 'add_col_vector_to_matrix_broadcast',
    originalPath: 'examples/onnx/add_col_vector_to_matrix.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] }, // matrix
      { name: 'C', dtype: 'float32', shape: [2, 1] },    // column vector
    ],
  },
  {
    label: 'mul_3d_channel',
    originalPath: 'examples/onnx/mul_3d_channel.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [2, 3, 4] },   // [C,H,W]
      { name: 'W', dtype: 'float32', shape: [1, 3, 1] },   // per-channel weight
    ],
  },
  {
    label: 'chain_broadcast',
    originalPath: 'examples/onnx/chain_broadcast.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'b_row', dtype: 'float32', shape: [3] },
      { name: 'c_col', dtype: 'float32', shape: [2, 1] },
      { name: 's_sub', dtype: 'float32', shape: [] },
      { name: 's_div', dtype: 'float32', shape: [] },
    ],
  },

  // ── transpose + broadcast (2D..5D)
  {
    label: 'transpose_broadcast_2d',
    originalPath: 'examples/onnx/transpose_broadcast_2d.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 3] },
      { name: 'Y', dtype: 'float32', shape: [3] },
    ],
  },
  {
    label: 'transpose_broadcast_3d',
    originalPath: 'examples/onnx/transpose_broadcast_3d.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X',   dtype: 'float32', shape: [2, 1, 3] },
      { name: 'Zin', dtype: 'float32', shape: [1, 3, 1] },
    ],
  },
  {
    label: 'transpose_broadcast_4d',
    originalPath: 'examples/onnx/transpose_broadcast_4d.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [2, 1, 3, 1] },
      { name: 'B', dtype: 'float32', shape: [3, 1, 1, 1] },
    ],
  },
  {
    label: 'transpose_broadcast_5d',
    originalPath: 'examples/onnx/transpose_broadcast_5d.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 2, 1, 3, 1] },
      { name: 'C', dtype: 'float32', shape: [1, 3, 1, 1, 1] },
    ],
  },

  /*
  // ── matmul variants
  {
    label: 'matmul_rect',
    originalPath: 'examples/onnx/matmul_rect_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'B', dtype: 'float32', shape: [3, 4] },
    ],
  },
  {
    label: 'matmul_bias',
    originalPath: 'examples/onnx/matmul_bias_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'B', dtype: 'float32', shape: [3, 4] },
      { name: 'C', dtype: 'float32', shape: [4] },  // bias
    ],
  },
  {
    label: 'matmul_chain_right',
    originalPath: 'examples/onnx/matmul_chain_right_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'B', dtype: 'float32', shape: [3, 4] },
      { name: 'C', dtype: 'float32', shape: [4, 5] },
    ],
  },
  {
    label: 'matmul_mv_v1',
    originalPath: 'examples/onnx/matmul_mv_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] },
      { name: 'v', dtype: 'float32', shape: [3] },
    ],
  },
  {
    label: 'matmul_vm_v2',
    originalPath: 'examples/onnx/matmul_vm_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'v', dtype: 'float32', shape: [3] },
      { name: 'B', dtype: 'float32', shape: [3, 2] },
    ],
  },
  */

  // ── slice/pad/clip
  /*
  {
    label: 'slice_decomposition',
    originalPath: 'examples/onnx/slice.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 2, 5, 6] },
    ],
  },
  {
    label: 'pad_decomposition',
    originalPath: 'examples/onnx/pad_normal.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 2, 3, 4] },
    ],
  },
  {
    label: 'clip_scalar',
    originalPath: 'examples/onnx/clip_scalar.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X',   dtype: 'float32', shape: [2, 3] },
      { name: 'Min', dtype: 'float32', shape: [] },
      { name: 'Max', dtype: 'float32', shape: [] },
    ],
  },
  

  // ── conv
  /*
  {
    label: 'conv_normal',
    originalPath: 'examples/onnx/conv_normal.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 1, 4, 4] }, // 16 elems
      { name: 'W', dtype: 'float32', shape: [1, 1, 3, 3] }, // 9 elems
      { name: 'B', dtype: 'float32', shape: [1] },
    ],
  },
  /*
  {
    label: 'conv_simple',
    originalPath: 'examples/onnx/conv_simple.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 1, 3, 3] },
      { name: 'W', dtype: 'float32', shape: [1, 1, 1, 1] },
      { name: 'B', dtype: 'float32', shape: [1] },
    ],
  },
  */

  // ── gemm/concat/dequantize/avgpool
  /*
  {
    label: 'gemm_standard',
    originalPath: 'examples/onnx/gemm_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'A', dtype: 'float32', shape: [2, 3] }, // 6
      { name: 'B', dtype: 'float32', shape: [3, 4] }, // 12
      { name: 'C', dtype: 'float32', shape: [2, 4] }, // 8
    ],
  },
  {
    label: 'concat_standard',
    originalPath: 'examples/onnx/concat_standard.onnx',
    tol: 1e-5,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X0', dtype: 'float32', shape: [2, 3] }, // 6
      { name: 'X1', dtype: 'float32', shape: [2, 4] }, // 8
      { name: 'X2', dtype: 'float32', shape: [2, 2] }, // 4  => total 18 elems
    ],
  },
  {
    label: 'dequantize_standard',
    originalPath: 'examples/onnx/dequantize_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'uint8',   shape: [2, 3, 4] },   // 24 elems total
      { name: 'S', dtype: 'float32', shape: [3] },      // per-channel scales
      { name: 'Z', dtype: 'uint8',   shape: [3] },      // per-channel zero-points
    ],
  },
  */
 /*
  {
    label: 'averagepool_standard',
    originalPath: 'examples/onnx/avgpool_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [
      { name: 'X', dtype: 'float32', shape: [1, 2, 5, 6] },
    ],
  },
  */
  
  
  // ===== Reduce ops (JSON) =====
  {
    label: 'reducesum_standard',
    originalPath: 'examples/onnx/reducesum_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },
  {
    label: 'reducemax_standard',
    originalPath: 'examples/onnx/reducemax_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },

  // ----- More Reduce ops (JSON) -----
  {
    label: 'reducemin_standard',
    originalPath: 'examples/onnx/reducemin_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },
  {
    label: 'reduceprod_standard',
    originalPath: 'examples/onnx/reduceprod_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4] }], // general random
  },
  {
    label: 'reducemean_standard',
    originalPath: 'examples/onnx/reducemean_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },
  {
    label: 'reducesumsquare_standard',
    originalPath: 'examples/onnx/reducesumsquare_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },
  {
    label: 'reducel1_standard',
    originalPath: 'examples/onnx/reducel1_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },
  {
    label: 'reducel2_standard',
    originalPath: 'examples/onnx/reducel2_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'negmix' }],
  },
  {
    // sum(x) must be positive for log(sum(x)): use ones for a stable positive sum
    label: 'reducelogsum_standard',
    originalPath: 'examples/onnx/reducelogsum_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'ones' }],
  },
  {
    // log(sum(exp(x))) works for any x; 'ones' is fine and avoids extreme magnitudes
    label: 'reducelogsumexp_standard',
    originalPath: 'examples/onnx/reducelogsumexp_standard.onnx',
    tol: 1e-6,
    cliArgs: jsonFullArgs,
    specs: [{ name: 'X', dtype: 'float32', shape: [2, 3, 4], gen: 'ones' }],
  },
];

/* ============================== RUN ================================== */

async function runAllUnified() {
  for (const t of TESTS) {
    const cli = typeof t.cliArgs === 'function' ? t.cliArgs(t.originalPath) : t.cliArgs;
    const feeds = buildFeeds(t.specs);
    await testReconversion({
      label: t.label,
      originalPath: t.originalPath,
      feeds,
      tol: t.tol,
      exact: t.exact,
      cliArgs: cli,
    });
  }
}

await runAllUnified();

// TOTAL
process.on('beforeExit', () => {
  console.log(`\n=== TOTAL ===\nPassed: ${RUN_TOTAL.passed}\nFailed: ${RUN_TOTAL.failed}`);

  const printList = (title: string, items: Array<{test: string; path: string}>) => {
    if (!items.length) return;
    console.log(`\n${title} (${items.length})`);
    for (const { test, path } of items) {
      console.log(`  • ${test}  —  ${path}`);
    }
  };

  printList('❌ Reconverted models that FAILED to run', FAILURES.error);
  printList('⚠️ Reconverted models with NON-EQUIVALENT outputs', FAILURES.warn);
});

