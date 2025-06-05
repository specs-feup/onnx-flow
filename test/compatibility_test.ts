import { InferenceSession, Tensor } from 'onnxruntime-web';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';


function printInputs(label: string, inputs: Record<string, Tensor>) {
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

async function runVectorAddEquivalenceTest() {
  const shape = [4];

  // Generate shared random inputs
  const A = Float32Array.from({ length: 4 }, () => Math.random() * 10);
  const B = Float32Array.from({ length: 4 }, () => Math.random() * 10);

  const feeds = {
    A: new Tensor('float32', A, shape),
    B: new Tensor('float32', B, shape),
    trip_count: new Tensor('int64', [BigInt(4)], []),
    cond: new Tensor('bool', [true], []),
  };

  console.log('\n=== Running vector add model comparison ===');
  printInputs('Both Models', feeds);

  try {
    // Load and run standard model
    const stdSession = await InferenceSession.create('examples/onnx/vector_add_standard.onnx');
    const stdOutput = await stdSession.run({ A: feeds.A, B: feeds.B });
    const stdKey = Object.keys(stdOutput)[0];
    const stdResult = Array.from(stdOutput[stdKey].data as Float32Array);

    console.log(`standard_vector_add.onnx → Output (${stdKey}):`, stdResult);

    // Load and run scalar-loop model
    const loopSession = await InferenceSession.create('examples/onnx/vector_add_decomposed.onnx');
    const loopOutput = await loopSession.run({
      A: feeds.A,
      B: feeds.B,
      trip_count: feeds.trip_count,
      cond: feeds.cond,
    });
    const loopKey = Object.keys(loopOutput)[0];
    const loopResult = Array.from(loopOutput[loopKey].data as Float32Array);

    console.log(`scalar_loop_vector_add.onnx → Output (${loopKey}):`, loopResult);

    // Compare
    const tolerance = 1e-5;
    const equivalent =
      stdResult.length === loopResult.length &&
      stdResult.every((v, i) => Math.abs(v - loopResult[i]) < tolerance);

    console.log('✅ Vector outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('vector add model comparison', err);
  }
}

async function runAddChainEquivalenceTest() {
  const shape = [4];

  // Generate shared random vectors
  const A = Float32Array.from({ length: 4 }, () => Math.random() * 10);
  const B = Float32Array.from({ length: 4 }, () => Math.random() * 10);
  const C = Float32Array.from({ length: 4 }, () => Math.random() * 10);
  const D = Float32Array.from({ length: 4 }, () => Math.random() * 10);

  const feeds = {
    A: new Tensor('float32', A, shape),
    B: new Tensor('float32', B, shape),
    C: new Tensor('float32', C, shape),
    D: new Tensor('float32', D, shape),
    trip_count: new Tensor('int64', [BigInt(4)], []),
    cond: new Tensor('bool', [true], []),
  };

  console.log('\n=== Running Add(Add(A, B), Add(C, D)) model comparison ===');
  printInputs('AddChain Models', feeds);

  try {
    // Run standard add chain model
    const stdSession = await InferenceSession.create('examples/onnx/add_chain_standard.onnx');
    const stdOutput = await stdSession.run({
      A: feeds.A,
      B: feeds.B,
      C: feeds.C,
      D: feeds.D,
    });
    const stdKey = Object.keys(stdOutput)[0];
    const stdResult = Array.from(stdOutput[stdKey].data as Float32Array);

    console.log(`add_chain_standard.onnx → Output (${stdKey}):`, stdResult);

    // Run decomposed scalar loop version
    const loopSession = await InferenceSession.create('examples/onnx/add_chain_decomposed.onnx');
    const loopOutput = await loopSession.run(feeds);
    const loopKey = Object.keys(loopOutput)[0];
    const loopResult = Array.from(loopOutput[loopKey].data as Float32Array);

    console.log(`add_chain_decomposed.onnx → Output (${loopKey}):`, loopResult);

    // Check equivalence
    const tolerance = 1e-5;
    const equivalent =
      stdResult.length === loopResult.length &&
      stdResult.every((v, i) => Math.abs(v - loopResult[i]) < tolerance);

    console.log('✅ Add-chain model outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('add-chain model comparison', err);
  }
}

async function runMatmulEquivalenceTest() {
  const shape = [2, 2];

  const A = Float32Array.from({ length: 4 }, () => Math.random() * 10);
  const B = Float32Array.from({ length: 4 }, () => Math.random() * 10);

  const feeds = {
    A: new Tensor('float32', A, shape),
    B: new Tensor('float32', B, shape),
    trip_count: new Tensor('int64', [BigInt(4)], []),
    cond: new Tensor('bool', [true], []),
  };

  console.log('\n=== Running MatMul model comparison ===');
  printInputs('MatMul Models', feeds);

  try {
    const stdSession = await InferenceSession.create('examples/onnx/matmul_standard.onnx');
    const stdOutput = await stdSession.run({ A: feeds.A, B: feeds.B });
    const stdKey = Object.keys(stdOutput)[0];
    const stdResult = Array.from(stdOutput[stdKey].data as Float32Array);

    console.log(`matmul_standard.onnx → Output (${stdKey}):`, stdResult);

    const loopSession = await InferenceSession.create('examples/onnx/matmul_decomposed.onnx');
    const loopOutput = await loopSession.run({
      A: feeds.A,
      B: feeds.B,
      trip_count: feeds.trip_count,
      cond: feeds.cond,
    });
    const loopKey = Object.keys(loopOutput)[0];
    const loopResult = Array.from(loopOutput[loopKey].data as Float32Array);

    console.log(`matmul_decomposed.onnx → Output (${loopKey}):`, loopResult);

    const tolerance = 1e-5;
    const equivalent =
      stdResult.length === loopResult.length &&
      stdResult.every((v, i) => Math.abs(v - loopResult[i]) < tolerance);

    console.log('✅ MatMul outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('MatMul model comparison', err);
  }
}

function getReconvertedPath(originalPath: string): string {
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  return `${base}_reconverted.onnx`;
}

async function runAddAddAddReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/AddAddAdd.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json --noLowLevel -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const A = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const B = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const C = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const X = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const shape = [2, 2];

    const feeds = {
      A: new Tensor('float32', A, shape),
      B: new Tensor('float32', B, shape),
      C: new Tensor('float32', C, shape),
      X: new Tensor('float32', X, shape),
    };

    console.log('\n=== Comparing original and reconverted AddAddAdd model ===');
    printInputs('AddAddAdd', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOutput = await originalSession.run(feeds);
    const originalKey = Object.keys(originalOutput)[0];
    const originalResult = Array.from(originalOutput[originalKey].data as Float32Array);
    console.log(`${path.basename(originalPath)} → Output (${originalKey}):`, originalResult);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOutput = await reconvertedSession.run(feeds);
    const reconvertedKey = Object.keys(reconvertedOutput)[0];
    const reconvertedResult = Array.from(reconvertedOutput[reconvertedKey].data as Float32Array);
    console.log(`${path.basename(reconvertedPath)} → Output (${reconvertedKey}):`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent =
      originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);

    console.log('✅ AddAddAdd outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('AddAddAdd reconversion test', err);
  }
}

async function runAddChainDecomposedReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/add_chain_decomposed.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted add_chain_decomposed model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json --noLowLevel -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const shape = [4];
    const feeds = {
      A: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), shape),
      B: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), shape),
      C: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), shape),
      D: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), shape),
      trip_count: new Tensor('int64', [BigInt(4)], []),
      cond: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing add_chain_decomposed and its reconverted version ===');
    printInputs('add_chain_decomposed', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOutput = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOutput)[0].data as Float32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOutput = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOutput)[0].data as Float32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent = originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);
    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('add_chain_decomposed reconversion test', err);
  }
}


async function runMatmulDecomposedReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_decomposed.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmul_decomposed model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json --noLowLevel -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const shape = [2, 2];
    const feeds = {
      A: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), shape),
      B: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), shape),
      trip_count: new Tensor('int64', [BigInt(4)], []),
      cond: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing matmul_decomposed and its reconverted version ===');
    printInputs('matmul_decomposed', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOutput = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOutput)[0].data as Float32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOutput = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOutput)[0].data as Float32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent = originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);
    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('matmul_decomposed reconversion test', err);
  }
}



// Tests to run
await runVectorAddEquivalenceTest();
await runAddChainEquivalenceTest();
await runMatmulEquivalenceTest();

await runAddAddAddReconversionEquivalenceTest();

await runAddChainDecomposedReconversionEquivalenceTest();
await runMatmulDecomposedReconversionEquivalenceTest();


