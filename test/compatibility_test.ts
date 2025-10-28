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

async function runMatmulAddEquivalenceTest() {
  const X = Int32Array.from({ length: 3 }, () => Math.floor(Math.random() * 10));
  const A = Int32Array.from({ length: 3 }, () => Math.floor(Math.random() * 10));
  const B = Int32Array.from({ length: 9 }, () => Math.floor(Math.random() * 10));

  const feeds = {
    X: new Tensor('int32', X, [3, 1]),
    A: new Tensor('int32', A, [1, 3]),
    B: new Tensor('int32', B, [3, 3]),
    trip_count: new Tensor('int64', [BigInt(9)], []),
    cond: new Tensor('bool', [true], []),
  };

  console.log('\n=== Running MatMul+Add model comparison ===');
  printInputs('MatMulAdd Models', feeds);

  try {
    const stdSession = await InferenceSession.create('examples/onnx/matmul_add_standard.onnx');
    const stdOut = await stdSession.run({ X: feeds.X, A: feeds.A, B: feeds.B });
    const stdResult = Array.from(Object.values(stdOut)[0].data as Int32Array);

    console.log(`matmul_add_standard.onnx → Output:`, stdResult);

    const decSession = await InferenceSession.create('examples/onnx/matmul_add_decomposed.onnx');
    const decOut = await decSession.run(feeds);
    const decResult = Array.from(Object.values(decOut)[0].data as Int32Array);

    console.log(`matmul_add_decomposed.onnx → Output:`, decResult);

    const equivalent = stdResult.length === decResult.length &&
      stdResult.every((v, i) => v === decResult[i]);

    console.log('✅ MatMul+Add outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('MatMul+Add comparison', err);
  }
}


function getReconvertedPath(originalPath: string): string {
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  return `${base}_reconverted.onnx`;
}

async function runVectorAddStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/vector_add_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted vector_add_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json --noLowLevel -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const shape = [4];
    const A = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const B = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const feeds = {
      A: new Tensor('float32', A, shape),
      B: new Tensor('float32', B, shape),
    };

    console.log('\n=== Comparing vector_add_standard and its reconverted version ===');
    printInputs('vector_add_standard', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOut)[0].data as Float32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Float32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent =
      originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('vector_add_standard reconversion test', err);
  }
}


async function runAddChainStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/add_chain_standard.onnx';
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

async function runMatmulStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmul_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json --noLowLevel -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const shape = [2, 2];
    const A = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const B = Float32Array.from({ length: 4 }, () => Math.random() * 10);
    const feeds = {
      A: new Tensor('float32', A, shape),
      B: new Tensor('float32', B, shape),
    };

    console.log('\n=== Comparing matmul_standard and its reconverted version ===');
    printInputs('matmul_standard', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOut)[0].data as Float32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Float32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent =
      originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('matmul_standard reconversion test', err);
  }
}

async function runMatmulAddStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_add_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmul_add_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json --noLowLevel -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const feeds = {
      X: new Tensor('int32', Int32Array.from({ length: 3 }, () => Math.floor(Math.random() * 10)), [3, 1]),
      A: new Tensor('int32', Int32Array.from({ length: 3 }, () => Math.floor(Math.random() * 10)), [1, 3]),
      B: new Tensor('int32', Int32Array.from({ length: 9 }, () => Math.floor(Math.random() * 10)), [3, 3]),
    };

    console.log('\n=== Comparing matmul_add_standard and its reconverted version ===');
    printInputs('matmul_add_standard', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOut)[0].data as Int32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Int32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const equivalent = originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => v === reconvertedResult[i]);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('matmul_add_standard reconversion test', err);
  }
}

async function runVectorAddDecomposedReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/vector_add_decomposed.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted vector_add_decomposed model ===');
    execSync(`node ./out/src/index.js ${originalPath} --noLowLevel --format dot  -vz 0 -v 0 -o ${base}.dot`, {
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
      trip_count: new Tensor('int64', [BigInt(4)], []),
      cond: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing vector_add_decomposed and its reconverted version ===');
    printInputs('vector_add_decomposed', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOut)[0].data as Float32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Float32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent =
      originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('vector_add_decomposed reconversion test', err);
  }
}

async function runAddChainDecomposedReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/add_chain_decomposed.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted add_chain_decomposed model ===');
    execSync(`node ./out/src/index.js ${originalPath} --noLowLevel --format dot  -vz 0 -v 0 -o ${base}.dot`, {
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
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmul_decomposed model ===');
    execSync(`node ./out/src/index.js ${originalPath} --noLowLevel --format dot  -vz 0 -v 0 -o ${base}.dot`, {
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

async function runMatmulAddDecomposedReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_add_decomposed.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmul_add_decomposed model ===');
    execSync(`node ./out/src/index.js ${originalPath} --noLowLevel --format dot  -vz 0 -v 0 -o ${base}.dot`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const feeds = {
      X: new Tensor('int32', Int32Array.from({ length: 3 }, () => Math.floor(Math.random() * 10)), [3, 1]),
      A: new Tensor('int32', Int32Array.from({ length: 3 }, () => Math.floor(Math.random() * 10)), [1, 3]),
      B: new Tensor('int32', Int32Array.from({ length: 9 }, () => Math.floor(Math.random() * 10)), [3, 3]),
      trip_count: new Tensor('int64', [BigInt(9)], []),
      cond: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing matmul_add_decomposed and its reconverted version ===');
    printInputs('matmul_add_decomposed', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run(feeds);
    const originalResult = Array.from(Object.values(originalOut)[0].data as Int32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Int32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const equivalent = originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => v === reconvertedResult[i]);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('matmul_add_decomposed reconversion test', err);
  }
}

async function runVectorAddCompleteEquivalenceTest() {
  const originalPath = 'examples/onnx/vectoradd_test.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted vectoradd_test model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format dot  -vz 0 -v 0 -o ${base}.dot`, {
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
      //trip_count_0: new Tensor('int64', [BigInt(4)], []),
      //cond_0: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing vectoradd_test and its decomposed reconverted version ===');
    printInputs('vectoradd_test', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run({
      A: feeds.A,
      B: feeds.B,
    });
    const originalResult = Array.from(Object.values(originalOut)[0].data as Float32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Float32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const tolerance = 1e-5;
    const equivalent =
      originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => Math.abs(v - reconvertedResult[i]) < tolerance);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('vectoradd_test reconversion test', err);
  }
}

async function runAddChainCompleteEquivalenceTest() {
  const originalPath = 'examples/onnx/addchain_test.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted addchain_test model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format dot  -vz 0 -v 0 -o ${base}.dot`, {
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
      //trip_count_1: new Tensor('int64', [BigInt(4)], []),
      //cond_1: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing addchain_test and its decomposed reconverted version ===');
    printInputs('addchain_test', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOutput = await originalSession.run(({
      A: feeds.A,
      B: feeds.B,
      C: feeds.C,
      D: feeds.D,
    }));
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
    logErrorDetails('addchain_test reconversion test', err);
  }
}


async function runMatmulCompleteEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_test.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmul_test model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format dot  -vz 0 -v 0 -o ${base}.dot`, {
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
      //trip_count_0: new Tensor('int64', [BigInt(4)], []),
      //cond_0: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing matmul_test and its decomposed reconverted version ===');
    printInputs('matmul_test', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOutput = await originalSession.run({
      A: feeds.A,
      B: feeds.B,
    });
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
    logErrorDetails('matmul_test reconversion test', err);
  }
}

async function runMatmulAddCompleteEquivalenceTest() {
  const originalPath = 'examples/onnx/matmuladd_test.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted matmuladd_test model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format dot  -vz 0 -v 0 -o ${base}.dot`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const feeds = {
      X: new Tensor('int32', Int32Array.from({ length: 9 }, () => Math.floor(Math.random() * 10)), [3, 3]),
      A: new Tensor('int32', Int32Array.from({ length: 9 }, () => Math.floor(Math.random() * 10)), [3, 3]),
      B: new Tensor('int32', Int32Array.from({ length: 9 }, () => Math.floor(Math.random() * 10)), [3, 3]),
      //trip_count_0: new Tensor('int64', [BigInt(9)], []),
      //cond_0: new Tensor('bool', [true], []),
    };

    console.log('\n=== Comparing matmuladd_test and its decomposed reconverted version ===');
    printInputs('matmuladd_test', feeds);

    const originalSession = await InferenceSession.create(originalPath);
    const originalOut = await originalSession.run({
      X: feeds.X,
      A: feeds.A,
      B: feeds.B,
    });
    const originalResult = Array.from(Object.values(originalOut)[0].data as Int32Array);

    const reconvertedSession = await InferenceSession.create(reconvertedPath);
    const reconvertedOut = await reconvertedSession.run(feeds);
    const reconvertedResult = Array.from(Object.values(reconvertedOut)[0].data as Int32Array);

    console.log(`→ original:`, originalResult);
    console.log(`→ reconverted:`, reconvertedResult);

    const equivalent = originalResult.length === reconvertedResult.length &&
      originalResult.every((v, i) => v === reconvertedResult[i]);

    console.log('✅ Outputs equivalent:', equivalent);
  } catch (err) {
    logErrorDetails('matmuladd_test reconversion test', err);
  }
}

async function runRangeStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/range_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted range_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`); return;
    }

    // Choose a clean set so length is known: start=0, limit=5, delta=1 => len=5
    const start = new Tensor('float32', new Float32Array([0]), []);
    const limit = new Tensor('float32', new Float32Array([5]), []);
    const delta = new Tensor('float32', new Float32Array([1]), []);
    const feeds = { start, limit, delta };

    console.log('\n=== Comparing range_standard and its reconverted version ===');
    printInputs('range_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('range_standard reconversion test', err);
  }
}

async function runRangeAddStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/range_add_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted range_add_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    // start=1, limit=6, delta=1.5 => ceil((6-1)/1.5)=4 elements: [1, 2.5, 4, 5.5]
    const start = 1, limit = 6, delta = 1.5;
    const L = Math.max(0, Math.ceil((limit - start) / delta));

    const feeds = {
      start: new Tensor('float32', new Float32Array([start]), []),
      limit: new Tensor('float32', new Float32Array([limit]), []),
      delta: new Tensor('float32', new Float32Array([delta]), []),
      V:     new Tensor('float32', Float32Array.from({ length: L }, () => Math.random() * 10), [L]),
    };

    console.log('\n=== Comparing range_add_standard and reconverted ===');
    printInputs('range_add_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('range_add_standard reconversion test', err);
  }
}

async function runTransposeStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/transpose_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted transpose_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const X = Float32Array.from({ length: 6 }, () => Math.random() * 10);
    const feeds = { X: new Tensor('float32', X, [2, 3]) };

    console.log('\n=== Comparing transpose_standard and reconverted ===');
    printInputs('transpose_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);
    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('transpose_standard reconversion test', err);
  }
}

async function runTransposeAddStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/transpose_add_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted transpose_add_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const X = Float32Array.from({ length: 6 }, () => Math.random() * 10); // [2,3]
    const Y = Float32Array.from({ length: 6 }, () => Math.random() * 10); // [3,2]
    const feeds = {
      X: new Tensor('float32', X, [2, 3]),
      Y: new Tensor('float32', Y, [3, 2]),
    };

    console.log('\n=== Comparing transpose_add_standard and reconverted ===');
    printInputs('transpose_add_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);
    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('transpose_add_standard reconversion test', err);
  }
}

async function runMatmulTransposeStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_transpose_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted matmul_transpose_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const A = Float32Array.from({ length: 6 }, () => Math.random() * 10); // [2,3]
    const B = Float32Array.from({ length: 12 }, () => Math.random() * 10); // [3,4]
    const feeds = {
      A: new Tensor('float32', A, [2, 3]),
      B: new Tensor('float32', B, [3, 4]),
    };

    console.log('\n=== Comparing matmul_transpose_standard and reconverted ===');
    printInputs('matmul_transpose_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);
    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('matmul_transpose_standard reconversion test', err);
  }
}

async function runReluStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/relu_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted relu_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    // Mix negatives/positives to exercise thresholding
    const X = Float32Array.from([-3.2, -0.01, 0, 0.25, 5.5, -7.3]);
    const feeds = { X: new Tensor('float32', X, [6]) };

    console.log('\n=== Comparing relu_standard and reconverted ===');
    printInputs('relu_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o); console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('relu_standard reconversion test', err);
  }
}

async function runSigmoidStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/sigmoid_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted sigmoid_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const X = Float32Array.from([-6, -1, 0, 1, 2, 6]);
    const feeds = { X: new Tensor('float32', X, [6]) };

    console.log('\n=== Comparing sigmoid_standard and reconverted ===');
    printInputs('sigmoid_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o); console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('sigmoid_standard reconversion test', err);
  }
}

async function runTanhStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/tanh_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted tanh_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const X = Float32Array.from([-3, -0.5, 0, 0.5, 1.25, 3]);
    const feeds = { X: new Tensor('float32', X, [6]) };

    console.log('\n=== Comparing tanh_standard and reconverted ===');
    printInputs('tanh_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o); console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('tanh_standard reconversion test', err);
  }
}

async function runExpStandardReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/exp_standard.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted exp_standard model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const X = Float32Array.from([-2, -1, 0, 0.5, 1, 2]); // keep values tame to avoid overflow in fp32
    const feeds = { X: new Tensor('float32', X, [6]) };

    console.log('\n=== Comparing exp_standard and reconverted ===');
    printInputs('exp_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o); console.log('→ reconverted:', r);

    const tol = 1e-5; // slightly looser due to potential tiny numeric drift
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('exp_standard reconversion test', err);
  }
}

async function runUnaryBinaryComboReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/unary_binary_combo.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    console.log('\n=== Running CLI to generate reconverted unary_binary_combo model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Not found: ${reconvertedPath}`); return;
    }

    // Make inputs: mix negatives/positives; keep S positive to avoid div-by-zero
    const X = Float32Array.from([-2.0, -0.5, 0.0, 0.25, 1.5, 3.0]);
    const A = Float32Array.from([ 0.1, -1.5, 2.0, -0.75, 0.5,  1.0]);
    const B = Float32Array.from([ 2.0,  0.5, 1.5,  3.25, 0.1, -2.0]);
    const Y = Float32Array.from([ 1.0,  0.2, 0.0,  0.1,  2.0,  3.0]);
    const S = Float32Array.from([ 4.0,  0.2, 2.5,  0.1,  -1.0,  -3.1]);

    const feeds = {
      X: new Tensor('float32', X, [6]),
      A: new Tensor('float32', A, [6]),
      B: new Tensor('float32', B, [6]),
      Y: new Tensor('float32', Y, [6]),
      S: new Tensor('float32', S, [6]),
    };

    console.log('\n=== Comparing unary_binary_combo and reconverted ===');
    printInputs('unary_binary_combo', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('unary_binary_combo reconversion test', err);
  }
}

export async function runClipScalarReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/clip_scalar.onnx';
  const decomposedPath = 'examples/onnx/clip_scalar_decomposed.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    if (!fs.existsSync(originalPath) || !fs.existsSync(decomposedPath)) {
      throw new Error('Clip scalar models not found. Run: python "python scripts/make_clip_models.py"');
    }

    const X = new Float32Array([-1.5, 0.25, 3, 0, 0.75, 10]); // [2,3]
    const Min = new Float32Array([0.0]); // []
    const Max = new Float32Array([2.0]); // []

    const feeds = {
      X: new Tensor('float32', X, [2, 3]),
      Min: new Tensor('float32', Min, []),
      Max: new Tensor('float32', Max, []),
    };

    printInputs('clip_scalar', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const o = Array.from(Object.values(orig)[0].data as Float32Array);

    const tol = 1e-6;

    console.log('\n=== Running CLI to generate reconverted clip_scalar model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Not found: ${reconvertedPath}`); return;
    }

    console.log('\n=== Comparing clip_scalar and reconverted ===');
    const rec = await (await InferenceSession.create(reconvertedPath)).run(feeds);
    const r = Array.from(Object.values(rec)[0].data as Float32Array);
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) <= tol);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);
    console.log('✅ clip_scalar vs reconverted equivalent:', eq);

    if (!eq) throw new Error('Clip scalar equivalence failed.');
  } catch (err) {
    logErrorDetails('clip scalar reconversion test', err);
  }
}

async function runAddScalarVectorBroadcastTest() {
  const originalPath = 'examples/onnx/add_scalar_vector.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted add_scalar_vector model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const X = Float32Array.from([-2.0, -0.5, 0.0, 0.25, 1.5, 3.0]); // [6]
    const S = new Float32Array([1.25]);                              // []
    const feeds = { X: new Tensor('float32', X, [6]),
                    S: new Tensor('float32', S, []) };

    console.log('\n=== Comparing add_scalar_vector and reconverted ===');
    printInputs('add_scalar_vector', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    console.log('✅ Outputs equivalent:', o.length === r.length && o.every((v,i)=>Math.abs(v-r[i])<tol));
  } catch (err) {
    logErrorDetails('add_scalar_vector reconversion test', err);
  }
}

async function runAddRowVectorToMatrixBroadcastTest() {
  const originalPath = 'examples/onnx/add_row_vector_to_matrix.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted add_row_vector_to_matrix model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const A = Float32Array.from([ 1, 2, 3,  4, 5, 6 ]); // [2,3]
    const B = Float32Array.from([ 10, -1, 0.5 ]);       // [3]
    const feeds = {
      A: new Tensor('float32', A, [2,3]),
      B: new Tensor('float32', B, [3]),
    };

    console.log('\n=== Comparing add_row_vector_to_matrix and reconverted ===');
    printInputs('add_row_vector_to_matrix', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    console.log('✅ Outputs equivalent:', o.length === r.length && o.every((v,i)=>Math.abs(v-r[i])<tol));
  } catch (err) {
    logErrorDetails('add_row_vector_to_matrix reconversion test', err);
  }
}

async function runAddColVectorToMatrixBroadcastTest() {
  const originalPath = 'examples/onnx/add_col_vector_to_matrix.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted add_col_vector_to_matrix model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const A = Float32Array.from([ 1, 2, 3,  4, 5, 6 ]); // [2,3]
    const C = Float32Array.from([ 100,  200 ]);         // [2,1]
    const feeds = {
      A: new Tensor('float32', A, [2,3]),
      C: new Tensor('float32', C, [2,1]),
    };

    console.log('\n=== Comparing add_col_vector_to_matrix and reconverted ===');
    printInputs('add_col_vector_to_matrix', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    console.log('✅ Outputs equivalent:', o.length === r.length && o.every((v,i)=>Math.abs(v-r[i])<tol));
  } catch (err) {
    logErrorDetails('add_col_vector_to_matrix reconversion test', err);
  }
}

async function runMul3DChannelBroadcastTest() {
  const originalPath = 'examples/onnx/mul_3d_channel.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted mul_3d_channel model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    // X [2,3,4]
    const X = Float32Array.from([
      // batch 0
      1,2,3,4,    5,6,7,8,    9,10,11,12,
      // batch 1
      2,4,6,8,    1,3,5,7,    0,-1,-2,-3,
    ]);
    // W [1,3,1] (per-channel scale)
    const W = Float32Array.from([0.5, 2.0, -1.0]);

    const feeds = {
      X: new Tensor('float32', X, [2,3,4]),
      W: new Tensor('float32', W, [1,3,1]),
    };

    console.log('\n=== Comparing mul_3d_channel and reconverted ===');
    printInputs('mul_3d_channel', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o.slice(0,12), '...');
    console.log('→ reconverted:', r.slice(0,12), '...');

    const tol = 1e-6;
    console.log('✅ Outputs equivalent:', o.length === r.length && o.every((v,i)=>Math.abs(v-r[i])<tol));
  } catch (err) {
    logErrorDetails('mul_3d_channel reconversion test', err);
  }
}

async function runChainBroadcastTest() {
  const originalPath = 'examples/onnx/chain_broadcast.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  try {
    console.log('\n=== Running CLI to generate reconverted chain_broadcast model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });
    if (!fs.existsSync(reconvertedPath)) { console.error(`❌ Not found: ${reconvertedPath}`); return; }

    const A     = Float32Array.from([ 1, 2, 3,  4, 5, 6 ]);  // [2,3]
    const b_row = Float32Array.from([ 0.1, -1.0, 2.5 ]);     // [3]
    const c_col = Float32Array.from([ 2.0, 0.5 ]);           // [2,1]
    const s_sub = new Float32Array([0.75]);                  // []
    const s_div = new Float32Array([1.25]);                  // [] > 0

    const feeds = {
      A:     new Tensor('float32', A,     [2,3]),
      b_row: new Tensor('float32', b_row, [3]),
      s_sub: new Tensor('float32', s_sub, []),
      c_col: new Tensor('float32', c_col, [2,1]),
      s_div: new Tensor('float32', s_div, []),
    };

    console.log('\n=== Comparing chain_broadcast and reconverted ===');
    printInputs('chain_broadcast', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);
    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    console.log('✅ Outputs equivalent:', o.length === r.length && o.every((v,i)=>Math.abs(v-r[i])<tol));
  } catch (err) {
    logErrorDetails('chain_broadcast reconversion test', err);
  }
}

async function runTransposeBroadcast2DReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/transpose_broadcast_2d.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = `${base}_reconverted.onnx`;

  try {
    console.log('\n=== Running CLI to generate reconverted transpose_broadcast_2d model ===');
    // Generate reconverted model (JSON → ONNX) using your CLI. No --noLowLevel.
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // Inputs:
    // X: [1,3]; Y: [3]. The ONNX model itself Unsqueezes Y → [3,1] before Add.
    const X = new Float32Array([1, 2, 3]);
    const Y = new Float32Array([10, 20, 30]);
    const feeds = {
      X: new Tensor('float32', X, [1, 3]),
      Y: new Tensor('float32', Y, [3]),
    };

    console.log('\n=== Comparing transpose_broadcast_2d and reconverted ===');
    printInputs('transpose_broadcast_2d', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    if (!eq) {
      console.error('❌ Outputs differ beyond tolerance.');
    } else {
      console.log('✅ Outputs equivalent:', eq);
    }
  } catch (err) {
    logErrorDetails('transpose_broadcast_2d reconversion test', err);
  }
}


async function runTransposeBroadcast3DReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/transpose_broadcast_3d.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = `${base}_reconverted.onnx`;

  try {
    console.log('\n=== Running CLI to generate reconverted transpose_broadcast_3d model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // X[2,1,3], Zin[1,3,1]; Transpose(perm=[0,2,1]) → [2,3,1]; then Add with broadcast
    const X = Float32Array.from([0,1,2,  3,4,5]); // shaped to [2,1,3]
    const Zin = Float32Array.from([100,200,300]);  // shaped to [1,3,1]
    const feeds = {
      X:   new Tensor('float32', X,   [2, 1, 3]),
      Zin: new Tensor('float32', Zin, [1, 3, 1]),
    };

    console.log('\n=== Comparing transpose_broadcast_3d and reconverted ===');
    printInputs('transpose_broadcast_3d', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('transpose_broadcast_3d reconversion test', err);
  }
}

async function runTransposeBroadcast4DReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/transpose_broadcast_4d.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = `${base}_reconverted.onnx`;

  try {
    console.log('\n=== Running CLI to generate reconverted transpose_broadcast_4d model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // 4D test feeds
    const X = Float32Array.from([0,1,2, 3,4,5]); // [2,1,3,1]
    const B = Float32Array.from([100,200,300]);  // [3,1,1,1]

    const feeds = {
      X: new Tensor('float32', X, [2, 1, 3, 1]),
      B: new Tensor('float32', B, [3, 1, 1, 1]),
    };

    console.log('\n=== Comparing transpose_broadcast_4d and reconverted ===');
    printInputs('transpose_broadcast_4d', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    if (!eq) console.error('❌ Outputs differ beyond tolerance.');
    else console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('transpose_broadcast_4d reconversion test', err);
  }
}

async function runTransposeBroadcast5DReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/transpose_broadcast_5d.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = `${base}_reconverted.onnx`;

  try {
    console.log('\n=== Running CLI to generate reconverted transpose_broadcast_5d model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // X: [1,2,1,3,1] → 1*2*1*3*1 = 6 elements: 0..5
    // C: [1,3,1,1,1] with [100,200,300]
    const X = Float32Array.from([0,1,2, 3,4,5]);
    const C = Float32Array.from([100,200,300]);

    const feeds = {
      X: new Tensor('float32', X, [1, 2, 1, 3, 1]),
      C: new Tensor('float32', C, [1, 3, 1, 1, 1]),
    };

    console.log('\n=== Comparing transpose_broadcast_5d and reconverted ===');
    printInputs('transpose_broadcast_5d', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    if (!eq) console.error('❌ Outputs differ beyond tolerance.');
    else console.log('✅ Outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('transpose_broadcast_5d reconversion test', err);
  }
}

async function runMatmulRectReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_rect_2x3_3x4.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');
  try {
    console.log('\n=== Running CLI to generate reconverted matmul_rect_2x3_3x4 model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    const A = Float32Array.from({ length: 2*3 }, () => Math.random()*5);
    const B = Float32Array.from({ length: 3*4 }, () => Math.random()*5);
    const feeds = { A: new Tensor('float32', A, [2,3]), B: new Tensor('float32', B, [3,4]) };

    console.log('\n=== Comparing matmul_rect_2x3_3x4 and reconverted ===');
    printInputs('matmul_rect_2x3_3x4', feeds);

    const o = await (await InferenceSession.create(originalPath)).run(feeds);
    const r = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const O = Array.from(Object.values(o)[0].data as Float32Array);
    const R = Array.from(Object.values(r)[0].data as Float32Array);

    const tol = 1e-5;
    const eq = O.length===R.length && O.every((v,i)=>Math.abs(v-R[i])<tol);
    console.log('✅ Rectangular MatMul outputs equivalent:', eq);
  } catch (err) { logErrorDetails('matmul_rect_2x3_3x4 reconversion test', err); }
}

async function runMatmulBiasReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_bias_3x2_2x5.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');
  try {
    console.log('\n=== Running CLI to generate reconverted matmul_bias_3x2_2x5 model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    const A = Float32Array.from({ length: 3*2 }, () => Math.random()*5);
    const B = Float32Array.from({ length: 2*5 }, () => Math.random()*5);
    const Bias = Float32Array.from({ length: 1*5 }, () => Math.random()*2);
    const feeds = {
      A: new Tensor('float32', A, [3,2]),
      B: new Tensor('float32', B, [2,5]),
      Bias: new Tensor('float32', Bias, [1,5]),
    };

    console.log('\n=== Comparing matmul_bias_3x2_2x5 and reconverted ===');
    printInputs('matmul_bias_3x2_2x5', feeds);

    const o = await (await InferenceSession.create(originalPath)).run(feeds);
    const r = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const O = Array.from(Object.values(o)[0].data as Float32Array);
    const R = Array.from(Object.values(r)[0].data as Float32Array);

    const tol = 1e-5;
    const eq = O.length===R.length && O.every((v,i)=>Math.abs(v-R[i])<tol);
    console.log('✅ MatMul+Bias outputs equivalent:', eq);
  } catch (err) { logErrorDetails('matmul_bias_3x2_2x5 reconversion test', err); }
}

async function runMatmulChainRightReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_chain_right.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');
  try {
    console.log('\n=== Running CLI to generate reconverted matmul_chain_right model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    const A = Float32Array.from({ length: 2*3 }, () => Math.random()*5);
    const B = Float32Array.from({ length: 3*2 }, () => Math.random()*5);
    const C = Float32Array.from({ length: 2*2 }, () => Math.random()*5);
    const feeds = {
      A: new Tensor('float32', A, [2,3]),
      B: new Tensor('float32', B, [3,2]),
      C: new Tensor('float32', C, [2,2]),
    };

    console.log('\n=== Comparing matmul_chain_right and reconverted ===');
    printInputs('matmul_chain_right', feeds);

    const o = await (await InferenceSession.create(originalPath)).run(feeds);
    const r = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const O = Array.from(Object.values(o)[0].data as Float32Array);
    const R = Array.from(Object.values(r)[0].data as Float32Array);

    const tol = 1e-5;
    const eq = O.length===R.length && O.every((v,i)=>Math.abs(v-R[i])<tol);
    console.log('✅ (A·B)·C outputs equivalent:', eq);
  } catch (err) { logErrorDetails('matmul_chain_right reconversion test', err); }
}

async function runMatmulMVReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_mv_4x3_k3.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');
  try {
    console.log('\n=== Running CLI to generate reconverted matmul_mv_4x3_k3 model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    const A = Float32Array.from({ length: 4*3 }, () => Math.random()*5);
    const v = Float32Array.from({ length: 3 }, () => Math.random()*5);
    const feeds = { A: new Tensor('float32', A, [4,3]), v: new Tensor('float32', v, [3]) };

    console.log('\n=== Comparing matmul_mv_4x3_k3 and reconverted ===');
    printInputs('matmul_mv_4x3_k3', feeds);

    const o = await (await InferenceSession.create(originalPath)).run(feeds);
    const r = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const O = Array.from(Object.values(o)[0].data as Float32Array);
    const R = Array.from(Object.values(r)[0].data as Float32Array);

    const tol = 1e-5;
    const eq = O.length===R.length && O.every((v,i)=>Math.abs(v-R[i])<tol);
    console.log('✅ Matrix–Vector outputs equivalent:', eq);
  } catch (err) { logErrorDetails('matmul_mv_4x3_k3 reconversion test', err); }
}

async function runMatmulVMReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/matmul_vm_k3_3x5.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');
  try {
    console.log('\n=== Running CLI to generate reconverted matmul_vm_k3_3x5 model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    const v = Float32Array.from({ length: 3 }, () => Math.random()*5);
    const B = Float32Array.from({ length: 3*5 }, () => Math.random()*5);
    const feeds = { v: new Tensor('float32', v, [3]), B: new Tensor('float32', B, [3,5]) };

    console.log('\n=== Comparing matmul_vm_k3_3x5 and reconverted ===');
    printInputs('matmul_vm_k3_3x5', feeds);

    const o = await (await InferenceSession.create(originalPath)).run(feeds);
    const r = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const O = Array.from(Object.values(o)[0].data as Float32Array);
    const R = Array.from(Object.values(r)[0].data as Float32Array);

    const tol = 1e-5;
    const eq = O.length===R.length && O.every((v,i)=>Math.abs(v-R[i])<tol);
    console.log('✅ Vector–Matrix outputs equivalent:', eq);
  } catch (err) { logErrorDetails('matmul_vm_k3_3x5 reconversion test', err); }
}

// Slice: [1,2,5,6] -> Slice(axes=[2,3], starts=[1,2], ends=[5,6], steps=[2,2]) -> [1,2,2,2]
async function runSliceDecompositionReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/slice.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted slice model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    const X = Float32Array.from({ length: 1*2*5*6 }, (_, i) => i / 10); // deterministic values
    const feeds = { X: new Tensor('float32', X, [1, 2, 5, 6]) };

    console.log('\n=== Comparing slice and reconverted ===');
    printInputs('slice', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    if (!eq) console.error('❌ Slice outputs differ beyond tolerance.');
    else console.log('✅ Slice outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('slice reconversion test', err);
  }
}

// Sum: Y = Sum(A[2,3], B[1,3], C[]) -> [2,3]
async function runSumDecompositionReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/sum_variadic.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted sum model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    // Deterministic inputs
    const A = Float32Array.from([ 1, 2, 3,  4, 5, 6 ]);       // [2,3]
    const B = Float32Array.from([ 10, 20, 30 ]);              // [1,3]
    const C = Float32Array.from([ 0.5 ]);                     // scalar []

    const feeds = {
      A: new Tensor('float32', A, [2, 3]),
      B: new Tensor('float32', B, [1, 3]),
      C: new Tensor('float32', C, []),
    };

    console.log('\n=== Comparing sum_variadic and reconverted ===');
    printInputs('sum_variadic', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    if (!eq) console.error('❌ Sum outputs differ beyond tolerance.');
    else console.log('✅ Sum outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('sum_variadic reconversion test', err);
  }
}

export async function runPadDecompositionEquivalenceTest() {
  const originalPath = 'examples/onnx/pad_normal.onnx';
  const decomposedPath = 'examples/onnx/pad_decomposed.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);

  try {
    // Sanity: models created by python script
    if (!fs.existsSync(originalPath) || !fs.existsSync(decomposedPath)) {
      throw new Error('Pad models not found. Run: python "python scripts/make_pad_models.py"');
    }

    // Input matches the python script (N=1,C=2,H=3,W=4); values = i/10
    const X = Float32Array.from({ length: 1*2*3*4 }, (_, i) => i / 10);
    const feeds = { X: new Tensor('float32', X, [1, 2, 3, 4]) };

    console.log('\n=== Comparing pad_normal and pad_decomposed ===');
    printInputs('pad', feeds);

    const origSession = await InferenceSession.create(originalPath);
    const origOut = await origSession.run(feeds);
    const o = Array.from(Object.values(origOut)[0].data as Float32Array);

    const tol = 1e-6;

    // Reconversion of the normal model
    console.log('\n=== Running CLI to generate reconverted pad_normal model ===');
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    console.log('\n=== Comparing pad_normal and reconverted ===');
    const recSession = await InferenceSession.create(reconvertedPath);
    const recOut = await recSession.run(feeds);
    const r = Array.from(Object.values(recOut)[0].data as Float32Array);

    console.log('→ original[0:16]:   ', o.slice(0, 16));
    console.log('→ reconverted[0:16]:', r.slice(0, 16));

    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) <= tol);
    console.log('✅ pad_normal vs reconverted equivalent:', eq);

    if (!eq) throw new Error('Pad decomposition/reconversion equivalence failed.');
  } catch (err) {
    logErrorDetails('pad decomposition/reconversion test', err);
  }
}

// === Conv (standard) → reconverted equivalence ===
async function runConvNormalReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/conv_normal.onnx';
  const reconvertedPath = getReconvertedPath(originalPath);
  const base = originalPath.replace(/\.onnx$/, '');
  const reconvertedJson = `${base}_reconverted.json`;

  try {
    console.log('\n=== Running CLI to generate reconverted conv_normal model ===');
    // Generate both ONNX and JSON so we can read the full input signature.
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }
    if (!fs.existsSync(reconvertedJson)) {
      console.error(`❌ Reconverted JSON not found: ${reconvertedJson}`);
      return;
    }

    // Build feeds for ALL model inputs (Conv typically has X, W, and maybe B)
    const j = JSON.parse(fs.readFileSync(reconvertedJson, 'utf-8'));
    const g = j?.graph;
    if (!g || !Array.isArray(g.input) || g.input.length === 0) {
      throw new Error(`No inputs found in ${reconvertedJson}`);
    }

    const typeMap: Record<string, 'float32'|'float64'|'int32'|'int64'|'bool'> = {
      'FLOAT': 'float32', '1': 'float32',
      'DOUBLE': 'float64', '11': 'float64',
      'INT32': 'int32',   '6': 'int32',
      'INT64': 'int64',   '7': 'int64',
      'BOOL':  'bool',    '9': 'bool',
    };

    const feeds: Record<string, Tensor> = {};
    for (const inp of g.input) {
      const name: string = inp.name || 'X';
      const tt = inp.type?.tensorType || {};
      const elem = tt.elemType ?? tt.elem_type ?? 'FLOAT';
      const dtype = (typeMap[String(elem)] ?? 'float32') as 'float32'|'float64'|'int32'|'int64'|'bool';

      // Use concrete dimValue if present; fallback to 1 for unknowns
      const dims: number[] = (tt.shape?.dim ?? []).map((d: any) => {
        if (typeof d?.dimValue === 'number') return d.dimValue;
        if (typeof d?.dim_value === 'number') return d.dim_value;
        return 1;
      });
      if (dims.length === 0) dims.push(1);

      const size = dims.reduce((a: number, b: number) => a * b, 1);

      let tensor: Tensor;
      if (dtype === 'float64') {
        const data = Float64Array.from({ length: size }, () => Math.random() * 2 - 1);
        tensor = new Tensor('float64', data, dims);
      } else if (dtype === 'float32') {
        const data = Float32Array.from({ length: size }, () => Math.random() * 2 - 1);
        tensor = new Tensor('float32', data, dims);
      } else if (dtype === 'int32') {
        const data = Int32Array.from({ length: size }, () => Math.floor(Math.random() * 7) - 3);
        tensor = new Tensor('int32', data, dims);
      } else if (dtype === 'int64') {
        const data = Array.from({ length: size }, () => BigInt(Math.floor(Math.random() * 7) - 3));
        tensor = new Tensor('int64', data as any, dims);
      } else {
        const data = Uint8Array.from({ length: size }, () => (Math.random() > 0.5 ? 1 : 0));
        tensor = new Tensor('bool', data, dims);
      }

      feeds[name] = tensor;
    }

    console.log('\n=== Comparing conv_normal and reconverted ===');
    printInputs('conv_normal', feeds);

    // Run both models
    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5; // float convs → small numeric tolerance
    const eq = o.length === r.length && o.every((v, i) => Math.abs((v as number) - (r[i] as number)) <= tol);

    if (!eq) {
      console.log('→ original[0:16]:', o.slice(0, 16));
      console.log('→ reconverted[0:16]:', r.slice(0, 16));
      throw new Error('conv_normal vs reconverted not equivalent');
    }
    console.log('✅ conv_normal vs reconverted equivalent: true');
  } catch (err) {
    logErrorDetails('conv_normal reconversion test', err);
  }
}

async function runConvSimpleReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/conv_simple.onnx';
  const extIndex = originalPath.lastIndexOf('.');
  const base = extIndex === -1 ? originalPath : originalPath.slice(0, extIndex);
  const reconvertedPath = `${base}_reconverted.onnx`;

  try {
    console.log('\n=== Running CLI to generate reconverted conv_simple model ===');
    // Keep it consistent with other *decomposed* tests in this file
    execSync(`node ./out/src/index.js ${originalPath} --format dot -vz 0 -v 0`, {
      stdio: 'inherit',
    });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // Same input shapes/values as the standard test
    const X = Float32Array.from({ length: 1 * 1 * 4 * 4 }, () => Math.random() * 2 - 1);
    const W = Float32Array.from({ length: 1 * 1 * 3 * 3 }, () => Math.random() * 2 - 1);
    const B = new Float32Array([ (Math.random() * 2 - 1) ]);

    const feeds = {
      X: new Tensor('float32', X, [1, 1, 4, 4]),
      W: new Tensor('float32', W, [1, 1, 3, 3]),
      B: new Tensor('float32', B, [1]),
    };

    console.log('\n=== Comparing conv_simple and its reconverted version ===');
    printInputs('conv_simple', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v,i)=>Math.abs(v-r[i])<tol);
    console.log('✅ conv_simple outputs equivalent:', eq);
  } catch (err) {
    logErrorDetails('conv_simple reconversion test', err);
  }
}

async function runGemmReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/gemm_standard.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted gemm_standard model ===');
    // Standard models: JSON roundtrip (no --noLowLevel)
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const feeds = {
      A: new Tensor('float32', Float32Array.from({ length: 6 }, () => Math.random() * 10), [2, 3]),
      B: new Tensor('float32', Float32Array.from({ length: 12 }, () => Math.random() * 10), [3, 4]),
      C: new Tensor('float32', Float32Array.from({ length: 8 }, () => Math.random() * 10), [2, 4]),
    };

    console.log('\n=== Comparing gemm_standard and its reconverted version ===');
    printInputs('gemm_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-5;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) < tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (e) {
    logErrorDetails('gemm_standard reconversion test', e);
  }
}

async function runConcatReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/concat_standard.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted concat_standard model ===');
    // Standard models: JSON roundtrip (no --noLowLevel)
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    const feeds = {
      X0: new Tensor('float32', Float32Array.from({ length: 6 }, () => Math.random() * 10), [2, 3]), // [2,3]
      X1: new Tensor('float32', Float32Array.from({ length: 8 }, () => Math.random() * 10), [2, 4]), // [2,4]
      X2: new Tensor('float32', Float32Array.from({ length: 4 }, () => Math.random() * 10), [2, 2]), // [2,2]
    };

    console.log('\n=== Comparing concat_standard and its reconverted version ===');
    printInputs('concat_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    // Concat should be an exact match
    const eq = o.length === r.length && o.every((v, i) => v === r[i]);
    console.log('✅ Outputs equivalent:', eq);
  } catch (e) {
    logErrorDetails('concat_standard reconversion test', e);
  }
}

async function runDequantizeLinearReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/dequantize_standard.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted dequantize_standard model ===');
    // Standard models: JSON roundtrip (no --noLowLevel)
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // Match the Python script: X uint8 [2,3,4], S float32 [3] (per-axis axis=1), Z uint8 [3]
    const X = Uint8Array.from({ length: 2 * 3 * 4 }, () => Math.floor(Math.random() * 256));
    const S = Float32Array.from({ length: 3 }, () => Math.random() * 0.05 + 0.01); // avoid zeros
    const Z = Uint8Array.from({ length: 3 }, () => Math.floor(Math.random() * 256));

    const feeds = {
      X: new Tensor('uint8', X, [2, 3, 4]),
      S: new Tensor('float32', S, [3]),
      Z: new Tensor('uint8', Z, [3]),
    };

    console.log('\n=== Comparing dequantize_standard and its reconverted version ===');
    printInputs('dequantize_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6; // float math tolerance
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) <= tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (e) {
    logErrorDetails('dequantize_standard reconversion test', e);
  }
}

async function runAveragePoolReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/avgpool_standard.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted avgpool_standard model ===');
    // Standard models: JSON roundtrip (no --noLowLevel)
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // Match the Python script defaults: N=1, C=2, H=5, W=6 (NCHW)
    const N = 1, C = 2, H = 5, W = 6;
    const X = Float32Array.from({ length: N * C * H * W }, () => Math.random() * 2 - 1);

    const feeds = {
      X: new Tensor('float32', X, [N, C, H, W]),
    };

    console.log('\n=== Comparing avgpool_standard and its reconverted version ===');
    printInputs('avgpool_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) <= tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (e) {
    logErrorDetails('avgpool_standard reconversion test', e);
  }
}

async function runReduceSumReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/reducesum_standard.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted reducesum_standard model ===');
    // Standard models: JSON roundtrip (no --noLowLevel)
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // Match the Python script: X float32 [2,3,4], ReduceSum over axis=1 with keepdims=1
    const X = Float32Array.from({ length: 2 * 3 * 4 }, () => Math.random() * 2 - 1);

    const feeds = {
      X: new Tensor('float32', X, [2, 3, 4]),
    };

    console.log('\n=== Comparing reducesum_standard and its reconverted version ===');
    printInputs('reducesum_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6; // float tolerance
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) <= tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (e) {
    logErrorDetails('reducesum_standard reconversion test', e);
  }
}

async function runReduceMaxReconversionEquivalenceTest() {
  const originalPath = 'examples/onnx/reducemax_standard.onnx';
  const reconvertedPath = originalPath.replace(/\.onnx$/, '_reconverted.onnx');

  try {
    console.log('\n=== Running CLI to generate reconverted reducemax_standard model ===');
    // Standard models: JSON roundtrip (no --noLowLevel)
    execSync(`node ./out/src/index.js ${originalPath} --format json -vz 0 -v 0`, { stdio: 'inherit' });

    if (!fs.existsSync(reconvertedPath)) {
      console.error(`❌ Reconverted file not found: ${reconvertedPath}`);
      return;
    }

    // Match generator defaults: X float32 [2,3,4]
    const X = Float32Array.from({ length: 2 * 3 * 4 }, () => Math.random() * 2 - 1);

    const feeds = {
      X: new Tensor('float32', X, [2, 3, 4]),
    };

    console.log('\n=== Comparing reducemax_standard and its reconverted version ===');
    printInputs('reducemax_standard', feeds);

    const orig = await (await InferenceSession.create(originalPath)).run(feeds);
    const rec  = await (await InferenceSession.create(reconvertedPath)).run(feeds);

    const o = Array.from(Object.values(orig)[0].data as Float32Array);
    const r = Array.from(Object.values(rec )[0].data as Float32Array);

    console.log('→ original:', o);
    console.log('→ reconverted:', r);

    const tol = 1e-6;
    const eq = o.length === r.length && o.every((v, i) => Math.abs(v - r[i]) <= tol);
    console.log('✅ Outputs equivalent:', eq);
  } catch (e) {
    logErrorDetails('reducemax_standard reconversion test', e);
  }
}




// Tests to run
// Standard vs Decomposed Equivalence
await runVectorAddEquivalenceTest();
await runAddChainEquivalenceTest();
await runMatmulEquivalenceTest();
await runMatmulAddEquivalenceTest();

// Standard vs Standadard Reconverted Equivalence
await runVectorAddStandardReconversionEquivalenceTest();
await runAddChainStandardReconversionEquivalenceTest();
await runMatmulStandardReconversionEquivalenceTest();
await runMatmulAddStandardReconversionEquivalenceTest();

// Decomposed vs Decomposed Reconverted Equivalence
await runVectorAddDecomposedReconversionEquivalenceTest();
await runAddChainDecomposedReconversionEquivalenceTest();
await runMatmulDecomposedReconversionEquivalenceTest();
await runMatmulAddDecomposedReconversionEquivalenceTest();

// Complete Equivalence
await runVectorAddCompleteEquivalenceTest();
await runAddChainCompleteEquivalenceTest();
await runMatmulCompleteEquivalenceTest();
await runMatmulAddCompleteEquivalenceTest();

// Range & Transpose
await runRangeStandardReconversionEquivalenceTest();
await runRangeAddStandardReconversionEquivalenceTest();
await runTransposeStandardReconversionEquivalenceTest();
await runTransposeAddStandardReconversionEquivalenceTest();
await runMatmulTransposeStandardReconversionEquivalenceTest();

// Other Element-wise Ops
await runReluStandardReconversionEquivalenceTest();
await runSigmoidStandardReconversionEquivalenceTest();
await runTanhStandardReconversionEquivalenceTest();
await runExpStandardReconversionEquivalenceTest();
await runUnaryBinaryComboReconversionEquivalenceTest();
await runSumDecompositionReconversionEquivalenceTest();

// Broadcast Element-wise
await runAddScalarVectorBroadcastTest();
await runAddRowVectorToMatrixBroadcastTest();
await runAddColVectorToMatrixBroadcastTest();
await runMul3DChannelBroadcastTest();
await runChainBroadcastTest();

await runTransposeBroadcast2DReconversionEquivalenceTest();
await runTransposeBroadcast3DReconversionEquivalenceTest();
await runTransposeBroadcast4DReconversionEquivalenceTest();
await runTransposeBroadcast5DReconversionEquivalenceTest();

/*
await runMatmulRectReconversionEquivalenceTest();
await runMatmulBiasReconversionEquivalenceTest();
await runMatmulChainRightReconversionEquivalenceTest();
await runMatmulMVReconversionEquivalenceTest();
await runMatmulVMReconversionEquivalenceTest();
*/

await runSliceDecompositionReconversionEquivalenceTest();
await runPadDecompositionEquivalenceTest();
await runClipScalarReconversionEquivalenceTest();

await runConvNormalReconversionEquivalenceTest();
await runConvSimpleReconversionEquivalenceTest();

await runGemmReconversionEquivalenceTest();
await runConcatReconversionEquivalenceTest();
await runDequantizeLinearReconversionEquivalenceTest();
await runAveragePoolReconversionEquivalenceTest();

//await runReduceSumReconversionEquivalenceTest();
//await runReduceMaxReconversionEquivalenceTest();

