import { InferenceSession, Tensor } from 'onnxruntime-web';

function printInputs(label: string, inputs: Record<string, Tensor>) {
  console.log(`Inputs for ${label}:`);
  for (const [key, tensor] of Object.entries(inputs)) {
    const dataArray = Array.from(tensor.data as Iterable<number>);
    console.log(`  ${key}:`, dataArray);
  }
}

function printTensorInfo(tensor: Tensor, name: string) {
  console.log(`  [${name}] shape: ${tensor.dims}, type: ${tensor.type}`);
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
    const stdSession = await InferenceSession.create('examples/onnx/standard_vector_add.onnx');
    const stdOutput = await stdSession.run({ A: feeds.A, B: feeds.B });
    const stdKey = Object.keys(stdOutput)[0];
    const stdResult = Array.from(stdOutput[stdKey].data as Float32Array);

    console.log(`standard_vector_add.onnx → Output (${stdKey}):`, stdResult);

    // Load and run scalar-loop model
    const loopSession = await InferenceSession.create('examples/onnx/scalar_loop_vector_add.onnx');
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

runVectorAddEquivalenceTest();
