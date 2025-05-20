import { InferenceSession, Tensor } from 'onnxruntime-web';

async function runLoopModel() {
  // Load ONNX model
  const session = await InferenceSession.create('examples/onnx/loop_example.onnx');

  // Generate random inputs
  const tripCountValue = BigInt(Math.floor(Math.random() * 10 + 1)); // Random N from 1 to 10
  const condValue = true;
  const initialSumValue = BigInt(0);

  const feeds = {
    trip_count: new Tensor('int64', [tripCountValue], []),
    cond: new Tensor('bool', [condValue], []),
    initial_sum: new Tensor('int64', [initialSumValue], []),
  };

  console.log("Running ONNX model with inputs:");
  console.log(`trip_count (N): ${tripCountValue}`);
  console.log(`initial_sum: ${initialSumValue}`);

  // Run inference
  const output = await session.run(feeds);
  const finalSumTensor = output.final_sum;

  // Display outputs
  console.log("Model output:");
  console.log(`Final Sum (sum of numbers 0 to ${tripCountValue - BigInt(1)}): ${finalSumTensor.data[0]}`);
}

runLoopModel().catch(err => console.error("Error running model:", err));
