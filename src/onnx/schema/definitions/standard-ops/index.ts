import type { OpSchema } from "../../OpSchema.js";
import { ActivationOps } from "./Activations.js";
import { ControlFlowOps } from "./ControlFlow.js";
import { DataMovementOps } from "./DataMovement.js";
import { GeneratorOps } from "./Generator.js";
import { MathOps } from "./Math.js";
import { NeuralNetOps } from "./NeuralNet.js";
import { NormalizationOps } from "./Normalization.js";
import { OptionalOps } from "./Optional.js";
import { QuantizationOps } from "./Quantization.js";
import { ReductionOps } from "./Reduction.js";
import { RNNOps } from "./RNN.js";
import { SearchOps } from "./Search.js";
import { SequenceOps } from "./Sequence.js";

// --- Aggregate all into one export ---
export const StandardOps: OpSchema[] = [
    ...MathOps,
    ...DataMovementOps,
    ...ReductionOps,
    ...GeneratorOps,
    ...NeuralNetOps,
    ...NormalizationOps,
    ...RNNOps,
    ...ControlFlowOps,
    ...QuantizationOps,
    ...ActivationOps,
    ...SearchOps,
    ...SequenceOps,
    ...OptionalOps,
];
