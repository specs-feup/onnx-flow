# ONNX Operation Status and Roadmap

_Last updated: 2026-01-26._

## Categories (how we group operations)

- **Element-wise**  
  Scalar math/logic applied independently to each element with NumPy-style broadcasting (e.g., `Add`, `Mul`, `Cos`, comparisons).
- **Reductions / Scans**  
  Aggregate along axes or sequentially (e.g., `ReduceSum`, `ArgMax`, `CumSum`).
- **Indexing / Slicing / Reordering**  
  Shape-preserving(ish) data movement: `Gather*`, `Slice`, `DepthToSpace`, `Pad`, `Concat`, `Split`, `Transpose`.
- **Shape & Layout / Meta**  
  Shape/value meta operations that don’t touch data or only change view: `Reshape`, `Squeeze/Unsqueeze`, `Shape`, `Size`, `Cast/Expand`, `Range`, `Identity`.
- **Linear Algebra**  
  Tensor contractions and matrix operations: `MatMul`, `Gemm`, `Einsum`, `Det`, `Transpose` (as reordering).
- **Activations & Normalizations**  
  Nonlinearities and normalizers: `Relu`, `Sigmoid`, `Tanh`, `Gelu`, `BatchNormalization`, `LayerNormalization`, etc.
- **Convolution & Pooling**  
  Spatial kernels and friends: `Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`, ROI operations, `GridSample`, `DeformConv`, `Col2Im`.
- **Signal / Spectral**  
  DSP and windows: `DFT`, `STFT`, `BlackmanWindow`, `HannWindow`, `MelWeightMatrix`.
- **Complex AI Blocks**  
  Higher-level layers: `Attention`, `RotaryEmbedding`, `RNN`, `GRU`, `LSTM`.
- **Control Flow & Sequences**  
  Graph-level control or sequence containers: `If`, `Loop`, `Scan`, and `Sequence*` operations.
- **Quantization**  
  Precision boundaries and quantized kernels: `QuantizeLinear`, `DequantizeLinear`, `QLinearConv`, `QLinearMatMul`, `MatMulInteger`, `ConvInteger`.
- **Strings & Text**  
  String processing: `StringSplit`, `StringConcat`, `RegexFullMatch`.
- **Utility & I/O**  
  Data ingress/egress or bookkeeping: `ImageDecoder`, `Dropout`, constants.
- **Random / Init**  
  Samplers and initializers: `RandomNormal*`, `RandomUniform*`, `Bernoulli`, `Constant*`, `EyeLike`.

## Features

### Current features (what our optimizer does today)
- **Loop decomposition**  
  Break supported operations into explicit loops to unlock fusion and locality.
- **Loop fusion**  
  Merge compatible loops (same iteration space / alignment) to reduce memory traffic and kernel launches.
- **Loop coalescing**  
  Combine adjacent loops to reduce passes over memory and improve cache behavior.


### Possibel future, generalizable features (with estimated effort) / Suggested Roadmap
- **XS/S (small passes)**  
  Algebraic peepholes & strength reductions · Constant folding (scalars/tiny tensors) · Dead code elimination · Shape folding & no-op removal · Redundant reshape/cast collapsing · Graph integrity checks.
- **M (medium analysis/rewrites)**  
  Transpose/permute sinking & cancellation · Redundant data-movement pruning (merge/kill copies, Slice/Concat chains) · Redundant gather elimination · CSE/GVN-lite · LICM within loop bodies · Pattern-rewrite engine (rule DSL) · Cost model v0 (heuristics).
- **L (deeper performance)**  
  Vectorization/SIMD (e.g., WASM/LLVM intrinsics) · Tiling/blocking of loop nests · Quantization-aware graph handling (Q/DQ plumbing, boundary minimization) · Memory planner v0 (buffer reuse & liveness).

_All of the above are generally applicable and **not tied to a specific kernel scheduler**._

## Status states (with emoji)

- ✅ **Implemented:** our current passes (loop decomposition/fusion/coalescing) already apply to this op family.
- 🟨 **To be implemented:** we intend to extend **our current** passes to these operations.
- 🔵 **Target of future features:** current passes don’t fit; the **general features** above are good candidates.
- 🚫 **Not a target:** currently out of scope (e.g., pure I/O, randomness, or training-only losses).

---

## Full operator list (ONNX order) with category & status

> Operator order mirrors the official “ONNX Operators” index.  
> “Feature suggestions” are shown **only** for 🔵 rows and reference the **general** features listed above (although they could also be applied to ✅ and 🟨 rows).

| Operator | Category | Status | Feature suggestions (for 🔵 only) |
|---|---|---|---|
| Abs | Element-wise | ✅ | - |
| Acos | Element-wise | ✅ | - |
| Acosh | Element-wise | ✅ | - |
| Add | Element-wise | ✅ | - |
| AffineGrid | Imaging/Geometry | 🔵 | Transpose sinking; redundant data-move pruning; vectorization; tiling |
| And | Comparisons & Logic | ✅ | - |
| ArgMax | Reduction | 🔵 | Constant folding; vectorization; tiling; LICM for surrounding shape operations |
| ArgMin | Reduction | 🔵 | Constant folding; vectorization; tiling; LICM |
| Asin | Element-wise | ✅ | - |
| Asinh | Element-wise | ✅ | - |
| Atan | Element-wise | ✅ | - |
| Atanh | Element-wise | ✅ | - |
| Attention | Complex AI Block | 🟨 | - |
| AveragePool | Convolution & Pooling | ✅ | - |
| BatchNormalization | Normalization | 🔵 | BN folding (weights rewrite); vectorization; tiling |
| Bernoulli | Random / Init | 🚫 | - |
| BitShift | Element-wise (bitwise) | ✅ | - |
| BitwiseAnd | Element-wise (bitwise) | ✅ | - |
| BitwiseNot | Element-wise (bitwise) | ✅ | - |
| BitwiseOr | Element-wise (bitwise) | ✅ | - |
| BitwiseXor | Element-wise (bitwise) | ✅ | - |
| BlackmanWindow | Signal / Windows | 🔵 | Constant folding; vectorization |
| Cast | Shape & Layout / Meta | 🔵 | Redundant cast collapsing; DCE; shape folding |
| CastLike | Shape & Layout / Meta | 🔵 | Redundant cast collapsing; DCE; shape folding |
| Ceil | Element-wise | ✅ | - |
| Celu | Activation | ✅ | - |
| CenterCropPad | Imaging/Geometry | 🔵 | Redundant data-move pruning; vectorization; tiling |
| Clip | Element-wise | ✅ | - |
| Col2Im | Conv/Im2Col family | 🔵 | Redundant data-move pruning; tiling; vectorization |
| Compress | Indexing / Slicing | 🔵 | Redundant movement pruning; DCE; CSE/GVN-lite |
| Concat | Shape/Layout/Reorder | ✅ | - |
| ConcatFromSequence | Sequences | 🔵 | DCE; redundant movement pruning |
| Constant | Init / Const | 🚫 | - |
| ConstantOfShape | Init / Const | 🚫 | - |
| Conv | Convolution & Pooling | ✅ | - |
| ConvInteger | Convolution & Pooling / Quant | 🟨 | - |
| ConvTranspose | Convolution & Pooling | 🟨 | - |
| Cos | Element-wise | ✅ | - |
| Cosh | Element-wise | ✅ | - |
| CumSum | Scan / Reduction-like | 🔵 | Vectorization; tiling; LICM |
| DFT | Signal / Spectral | 🔵 | Vectorization; tiling; constant folding (static windows) |
| DeformConv | Convolution & Pooling | 🟨 | - |
| DepthToSpace | Indexing / Reorder | 🔵 | Redundant movement pruning; transpose sinking |
| DequantizeLinear | Quantization | ✅ | - |
| Det | Linear Algebra | 🔵 | Transpose sinking; vectorization; tiling |
| Div | Element-wise | ✅ | - |
| Dropout | Utility | 🚫 | - |
| DynamicQuantizeLinear | Quantization | 🔵 | Quant-aware graph handling; DCE |
| Einsum | Linear Algebra (contraction) | 🔵 | Transpose sinking; tiling; vectorization |
| Elu | Activation | ✅ | - |
| Equal | Comparisons & Logic | ✅ | - |
| Erf | Element-wise | ✅ | - |
| Exp | Element-wise | ✅ | - |
| Expand | Shape & Layout / Broadcast | ✅ | - |
| EyeLike | Init / Const | 🚫 | - |
| Flatten | Shape & Layout | 🔵 | Redundant reshape/flatten collapsing; DCE |
| Floor | Element-wise | ✅ | - |
| GRU | Complex AI Block | 🟨 | - |
| Gather | Indexing / Slicing | 🔵 | Redundant gather elimination; CSE/GVN-lite; movement pruning |
| GatherElements | Indexing / Slicing | 🔵 | Redundant gather elimination; movement pruning |
| GatherND | Indexing / Slicing | 🔵 | Redundant gather elimination; movement pruning |
| Gelu | Activation | ✅ | - |
| Gemm | Linear Algebra | ✅ | - |
| GlobalAveragePool | Reduction / Pooling | 🔵 | Vectorization; tiling; constant folding when static |
| GlobalLpPool | Reduction / Pooling | 🔵 | Vectorization; tiling |
| GlobalMaxPool | Reduction / Pooling | 🔵 | Vectorization; tiling |
| Greater | Comparisons & Logic | ✅ | - |
| GreaterOrEqual | Comparisons & Logic | ✅ | - |
| GridSample | Sampling / Geometry | 🔵 | Vectorization; tiling; movement pruning |
| GroupNormalization | Normalization | 🔵 | Vectorization; tiling; constant-param folding |
| HammingWindow | Signal / Windows | 🔵 | Constant folding; vectorization |
| HannWindow | Signal / Windows | 🔵 | Constant folding; vectorization |
| HardSigmoid | Activation | ✅ | - |
| HardSwish | Activation | ✅ | - |
| Hardmax | Activation (axis) | 🔵 | Vectorization; tiling |
| Identity | Meta | 🚫 | - |
| If | Control Flow | 🔵 | Dead-branch elimination; LICM (hoist invariants) |
| ImageDecoder | IO / Imaging | 🚫 | - |
| InstanceNormalization | Normalization | 🔵 | Vectorization; tiling; constant-param folding |
| IsInf | Element-wise (predicate) | ✅ | - |
| IsNaN | Element-wise (predicate) | ✅ | - |
| LRN | Normalization | 🔵 | Vectorization; tiling |
| LSTM | Complex AI Block | 🟨 | - |
| LayerNormalization | Normalization | 🔵 | Vectorization; tiling |
| LeakyRelu | Activation | ✅ | - |
| Less | Comparisons & Logic | ✅ | - |
| LessOrEqual | Comparisons & Logic | ✅ | - |
| Log | Element-wise | ✅ | - |
| LogSoftmax | Activation (axis) | 🔵 | Vectorization; tiling |
| Loop | Control Flow | 🔵 | LICM; loop unrolling (static trip count); DCE |
| LpNormalization | Normalization | 🔵 | Vectorization; tiling |
| LpPool | Convolution & Pooling | 🟨 | - |
| MatMul | Linear Algebra | ✅ | - |
| MatMulInteger | Linear Algebra / Quant | 🟨 | - |
| Max | Element-wise (n-ary) | ✅ | - |
| MaxPool | Convolution & Pooling | 🟨 | - |
| MaxRoiPool | Convolution & Pooling (ROI) | 🟨 | - |
| MaxUnpool | Convolution & Pooling | 🟨 | - |
| Mean | Element-wise (n-ary) | ✅ | - |
| MeanVarianceNormalization | Normalization | 🔵 | Vectorization; tiling |
| MelWeightMatrix | Signal / Feature | 🔵 | Constant folding; vectorization |
| Min | Element-wise (n-ary) | ✅ | - |
| Mish | Activation | ✅ | - |
| Mod | Element-wise | ✅ | - |
| Mul | Element-wise | ✅ | - |
| Multinomial | Random / Sampling | 🚫 | - |
| Neg | Element-wise | ✅ | - |
| NegativeLogLikelihoodLoss | Loss / Training | 🚫 | - |
| NonMaxSuppression | Indexing / Selection | 🔵 | Movement pruning; CSE/GVN-lite |
| NonZero | Indexing / Selection | 🔵 | Movement pruning; DCE |
| Not | Logic | ✅ | - |
| OneHot | Indexing / Reorder | 🔵 | Movement pruning; constant folding (static indices) |
| Optional | Optional / Meta | 🔵 | DCE; constant folding (presence flags) |
| OptionalGetElement | Optional / Meta | 🔵 | DCE |
| OptionalHasElement | Optional / Meta | 🔵 | DCE |
| Or | Logic | ✅ | - |
| PRelu | Activation | ✅ | - |
| Pad | Indexing / Slicing | ✅ | - |
| Pow | Element-wise | ✅ | - |
| QLinearConv | Convolution & Pooling / Quant | 🟨 | - |
| QLinearMatMul | Linear Algebra / Quant | 🟨 | - |
| QuantizeLinear | Quantization | 🟨 | - |
| RMSNormalization | Normalization | 🔵 | Vectorization; tiling |
| RNN | Complex AI Block | 🟨 | - |
| RandomNormal | Random / Init | 🚫 | - |
| RandomNormalLike | Random / Init | 🚫 | - |
| RandomUniform | Random / Init | 🚫 | - |
| RandomUniformLike | Random / Init | 🚫 | - |
| Range | Meta (range gen) | ✅ | - |
| Reciprocal | Element-wise | ✅ | - |
| ReduceL1 | Reduction | ✅ | - |
| ReduceL2 | Reduction | ✅ | - |
| ReduceLogSum | Reduction | ✅ | - |
| ReduceLogSumExp | Reduction | ✅ | - |
| ReduceMax | Reduction | ✅ | - |
| ReduceMean | Reduction | ✅ | - |
| ReduceMin | Reduction | ✅ | - |
| ReduceProd | Reduction | ✅ | - |
| ReduceSum | Reduction | ✅ | - |
| ReduceSumSquare | Reduction | ✅ | - |
| RegexFullMatch | Strings & Text | 🚫 | - |
| Relu | Activation | ✅ | - |
| Reshape | Shape & Layout | 🔵 | Redundant reshape collapsing; DCE; shape folding |
| Resize | Imaging / Resampling | 🔵 | Movement pruning; vectorization; tiling |
| ReverseSequence | Indexing / Reorder | 🔵 | Movement pruning |
| RoiAlign | Convolution & Pooling (ROI) | 🟨 | - |
| RotaryEmbedding | Positional / Embedding | 🔵 | Vectorization; tiling |
| Round | Element-wise | ✅ | - |
| STFT | Signal / Spectral | 🔵 | Vectorization; tiling |
| Scan | Control Flow (iterator) | 🔵 | LICM; dead-branch elimination; DCE |
| Scatter | Indexing / Scatter | 🔵 | Movement pruning; CSE/GVN-lite |
| ScatterElements | Indexing / Scatter | 🔵 | Movement pruning; CSE/GVN-lite |
| ScatterND | Indexing / Scatter | 🔵 | Movement pruning; CSE/GVN-lite |
| Selu | Activation | ✅ | - |
| SequenceAt | Sequences | 🔵 | DCE; movement pruning |
| SequenceConstruct | Sequences | 🔵 | DCE |
| SequenceEmpty | Sequences | 🔵 | DCE |
| SequenceErase | Sequences | 🔵 | DCE |
| SequenceInsert | Sequences | 🔵 | DCE |
| SequenceLength | Sequences | 🔵 | DCE |
| SequenceMap | Sequences / Control-Flow Adj. | 🔵 | DCE |
| Shape | Shape & Layout / Meta | 🔵 | Shape folding; LICM (hoist out of loops) |
| Shrink | Element-wise (threshold) | ✅ | - |
| Sigmoid | Activation | ✅ | - |
| Sign | Element-wise | ✅ | - |
| Sin | Element-wise | ✅ | - |
| Sinh | Element-wise | ✅ | - |
| Size | Shape & Layout / Meta | 🔵 | Shape folding; LICM |
| Slice | Indexing / Slicing | ✅ | - |
| Softmax | Activation (axis) | ✅ | - |
| SoftmaxCrossEntropyLoss | Loss / Training | 🚫 | - |
| Softplus | Activation | ✅ | - |
| Softsign | Activation | ✅ | - |
| SpaceToDepth | Indexing / Reorder | 🔵 | Movement pruning; transpose sinking |
| Split | Shape/Layout / Reorder | 🔵 | Movement pruning; DCE |
| SplitToSequence | Sequences | 🔵 | DCE |
| Sqrt | Element-wise | ✅ | - |
| Squeeze | Shape & Layout | 🔵 | No-op elimination; DCE |
| StringConcat | Strings & Text | 🚫 | - |
| StringNormalizer | Strings & Text | 🚫 | - |
| StringSplit | Strings & Text | 🚫 | - |
| Sub | Element-wise | ✅ | - |
| Sum | Element-wise (n-ary) | ✅ | - |
| Swish | Activation | ✅ | - |
| Tan | Element-wise | ✅ | - |
| Tanh | Activation | ✅ | - |
| ThresholdedRelu | Activation | ✅ | - |
| Tile | Indexing / Reorder | 🔵 | Movement pruning; CSE/GVN-lite |
| TopK | Indexing / Selection | 🔵 | Movement pruning; vectorization |
| Transpose | Linear Algebra / Reorder | ✅ | - |
| Trilu | Indexing / Masking | 🔵 | Movement pruning; transpose sinking |
| Unique | Indexing / Selection | 🔵 | Movement pruning |
| Unsqueeze | Shape & Layout | 🔵 | No-op elimination; DCE |
| Upsample | Imaging / Resampling (legacy) | 🚫 | - |
| Where | Select / Logic | ✅ | - |
| Xor | Logic | ✅ | - |

