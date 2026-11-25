import path from "node:path";
import {
    executeEquivalenceTest,
    resolveModelPath,
    type EquivalenceTestConfig,
    type SingleEquivResult,
} from "./ortHarness.js";
import type { PartitionTestCase } from "./partitionHarness.js";

export const jsonFullArgs = "--format json -vz 0 -v 0";
export const jsonNoLowLevelArgs = "--format json --noLowLevel -vz 0 -v 0";
export const dotFullArgs = (p: string): string =>
    `--format dot -vz 0 -v 0 -o "${p.replace(/\.onnx$/, "")}.dot"`;
export const dotNoLowLevelArgs = (p: string): string =>
    `--noLowLevel --format dot -vz 0 -v 0 -o "${p.replace(/\.onnx$/, "")}.dot"`;

export const COMPATIBILITY_TESTS: EquivalenceTestConfig[] = [
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
            { name: "start", dtype: "float32", shape: [], gen: "range", value: 0 },
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
        rtol: 1e-5,
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

export const PARTITION_TESTS: PartitionTestCase[] = [
    ...COMPATIBILITY_TESTS
        // Preserve exact original 62 shared cases + kws_ref_model_float32_standard = 63 total,
        // plus allow matching range_add_standard explicit values
        .filter((t) =>
            [
                "vector_add_standard",
                "add_chain_standard",
                "matmul_standard",
                "matmul_add_standard",
                "vectoradd_test",
                "addchain_test",
                "matmul_test",
                "matmuladd_test",
                "range_standard",
                "range_add_standard",
                "transpose_standard",
                "transpose_add_standard",
                "matmul_transpose_standard",
                "relu_standard",
                "sigmoid_standard",
                "tanh_standard",
                "exp_standard",
                "unary_binary_combo",
                "sum_standard",
                "add_scalar_vector_broadcast",
                "add_row_vector_to_matrix_broadcast",
                "add_col_vector_to_matrix_broadcast",
                "mul_3d_channel",
                "chain_broadcast",
                "transpose_broadcast_2d",
                "transpose_broadcast_3d",
                "transpose_broadcast_4d",
                "transpose_broadcast_5d",
                "matmul_bcast_left_unbatched",
                "matmul_bcast_both_sides",
                "matmul_bcast_highrank",
                "slice_decomposition",
                "pad_decomposition",
                "clip_scalar",
                "conv_normal",
                "conv_simple",
                "gemm_standard",
                "concat_standard",
                "dequantize_standard",
                "averagepool_standard",
                "reducesum_standard",
                "reducemax_standard",
                "reducemin_standard",
                "reduceprod_standard",
                "reducemean_standard",
                "reducesumsquare_standard",
                "reducel1_standard",
                "reducel2_standard",
                "reducelogsum_standard",
                "reducelogsumexp_standard",
                "softmax_standard",
                "expand_scalar_to_2x3",
                "expand_vec_to_2x3",
                "expand_batch",
                "expand_middle_dim",
                "expand_highrank",
                "ad01_fp32_standard",
                "ad01_fp32_gemm_relu_standard",
                "averagepool_kws_like",
                "SC2_X",
                "SC2_Y",
                "SC2_Z",
            ].includes(t.label),
        )
        .map((t) => ({
            label: t.label,
            originalPath: t.originalPath,
            tol: t.tol ?? 1e-4,
            specs: t.specs,
        })),
    {
        label: "kws_ref_model_float32_standard",
        originalPath: "examples/onnx/kws_ref_model_float32.onnx",
        tol: 1e-4,
        specs: [{ name: "input_1", dtype: "float32", shape: [1, 49, 10, 1] }],
    },
];

/* ============================== CLI HOOK FOR index.ts ================================== */
export async function runEquivalenceForOriginalPath(
    originalPath: string,
    options?: { labelHint?: string; skipCli?: boolean },
): Promise<SingleEquivResult> {
    const target = path.resolve(originalPath).replace(/\\/g, "/");
    const config =
        COMPATIBILITY_TESTS.find(
            (t) => path.resolve(resolveModelPath(t.originalPath)).replace(/\\/g, "/") === target,
        ) ??
        (options?.labelHint !== undefined && options.labelHint !== ""
            ? COMPATIBILITY_TESTS.find((t) => t.label === options.labelHint)
            : undefined);

    if (!config) return "no-config";

    const res = await executeEquivalenceTest(config, {
        skipCli: options?.skipCli ?? false,
        showStats: true,
        verboseOnFailure: true,
    });
    return res.status;
}
