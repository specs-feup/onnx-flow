import { PartitionTestCase, runPartitionTest } from "./partitioning_test_utils.js";

const tests: PartitionTestCase[] = [
    {
        label: "vector_add_standard",
        originalPath: "examples/onnx/vector_add_standard.onnx",
        tol: 1e-5,
        // Missing splitNodeId
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "add_chain_standard",
        originalPath: "examples/onnx/add_chain_standard.onnx",
        tol: 1e-5,
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
        specs: [
            { name: "A", dtype: "float32", shape: [2, 2] },
            { name: "B", dtype: "float32", shape: [2, 2] },
        ],
    },
    {
        label: "matmul_add_standard",
        originalPath: "examples/onnx/matmul_add_standard.onnx",
        specs: [
            { name: "X", dtype: "int32", shape: [3, 1] },
            { name: "A", dtype: "int32", shape: [1, 3] },
            { name: "B", dtype: "int32", shape: [3, 3] },
        ],
    },
    {
        label: "vectoradd_test",
        originalPath: "examples/onnx/vectoradd_test.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [4] },
            { name: "B", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "addchain_test",
        originalPath: "examples/onnx/addchain_test.onnx",
        tol: 1e-5,
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
        specs: [
            { name: "A", dtype: "float32", shape: [2, 2] },
            { name: "B", dtype: "float32", shape: [2, 2] },
        ],
    },
    {
        label: "matmuladd_test",
        originalPath: "examples/onnx/matmuladd_test.onnx",
        specs: [
            { name: "X", dtype: "int32", shape: [3, 3] },
            { name: "A", dtype: "int32", shape: [3, 3] },
            { name: "B", dtype: "int32", shape: [3, 3] },
        ],
    },
    {
        label: "range_standard",
        originalPath: "examples/onnx/range_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "start", dtype: "float32", shape: [] },
            { name: "limit", dtype: "float32", shape: [] },
            { name: "delta", dtype: "float32", shape: [] },
        ],
    },
    {
        label: "range_add_standard",
        originalPath: "examples/onnx/range_add_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "start", dtype: "float32", shape: [] },
            { name: "limit", dtype: "float32", shape: [] },
            { name: "delta", dtype: "float32", shape: [] },
            // L = ceil((6-1)/1.5) = 4
            { name: "V", dtype: "float32", shape: [4] },
        ],
    },
    {
        label: "transpose_standard",
        originalPath: "examples/onnx/transpose_standard.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3] }],
    },
    {
        label: "transpose_add_standard",
        originalPath: "examples/onnx/transpose_add_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 3] },
            { name: "Y", dtype: "float32", shape: [3, 2] },
        ],
    },
    {
        label: "matmul_transpose_standard",
        originalPath: "examples/onnx/matmul_transpose_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [3, 4] },
        ],
    },
    {
        label: "relu_standard",
        originalPath: "examples/onnx/relu_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [6] }],
    },
    {
        label: "sigmoid_standard",
        originalPath: "examples/onnx/sigmoid_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [6] }],
    },
    {
        label: "tanh_standard",
        originalPath: "examples/onnx/tanh_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [6] }],
    },
    {
        label: "exp_standard",
        originalPath: "examples/onnx/exp_standard.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [6] }],
    },

    // ── unary/binary combo
    {
        label: "unary_binary_combo",
        originalPath: "examples/onnx/unary_binary_combo.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [6] },
            { name: "A", dtype: "float32", shape: [6] },
            { name: "B", dtype: "float32", shape: [6] },
            { name: "Y", dtype: "float32", shape: [6] },
            { name: "S", dtype: "float32", shape: [6] },
        ],
    },

    // ── sum
    {
        label: "sum_standard",
        originalPath: "examples/onnx/sum_variadic.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "B", dtype: "float32", shape: [1, 3] },
            { name: "C", dtype: "float32", shape: [] },
        ],
    },

    // ── broadcast add/mul
    {
        label: "add_scalar_vector_broadcast",
        originalPath: "examples/onnx/add_scalar_vector.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [6] }, // vector
            { name: "S", dtype: "float32", shape: [] }, // scalar
        ],
    },
    {
        label: "add_row_vector_to_matrix_broadcast",
        originalPath: "examples/onnx/add_row_vector_to_matrix.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] }, // matrix
            { name: "B", dtype: "float32", shape: [3] }, // row vector
        ],
    },
    {
        label: "add_col_vector_to_matrix_broadcast",
        originalPath: "examples/onnx/add_col_vector_to_matrix.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] }, // matrix
            { name: "C", dtype: "float32", shape: [2, 1] }, // column vector
        ],
    },
    {
        label: "mul_3d_channel",
        originalPath: "examples/onnx/mul_3d_channel.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 3, 4] }, // [C,H,W]
            { name: "W", dtype: "float32", shape: [1, 3, 1] }, // per-channel weight
        ],
    },
    {
        label: "chain_broadcast",
        originalPath: "examples/onnx/chain_broadcast.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] },
            { name: "b_row", dtype: "float32", shape: [3] },
            { name: "c_col", dtype: "float32", shape: [2, 1] },
            { name: "s_sub", dtype: "float32", shape: [] },
            { name: "s_div", dtype: "float32", shape: [] },
        ],
    },

    // ── transpose + broadcast (2D..5D)
    {
        label: "transpose_broadcast_2d",
        originalPath: "examples/onnx/transpose_broadcast_2d.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 3] },
            { name: "Y", dtype: "float32", shape: [3] },
        ],
    },
    {
        label: "transpose_broadcast_3d",
        originalPath: "examples/onnx/transpose_broadcast_3d.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 1, 3] },
            { name: "Zin", dtype: "float32", shape: [1, 3, 1] },
        ],
    },
    {
        label: "transpose_broadcast_4d",
        originalPath: "examples/onnx/transpose_broadcast_4d.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 1, 3, 1] },
            { name: "B", dtype: "float32", shape: [3, 1, 1, 1] },
        ],
    },
    {
        label: "transpose_broadcast_5d",
        originalPath: "examples/onnx/transpose_broadcast_5d.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 2, 1, 3, 1] },
            { name: "C", dtype: "float32", shape: [1, 3, 1, 1, 1] },
        ],
    },

    // ── matmul broadcast
    {
        label: "matmul_bcast_left_unbatched",
        originalPath: "examples/onnx/matmul_bcast_left_unbatched.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3, 4] },
            { name: "B", dtype: "float32", shape: [4, 5] },
        ],
    },
    {
        label: "matmul_bcast_both_sides",
        originalPath: "examples/onnx/matmul_bcast_both_sides.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [1, 3, 4] },
            { name: "B", dtype: "float32", shape: [2, 4, 5] },
        ],
    },
    {
        label: "matmul_bcast_highrank",
        originalPath: "examples/onnx/matmul_bcast_highrank.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 1, 3, 4] },
            { name: "B", dtype: "float32", shape: [1, 5, 4, 6] },
        ],
    },

    // ── slice/pad/clip
    {
        label: "slice_decomposition",
        originalPath: "examples/onnx/slice.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [1, 2, 5, 6] }],
    },
    {
        label: "pad_decomposition",
        originalPath: "examples/onnx/pad_normal.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [1, 2, 3, 4] }],
    },
    {
        label: "clip_scalar",
        originalPath: "examples/onnx/clip_scalar.onnx",
        tol: 1e-6,
        specs: [
            { name: "X", dtype: "float32", shape: [2, 3] },
            { name: "Min", dtype: "float32", shape: [] },
            { name: "Max", dtype: "float32", shape: [] },
        ],
    },

    // ── conv
    {
        label: "conv_normal",
        originalPath: "examples/onnx/conv_normal.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 1, 4, 4] }, // 16 elems
            { name: "W", dtype: "float32", shape: [1, 1, 3, 3] }, // 9 elems
            { name: "B", dtype: "float32", shape: [1] },
        ],
    },
    {
        label: "conv_simple",
        originalPath: "examples/onnx/conv_simple.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "float32", shape: [1, 1, 4, 4] },
            { name: "W", dtype: "float32", shape: [1, 1, 3, 3] },
            { name: "B", dtype: "float32", shape: [1] },
        ],
    },

    // ── gemm/concat/dequantize/avgpool
    {
        label: "gemm_standard",
        originalPath: "examples/onnx/gemm_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "A", dtype: "float32", shape: [2, 3] }, // 6
            { name: "B", dtype: "float32", shape: [3, 4] }, // 12
            { name: "C", dtype: "float32", shape: [2, 4] }, // 8
        ],
    },

    {
        label: "concat_standard",
        originalPath: "examples/onnx/concat_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "X0", dtype: "float32", shape: [2, 3] }, // 6
            { name: "X1", dtype: "float32", shape: [2, 4] }, // 8
            { name: "X2", dtype: "float32", shape: [2, 2] }, // 4  => total 18 elems
        ],
    },
    {
        label: "dequantize_standard",
        originalPath: "examples/onnx/dequantize_standard.onnx",
        tol: 1e-5,
        specs: [
            { name: "X", dtype: "uint8", shape: [2, 3, 4] }, // 24 elems total
            { name: "S", dtype: "float32", shape: [3] }, // per-channel scales
            { name: "Z", dtype: "uint8", shape: [3] }, // per-channel zero-points
        ],
    },

    {
        label: "averagepool_standard",
        originalPath: "examples/onnx/avgpool_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [1, 2, 5, 6] }],
    },

    // ===== Reduce ops (JSON) =====
    {
        label: "reducesum_standard",
        originalPath: "examples/onnx/reducesum_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        label: "reducemax_standard",
        originalPath: "examples/onnx/reducemax_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },

    // ----- More Reduce ops (JSON) -----
    {
        label: "reducemin_standard",
        originalPath: "examples/onnx/reducemin_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        label: "reduceprod_standard",
        originalPath: "examples/onnx/reduceprod_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }], // general random
    },
    {
        label: "reducemean_standard",
        originalPath: "examples/onnx/reducemean_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        label: "reducesumsquare_standard",
        originalPath: "examples/onnx/reducesumsquare_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        label: "reducel1_standard",
        originalPath: "examples/onnx/reducel1_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        label: "reducel2_standard",
        originalPath: "examples/onnx/reducel2_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        // sum(x) must be positive for log(sum(x)): use ones for a stable positive sum
        label: "reducelogsum_standard",
        originalPath: "examples/onnx/reducelogsum_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },
    {
        // log(sum(exp(x))) works for any x; 'ones' is fine and avoids extreme magnitudes
        label: "reducelogsumexp_standard",
        originalPath: "examples/onnx/reducelogsumexp_standard.onnx",
        tol: 1e-6,
        specs: [{ name: "X", dtype: "float32", shape: [2, 3, 4] }],
    },

    // ----- Softmax -----
    {
        label: "softmax_standard",
        originalPath: "examples/onnx/softmax_standard.onnx",
        specs: [{ name: "X", dtype: "float32", shape: [8, 3] }],
        tol: 1e-4,
    },

    // ----- Expand -----
    {
        label: "expand_scalar_to_2x3",
        originalPath: "examples/onnx/expand_scalar_to_2x3.onnx",
        tol: 1e-5,
        // Only X is fed; 'shape' is Constant in the graph
        specs: [{ name: "X", dtype: "float32", shape: [] }],
    },
    {
        label: "expand_vec_to_2x3",
        originalPath: "examples/onnx/expand_vec_to_2x3.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [3] }],
    },
    {
        label: "expand_batch",
        originalPath: "examples/onnx/expand_batch.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [1, 4, 5] }],
    },
    {
        label: "expand_middle_dim",
        originalPath: "examples/onnx/expand_middle_dim.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [2, 1, 4] }],
    },
    {
        label: "expand_highrank",
        originalPath: "examples/onnx/expand_highrank.onnx",
        tol: 1e-5,
        specs: [{ name: "X", dtype: "float32", shape: [1, 3, 1, 5] }],
    },

    // TinyML and SC Models (or subset of models)

    {
        label: "ad01_fp32_standard",
        originalPath: "examples/onnx/ad01_fp32.onnx",
        tol: 1e-4,
        specs: [{ name: "input_1", dtype: "float32", shape: [1, 640] }],
    },

    {
        label: "ad01_fp32_gemm_relu_standard",
        originalPath: "examples/onnx/ad01_fp32_gemm_relu.onnx",
        tol: 1e-4, // relaxed a bit, should be fine
        specs: [
            {
                name: "input_1",
                dtype: "float32",
                shape: [1, 640],
            },
            // output_1 [1, 128] float32 will be picked up automatically
        ],
    },

    {
        label: "kws_ref_model_float32_standard",
        originalPath: "examples/onnx/kws_ref_model_float32.onnx",
        tol: 1e-4, // softmax tail needs a little tolerance
        specs: [
            { name: "input_1", dtype: "float32", shape: [1, 49, 10, 1] }, // in
            // out: Identity [1,12] float32 (picked up automatically)
        ],
    },

    {
        label: "averagepool_kws_like",
        originalPath: "examples/onnx/avgpool_kws_like.onnx",
        tol: 1e-6,
        specs: [
            // Matches the KWS AveragePool input: [1, 64, 25, 5]
            { name: "X", dtype: "float32", shape: [1, 64, 25, 5] },
        ],
    },

    {
        label: "SC2_X",
        originalPath: "examples/onnx/SC2_X_toy.onnx",
        tol: 1e-4,
        specs: [{ name: "input", dtype: "float32", shape: [2, 1] }],
    },

    {
        label: "SC2_Y",
        originalPath: "examples/onnx/SC2_Y_toy.onnx",
        tol: 1e-4,
        specs: [{ name: "input", dtype: "float32", shape: [2, 1] }],
    },

    {
        label: "SC2_Z",
        originalPath: "examples/onnx/SC2_Z_toy.onnx",
        tol: 1e-4,
        specs: [{ name: "input", dtype: "float32", shape: [2, 1] }],
    },
];

async function runAll() {
    let passed = 0;
    let failed = 0;

    for (const t of tests) {
        try {
            await runPartitionTest(t);
            passed++;
        } catch (e) {
            failed++;
            console.error(`   ❌ Failed: ${e.message}`);
            // Print stack trace for debugging
            if (e.stack) {
                console.error("   Stack Trace:");
                console.error(
                    e.stack
                        .split("\n")
                        .map((line: string) => "      " + line)
                        .join("\n"),
                );
            }
        }
    }

    console.log(`\nResults: ${passed} Passed, ${failed} Failed.`);
    if (failed > 0) process.exit(1);
}

runAll();
