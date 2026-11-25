import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../src/Onnx/GraphBuilder.js";
import { json2onnx } from "../../src/json2onnx.js";
import { initializeSchemaRegistry } from "../../src/Onnx/Schema/index.js";
import inferShapes from "../../src/Onnx/InferShapes.js";
import {
    DataType,
    type AttributeMap,
    type KnownShape,
    type RawOnnxAttribute,
    type RawOnnxGraph,
    type RawOnnxModel,
    type RawOnnxNode,
    type RawOnnxValueInfo,
    type TensorProto,
} from "../../src/Onnx/OnnxTypes.js";
import { safeWriteJson } from "../../src/Onnx/Utils.js";
import {
    ensureProtoExists,
    exportGraphToOnnx,
    resolveModelPath,
    type DType,
    type FeedSpec,
} from "./ortHarness.js";

/* ============================== SYNTHETIC GRAPH BUILDER ================================== */
export interface SyntheticCase {
    name: string;
    opType: string;
    feeds: FeedSpec[];
    constInputs?: { index: number; proto: TensorProto }[];
    inputShapes?: KnownShape[];
    attrs?: AttributeMap;
    outputNames?: string[];
    outputSpecs?: { type: DataType; shape: KnownShape }[];
    tol?: number | Record<string, number>;
    rtol?: number;
    exact?: boolean;
    /** Skip LoopLoweringPass and verify canonicalized graph directly in ORT */
    skipLoopLowering?: boolean;
}

export const FEED_DTYPE_MAP: Record<DType, DataType> = {
    float32: DataType.FLOAT,
    int32: DataType.INT32,
    int64: DataType.INT64,
    uint8: DataType.UINT8,
    int8: DataType.INT8,
    bool: DataType.BOOL,
};

export function buildSyntheticGraph(c: SyntheticCase): OnnxGraph.Class {
    initializeSchemaRegistry();
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const builder = new GraphBuilder(graph, "synth");

    const totalInputs = Math.max(
        c.feeds.length,
        ...(c.constInputs?.map((ci) => ci.index + 1) ?? [0]),
    );

    const constByIdx = new Map(c.constInputs?.map((ci) => [ci.index, ci.proto]) ?? []);
    const opInputs = [];
    let feedCursor = 0;

    for (let i = 0; i < totalInputs; i++) {
        const constProto = constByIdx.get(i);
        if (constProto) {
            opInputs.push(builder.createConstant(`c_${i}`, constProto));
        } else {
            const feedIdx = feedCursor++;
            const f = c.feeds[feedIdx];
            const graphShape = c.inputShapes?.[feedIdx] ?? f.shape;
            opInputs.push(
                graph
                    .addNode(f.name)
                    .init(new TensorNode.Builder(FEED_DTYPE_MAP[f.dtype], graphShape, "input"))
                    .as(TensorNode),
            );
        }
    }

    const outNames = c.outputNames ?? ["Y"];
    builder.createOpWithExact("target_op", c.opType, opInputs, outNames, c.attrs ?? {});
    inferShapes(graph);

    const outNodes = graph.getNodeById("target_op")!.outgoers.targets.toArray();
    outNodes.forEach((out, idx) => {
        if (out.is(TensorNode)) {
            const tn = out.as(TensorNode);
            tn.setType("output");
            const spec = c.outputSpecs?.[idx];
            if (spec) {
                tn.setLiteralType(spec.type);
                tn.setShape(spec.shape);
            }
        }
    });
    return graph;
}

export { exportGraphToOnnx };

/* ============================== RAW ONNX MODEL HELPERS ================================== */
function vi(name: string, elemType: DataType, dims: number[]): RawOnnxValueInfo {
    return {
        name,
        type: {
            tensorType: {
                elemType,
                shape: { dim: dims.map((d) => ({ dimValue: d })) },
            },
        },
    };
}

function attrInt(name: string, i: number): RawOnnxAttribute {
    return { name, type: "INT", i };
}

function attrFloat(name: string, f: number): RawOnnxAttribute {
    return { name, type: "FLOAT", f };
}

function attrString(name: string, s: string): RawOnnxAttribute {
    return { name, type: "STRING", s: Buffer.from(s, "utf8").toString("base64") };
}

function attrInts(name: string, ints: number[]): RawOnnxAttribute {
    return { name, type: "INTS", ints };
}

function attrGraph(name: string, g: RawOnnxGraph): RawOnnxAttribute {
    return { name, type: "GRAPH", g };
}

function attrTensor(name: string, t: TensorProto): RawOnnxAttribute {
    return { name, type: "TENSOR", t };
}

function constI64Node(outName: string, dims: number[], vals: number[]): RawOnnxNode {
    return {
        opType: "Constant",
        input: [],
        output: [outName],
        attribute: [
            attrTensor("value", {
                name: outName,
                dataType: DataType.INT64,
                dims,
                int64Data: vals,
            }),
        ],
    };
}

function makeModel(
    inputs: [string, DataType, number[]][],
    outputs: [string, DataType, number[]][],
    nodes: RawOnnxNode[],
    options: { opset?: number; initializers?: TensorProto[]; irVersion?: number } = {},
): RawOnnxModel {
    const opset = options.opset ?? 13;
    return {
        irVersion: options.irVersion ?? (opset >= 19 ? 9 : 7),
        opsetImport: [{ domain: "", version: opset }],
        graph: {
            name: "fixture_graph",
            input: inputs.map(([n, t, s]) => vi(n, t, s)),
            output: outputs.map(([n, t, s]) => vi(n, t, s)),
            initializer: options.initializers ?? [],
            node: nodes,
        },
    };
}

function singleOpModel(
    opType: string,
    inputs: [string, DataType, number[]][],
    outputs: [string, DataType, number[]][],
    attrs: RawOnnxAttribute[] = [],
    opset = 13,
): RawOnnxModel {
    return makeModel(
        inputs,
        outputs,
        [
            {
                opType,
                input: inputs.map(([n]) => n),
                output: outputs.map(([n]) => n),
                attribute: attrs,
            },
        ],
        { opset },
    );
}

function makeVectorAddDecomposed(): RawOnnxModel {
    const body: RawOnnxGraph = {
        name: "loop_body",
        input: [
            vi("iter_idx", DataType.INT64, []),
            vi("cond_in", DataType.BOOL, []),
            vi("carry", DataType.FLOAT, [4]),
        ],
        output: [vi("cond_out", DataType.BOOL, []), vi("carry_out", DataType.FLOAT, [4])],
        node: [
            constI64Node("axes", [1], [0]),
            { opType: "Unsqueeze", input: ["iter_idx", "axes"], output: ["idx"] },
            {
                opType: "Gather",
                input: ["A", "idx"],
                output: ["a_val"],
                attribute: [attrInt("axis", 0)],
            },
            {
                opType: "Gather",
                input: ["B", "idx"],
                output: ["b_val"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Add", input: ["a_val", "b_val"], output: ["sum_val"] },
            {
                opType: "ScatterElements",
                input: ["carry", "idx", "sum_val"],
                output: ["carry_out"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Identity", input: ["cond_in"], output: ["cond_out"] },
        ],
    };
    return makeModel(
        [
            ["trip_count", DataType.INT64, []],
            ["cond", DataType.BOOL, []],
            ["A", DataType.FLOAT, [4]],
            ["B", DataType.FLOAT, [4]],
        ],
        [["final_vec", DataType.FLOAT, [4]]],
        [
            {
                opType: "Loop",
                input: ["trip_count", "cond", "init_carry"],
                output: ["final_vec"],
                attribute: [attrGraph("body", body)],
            },
        ],
        {
            opset: 17,
            initializers: [
                {
                    name: "init_carry",
                    dataType: DataType.FLOAT,
                    dims: [4],
                    floatData: [0, 0, 0, 0],
                },
            ],
        },
    );
}

function makeAddChainDecomposed(): RawOnnxModel {
    const body: RawOnnxGraph = {
        name: "loop_body",
        input: [
            vi("iter", DataType.INT64, []),
            vi("cond_in", DataType.BOOL, []),
            vi("carry", DataType.FLOAT, [4]),
        ],
        output: [vi("cond_out", DataType.BOOL, []), vi("carry_out", DataType.FLOAT, [4])],
        node: [
            constI64Node("axes", [1], [0]),
            { opType: "Unsqueeze", input: ["iter", "axes"], output: ["idx"] },
            {
                opType: "Gather",
                input: ["A", "idx"],
                output: ["a_val"],
                attribute: [attrInt("axis", 0)],
            },
            {
                opType: "Gather",
                input: ["B", "idx"],
                output: ["b_val"],
                attribute: [attrInt("axis", 0)],
            },
            {
                opType: "Gather",
                input: ["C", "idx"],
                output: ["c_val"],
                attribute: [attrInt("axis", 0)],
            },
            {
                opType: "Gather",
                input: ["D", "idx"],
                output: ["d_val"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Add", input: ["a_val", "b_val"], output: ["ab_val"] },
            { opType: "Add", input: ["c_val", "d_val"], output: ["cd_val"] },
            { opType: "Add", input: ["ab_val", "cd_val"], output: ["sum_val"] },
            {
                opType: "ScatterElements",
                input: ["carry", "idx", "sum_val"],
                output: ["carry_out"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Identity", input: ["cond_in"], output: ["cond_out"] },
        ],
    };
    return makeModel(
        [
            ["A", DataType.FLOAT, [4]],
            ["B", DataType.FLOAT, [4]],
            ["C", DataType.FLOAT, [4]],
            ["D", DataType.FLOAT, [4]],
            ["trip_count", DataType.INT64, []],
            ["cond", DataType.BOOL, []],
        ],
        [["Y", DataType.FLOAT, [4]]],
        [
            {
                opType: "Loop",
                input: ["trip_count", "cond", "init_carry"],
                output: ["Y"],
                attribute: [attrGraph("body", body)],
            },
        ],
        {
            opset: 13,
            initializers: [
                {
                    name: "init_carry",
                    dataType: DataType.FLOAT,
                    dims: [4],
                    floatData: [0, 0, 0, 0],
                },
            ],
        },
    );
}

function makeMatMulDecomposed(): RawOnnxModel {
    const body: RawOnnxGraph = {
        name: "MatMulLoopBody",
        input: [
            vi("i", DataType.INT64, []),
            vi("cond_in", DataType.BOOL, []),
            vi("carry", DataType.FLOAT, [4]),
        ],
        output: [vi("cond_out", DataType.BOOL, []), vi("carry_out", DataType.FLOAT, [4])],
        initializer: [
            { name: "const_N", dataType: DataType.INT64, dims: [], int64Data: [2] },
            { name: "axes0", dataType: DataType.INT64, dims: [1], int64Data: [0] },
            { name: "shape_1", dataType: DataType.INT64, dims: [1], int64Data: [1] },
            { name: "reshape_k", dataType: DataType.INT64, dims: [1], int64Data: [2] },
        ],
        node: [
            { opType: "Unsqueeze", input: ["i", "axes0"], output: ["idx"] },
            { opType: "Div", input: ["i", "const_N"], output: ["row"] },
            { opType: "Mod", input: ["i", "const_N"], output: ["col"] },
            { opType: "Unsqueeze", input: ["row", "axes0"], output: ["row_idx"] },
            { opType: "Unsqueeze", input: ["col", "axes0"], output: ["col_idx"] },
            {
                opType: "Gather",
                input: ["A", "row_idx"],
                output: ["a_row"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Reshape", input: ["a_row", "reshape_k"], output: ["a_vec"] },
            {
                opType: "Gather",
                input: ["B", "col_idx"],
                output: ["b_col"],
                attribute: [attrInt("axis", 1)],
            },
            { opType: "Reshape", input: ["b_col", "reshape_k"], output: ["b_vec"] },
            { opType: "Mul", input: ["a_vec", "b_vec"], output: ["prod"] },
            {
                opType: "ReduceSum",
                input: ["prod", "axes0"],
                output: ["dot"],
                attribute: [attrInt("keepdims", 0)],
            },
            { opType: "Reshape", input: ["dot", "shape_1"], output: ["dot_unsqueezed"] },
            {
                opType: "ScatterElements",
                input: ["carry", "idx", "dot_unsqueezed"],
                output: ["carry_out"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Identity", input: ["cond_in"], output: ["cond_out"] },
        ],
    };
    return makeModel(
        [
            ["A", DataType.FLOAT, [2, 2]],
            ["B", DataType.FLOAT, [2, 2]],
            ["trip_count", DataType.INT64, []],
            ["cond", DataType.BOOL, []],
        ],
        [["Y", DataType.FLOAT, [2, 2]]],
        [
            {
                opType: "Loop",
                input: ["trip_count", "cond", "init_carry"],
                output: ["Y_flat"],
                attribute: [attrGraph("body", body)],
            },
            { opType: "Reshape", input: ["Y_flat", "final_shape"], output: ["Y"] },
        ],
        {
            opset: 13,
            initializers: [
                {
                    name: "init_carry",
                    dataType: DataType.FLOAT,
                    dims: [4],
                    floatData: [0, 0, 0, 0],
                },
                { name: "final_shape", dataType: DataType.INT64, dims: [2], int64Data: [2, 2] },
            ],
        },
    );
}

function makeMatMulAddDecomposed(): RawOnnxModel {
    const body: RawOnnxGraph = {
        name: "loop_body",
        input: [
            vi("i", DataType.INT64, []),
            vi("cond_in", DataType.BOOL, []),
            vi("carry", DataType.INT32, [9]),
        ],
        output: [vi("cond_out", DataType.BOOL, []), vi("carry_out", DataType.INT32, [9])],
        initializer: [
            { name: "axes0", dataType: DataType.INT64, dims: [1], int64Data: [0] },
            { name: "N", dataType: DataType.INT64, dims: [], int64Data: [3] },
            { name: "shape1", dataType: DataType.INT64, dims: [1], int64Data: [1] },
        ],
        node: [
            { opType: "Unsqueeze", input: ["i", "axes0"], output: ["i_unsqueezed"] },
            { opType: "Div", input: ["i", "N"], output: ["row"] },
            { opType: "Mod", input: ["i", "N"], output: ["col"] },
            { opType: "Unsqueeze", input: ["row", "axes0"], output: ["row_idx"] },
            { opType: "Unsqueeze", input: ["col", "axes0"], output: ["col_idx"] },
            {
                opType: "Gather",
                input: ["X", "row_idx"],
                output: ["x_val"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Reshape", input: ["x_val", "shape1"], output: ["x_scalar"] },
            {
                opType: "Gather",
                input: ["A", "col_idx"],
                output: ["a_val"],
                attribute: [attrInt("axis", 1)],
            },
            { opType: "Reshape", input: ["a_val", "shape1"], output: ["a_scalar"] },
            { opType: "Mul", input: ["x_scalar", "a_scalar"], output: ["dot"] },
            {
                opType: "Gather",
                input: ["B", "row_idx"],
                output: ["b_row"],
                attribute: [attrInt("axis", 0)],
            },
            {
                opType: "Gather",
                input: ["b_row", "col_idx"],
                output: ["b_val"],
                attribute: [attrInt("axis", 1)],
            },
            { opType: "Reshape", input: ["b_val", "shape1"], output: ["b_scalar"] },
            { opType: "Add", input: ["dot", "b_scalar"], output: ["sum_val"] },
            {
                opType: "ScatterElements",
                input: ["carry", "i_unsqueezed", "sum_val"],
                output: ["carry_out"],
                attribute: [attrInt("axis", 0)],
            },
            { opType: "Identity", input: ["cond_in"], output: ["cond_out"] },
        ],
    };
    return makeModel(
        [
            ["X", DataType.INT32, [3, 1]],
            ["A", DataType.INT32, [1, 3]],
            ["B", DataType.INT32, [3, 3]],
            ["trip_count", DataType.INT64, []],
            ["cond", DataType.BOOL, []],
        ],
        [["Y", DataType.INT32, [3, 3]]],
        [
            {
                opType: "Loop",
                input: ["trip_count", "cond", "init_carry"],
                output: ["Y_flat"],
                attribute: [attrGraph("body", body)],
            },
            { opType: "Reshape", input: ["Y_flat", "final_shape"], output: ["Y"] },
        ],
        {
            opset: 13,
            initializers: [
                {
                    name: "init_carry",
                    dataType: DataType.INT32,
                    dims: [9],
                    int32Data: Array(9).fill(0) as number[],
                },
                { name: "final_shape", dataType: DataType.INT64, dims: [2], int64Data: [3, 3] },
            ],
        },
    );
}

function makeSc2ToyFallback(divScale: number): RawOnnxModel {
    const scanLen = 8;
    const refPts = Array.from({ length: scanLen }, (_, i) => i + 1);
    const weights = Array.from({ length: scanLen }, (_, i) => ((i % 5) - 2) * 0.1);
    const body: RawOnnxGraph = {
        name: "sc2_scan_body",
        input: [vi("next_in", DataType.FLOAT, [2, 1]), vi("next", DataType.FLOAT, [1])],
        output: [vi("next_out", DataType.FLOAT, [2, 1]), vi("scan_out", DataType.FLOAT, [2])],
        initializer: [{ name: "rss_axes", dataType: DataType.INT64, dims: [1], int64Data: [1] }],
        node: [
            { opType: "Identity", input: ["next_in"], output: ["next_out"] },
            { opType: "Sub", input: ["next_in", "next"], output: ["diff"] },
            {
                opType: "ReduceSumSquare",
                input: ["diff", "rss_axes"],
                output: ["scan_out"],
                attribute: [attrInt("keepdims", 0)],
            },
        ],
    };
    return makeModel(
        [["input", DataType.FLOAT, [2, 1]]],
        [["GPmean", DataType.FLOAT, [2, 1]]],
        [
            { opType: "Div", input: ["input", "Di_Divcst"], output: ["Di_C0"] },
            { opType: "Div", input: ["Di_Divcst1", "Di_Divcst"], output: ["Di_C02"] },
            {
                opType: "Scan",
                input: ["Di_C0", "Di_C02"],
                output: ["u0", "u1"],
                attribute: [attrGraph("body", body), attrInt("num_scan_inputs", 1)],
            },
            {
                opType: "Transpose",
                input: ["u1"],
                output: ["tp"],
                attribute: [attrInts("perm", [1, 0])],
            },
            { opType: "Mul", input: ["tp", "kgpd_Mulcst"], output: ["kc0"] },
            { opType: "Neg", input: ["kc0"], output: ["ky0"] },
            { opType: "Exp", input: ["ky0"], output: ["kout"] },
            { opType: "MatMul", input: ["kout", "gpr_MatMulcst"], output: ["gy0"] },
            { opType: "Mul", input: ["gy0", "gpr_Mulcst"], output: ["gc1"] },
            { opType: "Add", input: ["gc1", "gpr_Addcst"], output: ["gc0"] },
            { opType: "Reshape", input: ["gc0", "Re_Reshapecst"], output: ["GPmean"] },
        ],
        {
            opset: 19,
            initializers: [
                { name: "Di_Divcst", dataType: DataType.FLOAT, dims: [1], floatData: [divScale] },
                {
                    name: "Di_Divcst1",
                    dataType: DataType.FLOAT,
                    dims: [scanLen, 1],
                    floatData: refPts,
                },
                { name: "kgpd_Mulcst", dataType: DataType.FLOAT, dims: [1], floatData: [0.5] },
                {
                    name: "gpr_MatMulcst",
                    dataType: DataType.FLOAT,
                    dims: [scanLen],
                    floatData: weights,
                },
                { name: "gpr_Mulcst", dataType: DataType.FLOAT, dims: [1], floatData: [0.4] },
                { name: "gpr_Addcst", dataType: DataType.FLOAT, dims: [1], floatData: [-0.2] },
                { name: "Re_Reshapecst", dataType: DataType.INT64, dims: [2], int64Data: [-1, 1] },
            ],
        },
    );
}

function makeAd01GemmReluFallback(): RawOnnxModel {
    const w = Array.from({ length: 640 * 128 }, (_, i) => (((i * 7) % 13) - 6) * 0.01);
    const b = Array.from({ length: 128 }, (_, i) => ((i % 5) - 2) * 0.05);
    return makeModel(
        [["input_1", DataType.FLOAT, [1, 640]]],
        [["output_1", DataType.FLOAT, [1, 128]]],
        [
            {
                opType: "Gemm",
                input: ["input_1", "w0", "b0"],
                output: ["gemm_out"],
                attribute: [
                    attrFloat("alpha", 1),
                    attrFloat("beta", 1),
                    attrInt("transA", 0),
                    attrInt("transB", 0),
                ],
            },
            { opType: "Relu", input: ["gemm_out"], output: ["output_1"] },
        ],
        {
            opset: 13,
            initializers: [
                { name: "w0", dataType: DataType.FLOAT, dims: [640, 128], floatData: w },
                { name: "b0", dataType: DataType.FLOAT, dims: [128], floatData: b },
            ],
        },
    );
}

function makeAd01Fallback(): RawOnnxModel {
    const w1 = Array.from({ length: 640 * 32 }, (_, i) => (((i * 3) % 11) - 5) * 0.01);
    const b1 = Array.from({ length: 32 }, () => 0.01);
    const w2 = Array.from({ length: 32 * 640 }, (_, i) => (((i * 5) % 11) - 5) * 0.01);
    const b2 = Array.from({ length: 640 }, () => 0.01);
    return makeModel(
        [["input_1", DataType.FLOAT, [1, 640]]],
        [["Identity", DataType.FLOAT, [1, 640]]],
        [
            {
                opType: "Gemm",
                input: ["input_1", "w1", "b1"],
                output: ["h1"],
                attribute: [
                    attrFloat("alpha", 1),
                    attrFloat("beta", 1),
                    attrInt("transA", 0),
                    attrInt("transB", 0),
                ],
            },
            { opType: "Relu", input: ["h1"], output: ["r1"] },
            {
                opType: "Gemm",
                input: ["r1", "w2", "b2"],
                output: ["Identity"],
                attribute: [
                    attrFloat("alpha", 1),
                    attrFloat("beta", 1),
                    attrInt("transA", 0),
                    attrInt("transB", 0),
                ],
            },
        ],
        {
            opset: 13,
            initializers: [
                { name: "w1", dataType: DataType.FLOAT, dims: [640, 32], floatData: w1 },
                { name: "b1", dataType: DataType.FLOAT, dims: [32], floatData: b1 },
                { name: "w2", dataType: DataType.FLOAT, dims: [32, 640], floatData: w2 },
                { name: "b2", dataType: DataType.FLOAT, dims: [640], floatData: b2 },
            ],
        },
    );
}

function makeKwsFallback(): RawOnnxModel {
    // Structural equivalent to KWS: [1, 49, 10, 1] -> Transpose/Reshape -> Conv -> Relu -> AveragePool -> Reshape -> Gemm -> Softmax -> [1, 12]
    const convW = Array.from({ length: 4 * 1 * 3 * 3 }, (_, i) => (((i * 3) % 7) - 3) * 0.05);
    const convB = Array.from({ length: 4 }, () => 0.01);
    const gemmW = Array.from({ length: 4 * 12 }, (_, i) => (((i * 5) % 9) - 4) * 0.05);
    const gemmB = Array.from({ length: 12 }, () => 0.0);
    return makeModel(
        [["input_1", DataType.FLOAT, [1, 49, 10, 1]]],
        [["Identity", DataType.FLOAT, [1, 12]]],
        [
            { opType: "Reshape", input: ["input_1", "nchw_shape"], output: ["x_nchw"] },
            {
                opType: "Conv",
                input: ["x_nchw", "convW", "convB"],
                output: ["c1"],
                attribute: [attrInts("strides", [2, 2]), attrInts("pads", [1, 1, 1, 1])],
            },
            { opType: "Relu", input: ["c1"], output: ["r1"] },
            {
                opType: "AveragePool",
                input: ["r1"],
                output: ["p1"],
                attribute: [attrInts("kernel_shape", [25, 5]), attrInts("strides", [25, 5])],
            },
            { opType: "Reshape", input: ["p1", "flat_shape"], output: ["flat"] },
            {
                opType: "Gemm",
                input: ["flat", "gemmW", "gemmB"],
                output: ["logits"],
                attribute: [attrFloat("alpha", 1), attrFloat("beta", 1)],
            },
            {
                opType: "Softmax",
                input: ["logits"],
                output: ["Identity"],
                attribute: [attrInt("axis", 1)],
            },
        ],
        {
            opset: 13,
            initializers: [
                {
                    name: "nchw_shape",
                    dataType: DataType.INT64,
                    dims: [4],
                    int64Data: [1, 1, 49, 10],
                },
                { name: "convW", dataType: DataType.FLOAT, dims: [4, 1, 3, 3], floatData: convW },
                { name: "convB", dataType: DataType.FLOAT, dims: [4], floatData: convB },
                { name: "flat_shape", dataType: DataType.INT64, dims: [2], int64Data: [1, 4] },
                { name: "gemmW", dataType: DataType.FLOAT, dims: [4, 12], floatData: gemmW },
                { name: "gemmB", dataType: DataType.FLOAT, dims: [12], floatData: gemmB },
            ],
        },
    );
}

/* ============================== REGISTRY OF ALL 99 EXAMPLE MODELS ================================== */
function makeAddChainStandard(): RawOnnxModel {
    return makeModel(
        [
            ["A", DataType.FLOAT, [4]],
            ["B", DataType.FLOAT, [4]],
            ["C", DataType.FLOAT, [4]],
            ["D", DataType.FLOAT, [4]],
        ],
        [["Y", DataType.FLOAT, [4]]],
        [
            { opType: "Add", input: ["A", "B"], output: ["AB"] },
            { opType: "Add", input: ["C", "D"], output: ["CD"] },
            { opType: "Add", input: ["AB", "CD"], output: ["Y"] },
        ],
    );
}

const MODEL_BUILDERS: Partial<Record<string, () => RawOnnxModel>> = {
    "examples/onnx/vector_add_standard.onnx": () =>
        singleOpModel(
            "Add",
            [
                ["A", DataType.FLOAT, [4]],
                ["B", DataType.FLOAT, [4]],
            ],
            [["C", DataType.FLOAT, [4]]],
        ),
    "examples/onnx/vectoradd_test.onnx": () =>
        singleOpModel(
            "Add",
            [
                ["A", DataType.FLOAT, [4]],
                ["B", DataType.FLOAT, [4]],
            ],
            [["C", DataType.FLOAT, [4]]],
        ),
    "examples/onnx/add_chain_standard.onnx": makeAddChainStandard,
    "examples/onnx/addchain_test.onnx": makeAddChainStandard,
    "examples/onnx/matmul_standard.onnx": () =>
        singleOpModel(
            "MatMul",
            [
                ["A", DataType.FLOAT, [2, 2]],
                ["B", DataType.FLOAT, [2, 2]],
            ],
            [["Y", DataType.FLOAT, [2, 2]]],
        ),
    "examples/onnx/matmul_test.onnx": () =>
        singleOpModel(
            "MatMul",
            [
                ["A", DataType.FLOAT, [2, 2]],
                ["B", DataType.FLOAT, [2, 2]],
            ],
            [["Y", DataType.FLOAT, [2, 2]]],
        ),
    "examples/onnx/matmul_add_standard.onnx": () =>
        makeModel(
            [
                ["X", DataType.INT32, [3, 1]],
                ["A", DataType.INT32, [1, 3]],
                ["B", DataType.INT32, [3, 3]],
            ],
            [["Y", DataType.INT32, [3, 3]]],
            [
                { opType: "MatMul", input: ["X", "A"], output: ["XA"] },
                { opType: "Add", input: ["B", "XA"], output: ["Y"] },
            ],
        ),
    "examples/onnx/matmuladd_test.onnx": () =>
        makeModel(
            [
                ["X", DataType.INT32, [3, 3]],
                ["A", DataType.INT32, [3, 3]],
                ["B", DataType.INT32, [3, 3]],
            ],
            [["Y", DataType.INT32, [3, 3]]],
            [
                { opType: "MatMul", input: ["X", "A"], output: ["XA"] },
                { opType: "Add", input: ["B", "XA"], output: ["Y"] },
            ],
        ),
    "examples/onnx/vector_add_decomposed.onnx": makeVectorAddDecomposed,
    "examples/onnx/add_chain_decomposed.onnx": makeAddChainDecomposed,
    "examples/onnx/matmul_decomposed.onnx": makeMatMulDecomposed,
    "examples/onnx/matmul_add_decomposed.onnx": makeMatMulAddDecomposed,
    "examples/onnx/range_standard.onnx": () =>
        singleOpModel(
            "Range",
            [
                ["start", DataType.FLOAT, []],
                ["limit", DataType.FLOAT, []],
                ["delta", DataType.FLOAT, []],
            ],
            [["Y", DataType.FLOAT, [-1]]],
        ),
    "examples/onnx/range_add_standard.onnx": () =>
        makeModel(
            [
                ["start", DataType.FLOAT, []],
                ["limit", DataType.FLOAT, []],
                ["delta", DataType.FLOAT, []],
                ["V", DataType.FLOAT, [-1]],
            ],
            [["Z", DataType.FLOAT, [-1]]],
            [
                { opType: "Range", input: ["start", "limit", "delta"], output: ["Y"] },
                { opType: "Add", input: ["Y", "V"], output: ["Z"] },
            ],
        ),
    "examples/onnx/transpose_standard.onnx": () =>
        singleOpModel(
            "Transpose",
            [["X", DataType.FLOAT, [2, 3]]],
            [["Y", DataType.FLOAT, [3, 2]]],
            [attrInts("perm", [1, 0])],
        ),
    "examples/onnx/transpose_add_standard.onnx": () =>
        makeModel(
            [
                ["X", DataType.FLOAT, [2, 3]],
                ["Y", DataType.FLOAT, [3, 2]],
            ],
            [["Z", DataType.FLOAT, [3, 2]]],
            [
                {
                    opType: "Transpose",
                    input: ["X"],
                    output: ["T"],
                    attribute: [attrInts("perm", [1, 0])],
                },
                { opType: "Add", input: ["T", "Y"], output: ["Z"] },
            ],
        ),
    "examples/onnx/matmul_transpose_standard.onnx": () =>
        makeModel(
            [
                ["A", DataType.FLOAT, [2, 3]],
                ["B", DataType.FLOAT, [3, 4]],
            ],
            [["Z", DataType.FLOAT, [4, 2]]],
            [
                { opType: "MatMul", input: ["A", "B"], output: ["M"] },
                {
                    opType: "Transpose",
                    input: ["M"],
                    output: ["Z"],
                    attribute: [attrInts("perm", [1, 0])],
                },
            ],
        ),
    "examples/onnx/relu_standard.onnx": () =>
        singleOpModel("Relu", [["X", DataType.FLOAT, [6]]], [["Y", DataType.FLOAT, [6]]]),
    "examples/onnx/sigmoid_standard.onnx": () =>
        singleOpModel("Sigmoid", [["X", DataType.FLOAT, [6]]], [["Y", DataType.FLOAT, [6]]]),
    "examples/onnx/tanh_standard.onnx": () =>
        singleOpModel("Tanh", [["X", DataType.FLOAT, [6]]], [["Y", DataType.FLOAT, [6]]]),
    "examples/onnx/exp_standard.onnx": () =>
        singleOpModel("Exp", [["X", DataType.FLOAT, [6]]], [["Y", DataType.FLOAT, [6]]]),
    "examples/onnx/unary_binary_combo.onnx": () =>
        makeModel(
            [
                ["X", DataType.FLOAT, [6]],
                ["A", DataType.FLOAT, [6]],
                ["B", DataType.FLOAT, [6]],
                ["Y", DataType.FLOAT, [6]],
                ["S", DataType.FLOAT, [6]],
            ],
            [["O", DataType.FLOAT, [6]]],
            [
                { opType: "Add", input: ["X", "A"], output: ["t1"] },
                { opType: "Mul", input: ["t1", "B"], output: ["t2"] },
                { opType: "Relu", input: ["t2"], output: ["t3"] },
                { opType: "Sigmoid", input: ["t3"], output: ["t4"] },
                { opType: "Sub", input: ["t4", "Y"], output: ["t5"] },
                { opType: "Exp", input: ["t5"], output: ["t6"] },
                { opType: "Div", input: ["t6", "S"], output: ["Z"] },
                { opType: "Tanh", input: ["Z"], output: ["O"] },
            ],
        ),
    "examples/onnx/sum_variadic.onnx": () =>
        singleOpModel(
            "Sum",
            [
                ["A", DataType.FLOAT, [2, 3]],
                ["B", DataType.FLOAT, [1, 3]],
                ["C", DataType.FLOAT, []],
            ],
            [["Y", DataType.FLOAT, [2, 3]]],
        ),
    "examples/onnx/add_scalar_vector.onnx": () =>
        singleOpModel(
            "Add",
            [
                ["X", DataType.FLOAT, [6]],
                ["S", DataType.FLOAT, []],
            ],
            [["Y", DataType.FLOAT, [6]]],
        ),
    "examples/onnx/add_row_vector_to_matrix.onnx": () =>
        singleOpModel(
            "Add",
            [
                ["A", DataType.FLOAT, [2, 3]],
                ["B", DataType.FLOAT, [3]],
            ],
            [["Y", DataType.FLOAT, [2, 3]]],
        ),
    "examples/onnx/add_col_vector_to_matrix.onnx": () =>
        singleOpModel(
            "Add",
            [
                ["A", DataType.FLOAT, [2, 3]],
                ["C", DataType.FLOAT, [2, 1]],
            ],
            [["Y", DataType.FLOAT, [2, 3]]],
        ),
    "examples/onnx/mul_3d_channel.onnx": () =>
        singleOpModel(
            "Mul",
            [
                ["X", DataType.FLOAT, [2, 3, 4]],
                ["W", DataType.FLOAT, [1, 3, 1]],
            ],
            [["Y", DataType.FLOAT, [2, 3, 4]]],
        ),
    "examples/onnx/chain_broadcast.onnx": () =>
        makeModel(
            [
                ["A", DataType.FLOAT, [2, 3]],
                ["b_row", DataType.FLOAT, [3]],
                ["s_sub", DataType.FLOAT, []],
                ["c_col", DataType.FLOAT, [2, 1]],
                ["s_div", DataType.FLOAT, []],
            ],
            [["Y", DataType.FLOAT, [2, 3]]],
            [
                { opType: "Add", input: ["A", "b_row"], output: ["t1"] },
                { opType: "Sub", input: ["t1", "s_sub"], output: ["t2"] },
                { opType: "Mul", input: ["t2", "c_col"], output: ["t3"] },
                { opType: "Div", input: ["t3", "s_div"], output: ["Y"] },
            ],
        ),
    "examples/onnx/transpose_broadcast_2d.onnx": () =>
        makeModel(
            [
                ["X", DataType.FLOAT, [1, 3]],
                ["Y", DataType.FLOAT, [3]],
            ],
            [["Z", DataType.FLOAT, [3, 1]]],
            [
                {
                    opType: "Transpose",
                    input: ["X"],
                    output: ["XT"],
                    attribute: [attrInts("perm", [1, 0])],
                },
                constI64Node("axes_unsq", [1], [1]),
                { opType: "Unsqueeze", input: ["Y", "axes_unsq"], output: ["Ycol"] },
                { opType: "Add", input: ["XT", "Ycol"], output: ["Z"] },
            ],
        ),
    "examples/onnx/transpose_broadcast_3d.onnx": () =>
        makeModel(
            [
                ["X", DataType.FLOAT, [2, 1, 3]],
                ["Zin", DataType.FLOAT, [1, 3, 1]],
            ],
            [["Out", DataType.FLOAT, [2, 3, 1]]],
            [
                {
                    opType: "Transpose",
                    input: ["X"],
                    output: ["XT"],
                    attribute: [attrInts("perm", [0, 2, 1])],
                },
                { opType: "Add", input: ["XT", "Zin"], output: ["Out"] },
            ],
        ),
    "examples/onnx/transpose_broadcast_4d.onnx": () =>
        makeModel(
            [
                ["X", DataType.FLOAT, [2, 1, 3, 1]],
                ["B", DataType.FLOAT, [3, 1, 1, 1]],
            ],
            [["Out", DataType.FLOAT, [3, 2, 1, 1]]],
            [
                {
                    opType: "Transpose",
                    input: ["X"],
                    output: ["XT"],
                    attribute: [attrInts("perm", [2, 0, 3, 1])],
                },
                { opType: "Add", input: ["XT", "B"], output: ["Out"] },
            ],
        ),
    "examples/onnx/transpose_broadcast_5d.onnx": () =>
        makeModel(
            [
                ["X", DataType.FLOAT, [1, 2, 1, 3, 1]],
                ["C", DataType.FLOAT, [1, 3, 1, 1, 1]],
            ],
            [["Out", DataType.FLOAT, [1, 3, 2, 1, 1]]],
            [
                {
                    opType: "Transpose",
                    input: ["X"],
                    output: ["XT"],
                    attribute: [attrInts("perm", [0, 3, 1, 4, 2])],
                },
                { opType: "Add", input: ["XT", "C"], output: ["Out"] },
            ],
        ),
    "examples/onnx/matmul_bcast_left_unbatched.onnx": () =>
        singleOpModel(
            "MatMul",
            [
                ["A", DataType.FLOAT, [2, 3, 4]],
                ["B", DataType.FLOAT, [4, 5]],
            ],
            [["Y", DataType.FLOAT, [2, 3, 5]]],
        ),
    "examples/onnx/matmul_bcast_both_sides.onnx": () =>
        singleOpModel(
            "MatMul",
            [
                ["A", DataType.FLOAT, [1, 3, 4]],
                ["B", DataType.FLOAT, [2, 4, 5]],
            ],
            [["Y", DataType.FLOAT, [2, 3, 5]]],
        ),
    "examples/onnx/matmul_bcast_highrank.onnx": () =>
        singleOpModel(
            "MatMul",
            [
                ["A", DataType.FLOAT, [2, 1, 3, 4]],
                ["B", DataType.FLOAT, [1, 5, 4, 6]],
            ],
            [["Y", DataType.FLOAT, [2, 5, 3, 6]]],
        ),
    "examples/onnx/slice.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [1, 2, 5, 6]]],
            [["Y", DataType.FLOAT, [1, 2, 2, 2]]],
            [
                constI64Node("starts", [2], [1, 2]),
                constI64Node("ends", [2], [5, 6]),
                constI64Node("axes", [2], [2, 3]),
                constI64Node("steps", [2], [2, 2]),
                { opType: "Slice", input: ["X", "starts", "ends", "axes", "steps"], output: ["Y"] },
            ],
        ),
    "examples/onnx/pad_normal.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [1, 2, 3, 4]]],
            [["Y", DataType.FLOAT, [1, 2, 6, 11]]],
            [
                {
                    opType: "Pad",
                    input: ["X", "pads", "pad_val"],
                    output: ["Y"],
                    attribute: [attrString("mode", "constant")],
                },
            ],
            {
                initializers: [
                    {
                        name: "pads",
                        dataType: DataType.INT64,
                        dims: [8],
                        int64Data: [0, 0, 1, 3, 0, 0, 2, 4],
                    },
                    { name: "pad_val", dataType: DataType.FLOAT, dims: [], floatData: [0.5] },
                ],
            },
        ),
    "examples/onnx/clip_scalar.onnx": () =>
        singleOpModel(
            "Clip",
            [
                ["X", DataType.FLOAT, [2, 3]],
                ["Min", DataType.FLOAT, []],
                ["Max", DataType.FLOAT, []],
            ],
            [["Y", DataType.FLOAT, [2, 3]]],
        ),
    "examples/onnx/conv_normal.onnx": () =>
        singleOpModel(
            "Conv",
            [
                ["X", DataType.FLOAT, [1, 1, 4, 4]],
                ["W", DataType.FLOAT, [1, 1, 3, 3]],
                ["B", DataType.FLOAT, [1]],
            ],
            [["Y", DataType.FLOAT, [1, 1, 2, 2]]],
            [
                attrInts("dilations", [1, 1]),
                attrInt("group", 1),
                attrInts("pads", [0, 0, 0, 0]),
                attrInts("strides", [1, 1]),
            ],
        ),
    "examples/onnx/conv_simple.onnx": () =>
        singleOpModel(
            "Conv",
            [
                ["X", DataType.FLOAT, [1, 1, 4, 4]],
                ["W", DataType.FLOAT, [1, 1, 3, 3]],
                ["B", DataType.FLOAT, [1]],
            ],
            [["Y", DataType.FLOAT, [1, 1, 2, 2]]],
            [
                attrInts("dilations", [1, 1]),
                attrInt("group", 1),
                attrInts("pads", [0, 0, 0, 0]),
                attrInts("strides", [1, 1]),
            ],
            17,
        ),
    "examples/onnx/gemm_standard.onnx": () =>
        singleOpModel(
            "Gemm",
            [
                ["A", DataType.FLOAT, [2, 3]],
                ["B", DataType.FLOAT, [3, 4]],
                ["C", DataType.FLOAT, [2, 4]],
            ],
            [["Y", DataType.FLOAT, [2, 4]]],
            [
                attrFloat("alpha", 1.2),
                attrFloat("beta", 0.5),
                attrInt("transA", 0),
                attrInt("transB", 0),
            ],
        ),
    "examples/onnx/concat_standard.onnx": () =>
        singleOpModel(
            "Concat",
            [
                ["X0", DataType.FLOAT, [2, 3]],
                ["X1", DataType.FLOAT, [2, 4]],
                ["X2", DataType.FLOAT, [2, 2]],
            ],
            [["Y", DataType.FLOAT, [2, 9]]],
            [attrInt("axis", 1)],
        ),
    "examples/onnx/dequantize_standard.onnx": () =>
        singleOpModel(
            "DequantizeLinear",
            [
                ["X", DataType.UINT8, [2, 3, 4]],
                ["S", DataType.FLOAT, [3]],
                ["Z", DataType.UINT8, [3]],
            ],
            [["Y", DataType.FLOAT, [2, 3, 4]]],
            [attrInt("axis", 1)],
        ),
    "examples/onnx/avgpool_standard.onnx": () =>
        singleOpModel(
            "AveragePool",
            [["X", DataType.FLOAT, [1, 2, 5, 6]]],
            [["Y", DataType.FLOAT, [1, 2, 3, 3]]],
            [
                attrInt("ceil_mode", 0),
                attrInt("count_include_pad", 0),
                attrInts("kernel_shape", [3, 3]),
                attrInts("pads", [1, 1, 1, 1]),
                attrInts("strides", [2, 2]),
            ],
        ),
    "examples/onnx/avgpool_kws_like.onnx": () =>
        singleOpModel(
            "AveragePool",
            [["X", DataType.FLOAT, [1, 64, 25, 5]]],
            [["Y", DataType.FLOAT, [1, 64, 1, 1]]],
            [
                attrInt("ceil_mode", 0),
                attrInt("count_include_pad", 0),
                attrInts("kernel_shape", [25, 5]),
                attrInts("strides", [25, 5]),
            ],
        ),
    "examples/onnx/reducesum_standard.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [
                constI64Node("Axes", [1], [1]),
                {
                    opType: "ReduceSum",
                    input: ["X", "Axes"],
                    output: ["Y"],
                    attribute: [attrInt("keepdims", 1)],
                },
            ],
        ),
    "examples/onnx/reducemax_standard.onnx": () =>
        singleOpModel(
            "ReduceMax",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducemin_standard.onnx": () =>
        singleOpModel(
            "ReduceMin",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reduceprod_standard.onnx": () =>
        singleOpModel(
            "ReduceProd",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducemean_standard.onnx": () =>
        singleOpModel(
            "ReduceMean",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducesumsquare_standard.onnx": () =>
        singleOpModel(
            "ReduceSumSquare",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducel1_standard.onnx": () =>
        singleOpModel(
            "ReduceL1",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducel2_standard.onnx": () =>
        singleOpModel(
            "ReduceL2",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducelogsum_standard.onnx": () =>
        singleOpModel(
            "ReduceLogSum",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/reducelogsumexp_standard.onnx": () =>
        singleOpModel(
            "ReduceLogSumExp",
            [["X", DataType.FLOAT, [2, 3, 4]]],
            [["Y", DataType.FLOAT, [2, 1, 4]]],
            [attrInts("axes", [1]), attrInt("keepdims", 1)],
        ),
    "examples/onnx/softmax_standard.onnx": () =>
        singleOpModel(
            "Softmax",
            [["X", DataType.FLOAT, [8, 3]]],
            [["Y", DataType.FLOAT, [8, 3]]],
            [attrInt("axis", 1)],
        ),
    "examples/onnx/expand_scalar_to_2x3.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, []]],
            [["Y", DataType.FLOAT, [2, 3]]],
            [
                constI64Node("shape", [2], [2, 3]),
                { opType: "Expand", input: ["X", "shape"], output: ["Y"] },
            ],
        ),
    "examples/onnx/expand_vec_to_2x3.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [3]]],
            [["Y", DataType.FLOAT, [2, 3]]],
            [
                constI64Node("shape", [2], [2, 3]),
                { opType: "Expand", input: ["X", "shape"], output: ["Y"] },
            ],
        ),
    "examples/onnx/expand_batch.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [1, 4, 5]]],
            [["Y", DataType.FLOAT, [2, 4, 5]]],
            [
                constI64Node("shape", [3], [2, 4, 5]),
                { opType: "Expand", input: ["X", "shape"], output: ["Y"] },
            ],
        ),
    "examples/onnx/expand_middle_dim.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [2, 1, 4]]],
            [["Y", DataType.FLOAT, [2, 3, 4]]],
            [
                constI64Node("shape", [3], [2, 3, 4]),
                { opType: "Expand", input: ["X", "shape"], output: ["Y"] },
            ],
        ),
    "examples/onnx/expand_highrank.onnx": () =>
        makeModel(
            [["X", DataType.FLOAT, [1, 3, 1, 5]]],
            [["Y", DataType.FLOAT, [2, 3, 4, 5]]],
            [
                constI64Node("shape", [4], [2, 3, 4, 5]),
                { opType: "Expand", input: ["X", "shape"], output: ["Y"] },
            ],
        ),

    // Benchmark / TinyML / SC2 fallbacks
    "examples/onnx/ad01_fp32.onnx": makeAd01Fallback,
    "examples/onnx/ad01_fp32_gemm_relu.onnx": makeAd01GemmReluFallback,
    "examples/onnx/kws_ref_model_float32.onnx": makeKwsFallback,
    "examples/onnx/SC2_X_toy.onnx": () => makeSc2ToyFallback(4.691421031951904),
    "examples/onnx/SC2_Y_toy.onnx": () => makeSc2ToyFallback(5.142068862915039),
    "examples/onnx/SC2_Z_toy.onnx": () => makeSc2ToyFallback(2.933396339416504),

    // Canonicalization elemental ops (opset 19)
    "examples/onnx/sub.onnx": () =>
        singleOpModel(
            "Sub",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/greater_or_equal.onnx": () =>
        singleOpModel(
            "GreaterOrEqual",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.BOOL, [2]]],
            [],
            19,
        ),
    "examples/onnx/less.onnx": () =>
        singleOpModel(
            "Less",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.BOOL, [2]]],
            [],
            19,
        ),
    "examples/onnx/less_or_equal.onnx": () =>
        singleOpModel(
            "LessOrEqual",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.BOOL, [2]]],
            [],
            19,
        ),
    "examples/onnx/min.onnx": () =>
        singleOpModel(
            "Min",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/max.onnx": () =>
        singleOpModel(
            "Max",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/abs.onnx": () =>
        singleOpModel("Abs", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/sign.onnx": () =>
        singleOpModel("Sign", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/leaky_relu.onnx": () =>
        singleOpModel(
            "LeakyRelu",
            [["A", DataType.FLOAT, [2]]],
            [["B", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/hard_sigmoid.onnx": () =>
        singleOpModel(
            "HardSigmoid",
            [["A", DataType.FLOAT, [2]]],
            [["B", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/softplus.onnx": () =>
        singleOpModel(
            "Softplus",
            [["A", DataType.FLOAT, [2]]],
            [["B", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/mish.onnx": () =>
        singleOpModel("Mish", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/elu.onnx": () =>
        singleOpModel("Elu", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/celu.onnx": () =>
        singleOpModel("Celu", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/selu.onnx": () =>
        singleOpModel("Selu", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/or.onnx": () =>
        singleOpModel(
            "Or",
            [
                ["A", DataType.BOOL, [2]],
                ["B", DataType.BOOL, [2]],
            ],
            [["C", DataType.BOOL, [2]]],
            [],
            19,
        ),
    "examples/onnx/Div.onnx": () =>
        singleOpModel(
            "Div",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/mod_fmod0.onnx": () =>
        singleOpModel(
            "Mod",
            [
                ["A", DataType.INT64, [6]],
                ["B", DataType.INT64, [6]],
            ],
            [["C", DataType.INT64, [6]]],
            [attrInt("fmod", 0)],
            19,
        ),
    "examples/onnx/mod_fmod1.onnx": () =>
        singleOpModel(
            "Mod",
            [
                ["A", DataType.FLOAT, [6]],
                ["B", DataType.FLOAT, [6]],
            ],
            [["C", DataType.FLOAT, [6]]],
            [attrInt("fmod", 1)],
            19,
        ),
    "examples/onnx/pow.onnx": () =>
        singleOpModel(
            "Pow",
            [
                ["A", DataType.FLOAT, [2]],
                ["B", DataType.FLOAT, [2]],
            ],
            [["C", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/not.onnx": () =>
        singleOpModel("Not", [["A", DataType.BOOL, [2]]], [["B", DataType.BOOL, [2]]], [], 19),
    "examples/onnx/sigmoid.onnx": () =>
        singleOpModel(
            "Sigmoid",
            [["A", DataType.FLOAT, [2]]],
            [["B", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/softsign.onnx": () =>
        singleOpModel(
            "Softsign",
            [["A", DataType.FLOAT, [2]]],
            [["B", DataType.FLOAT, [2]]],
            [],
            19,
        ),
    "examples/onnx/tan.onnx": () =>
        singleOpModel("Tan", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/sinh.onnx": () =>
        singleOpModel("Sinh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/cosh.onnx": () =>
        singleOpModel("Cosh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/tanh_sigmoid.onnx": () =>
        singleOpModel("Tanh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/tanh_div.onnx": () =>
        singleOpModel("Tanh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/asinh.onnx": () =>
        singleOpModel("Asinh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/acosh.onnx": () =>
        singleOpModel("Acosh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/atanh.onnx": () =>
        singleOpModel("Atanh", [["A", DataType.FLOAT, [2]]], [["B", DataType.FLOAT, [2]]], [], 19),
    "examples/onnx/group_normalization.onnx": () =>
        singleOpModel(
            "GroupNormalization",
            [
                ["X", DataType.FLOAT, [1, 4, 4, 4]],
                ["scale", DataType.FLOAT, [4]],
                ["bias", DataType.FLOAT, [4]],
            ],
            [["Y", DataType.FLOAT, [1, 4, 4, 4]]],
            [attrFloat("epsilon", 1e-5), attrInt("num_groups", 4)],
            19,
        ),
};

/**
 * Ensures an input ONNX model file exists on disk.
 * If it already exists, returns immediately without touching it.
 * Otherwise, generates and materializes it using the registered factory.
 */
export async function ensureModelExists(relOrAbsPath: string): Promise<string> {
    const absPath = resolveModelPath(relOrAbsPath);
    if (fs.existsSync(absPath)) {
        return absPath;
    }

    ensureProtoExists();
    const normalizedRel = relOrAbsPath.replace(/\\/g, "/");
    const builder = MODEL_BUILDERS[normalizedRel];
    if (builder === undefined) {
        throw new Error(`No model factory registered for missing ONNX file: ${relOrAbsPath}`);
    }

    const rawModel = builder();
    fs.mkdirSync(path.dirname(absPath), { recursive: true });
    const tmpJson = path.join(
        os.tmpdir(),
        `onnx-flow-gen-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.json`,
    );
    try {
        safeWriteJson(tmpJson, rawModel);
        await json2onnx(tmpJson, absPath);
    } finally {
        if (fs.existsSync(tmpJson)) {
            try {
                fs.unlinkSync(tmpJson);
            } catch {
                /* ignore */
            }
        }
    }
    return absPath;
}
