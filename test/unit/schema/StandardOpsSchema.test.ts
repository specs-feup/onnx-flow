import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/OnnxGraph.js";
import TensorNode from "../../../src/onnx/TensorNode.js";
import OperationNode from "../../../src/onnx/OperationNode.js";
import { GraphBuilder } from "../../../src/onnx/GraphBuilder.js";
import { DataType } from "../../../src/onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { OpRegistry } from "../../../src/onnx/schema/OpRegistry.js";
import { resolveExportVersions } from "../../../src/onnx/schema/Versioning.js";
import { int64Vec, floatVec } from "../../../src/onnx/utils/index.js";
import { inferNodeShape } from "../../../src/onnx/InferShapes.js";

describe("StandardOps Schema & Shape Inference", () => {
    initializeSchemaRegistry();

    function setup() {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);
        return { graph, builder };
    }

    it("infers shapes for Search ops (Where, TopK, NonZero, Hardmax, Compress, OneHot, Unique)", () => {
        const { graph, builder } = setup();

        const cond = graph
            .addNode("cond")
            .init(new TensorNode.Builder(DataType.BOOL, [2, 3], "input"))
            .as(TensorNode);
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);

        const [whereOut] = builder.createOp("Where", [cond, a, b]);
        expect(whereOut.shape).toEqual([2, 3]);

        const kConst = builder.createConstant("K", int64Vec([2]));
        const topkOuts = builder.createOp("TopK", [a, kConst], { axis: -1, largest: 1, sorted: 1 });
        expect(topkOuts.length).toBe(2);
        expect(topkOuts[0].shape).toEqual([2, 2]);

        const [nzOut] = builder.createOp("NonZero", [a]);
        expect(nzOut.shape[0]).toBe(2);

        const [hmOut] = builder.createOp("Hardmax", [a], { axis: -1 });
        expect(hmOut.shape).toEqual([2, 3]);

        // Compress
        const cond1D = graph
            .addNode("c1D")
            .init(new TensorNode.Builder(DataType.BOOL, [3], "input"))
            .as(TensorNode);
        const [compOut] = builder.createOp("Compress", [a, cond1D], { axis: 1 });
        expect(compOut.shape[0]).toBe(2);

        // OneHot
        const indices = graph
            .addNode("idx")
            .init(new TensorNode.Builder(DataType.INT64, [2, 3], "input"))
            .as(TensorNode);
        const depth = builder.createConstant("depth", int64Vec([5]));
        const values = builder.createConstant("vals", floatVec([0.0, 1.0]));

        const [oneHotOut] = builder.createOp("OneHot", [indices, depth, values], { axis: -1 });
        expect(oneHotOut.shape).toEqual([2, 3, 5]);

        const [oneHotDefault] = builder.createOp("OneHot", [indices, depth, values]);
        expect(oneHotDefault.shape).toEqual([2, 3, 5]);

        // Unique
        const [uniqOut] = builder.createOp("Unique", [a], { axis: 0 });
        expect(uniqOut.shape.length).toBeGreaterThan(0);
    });

    it("infers shapes for Generator ops (Range, ConstantOfShape, Random*, EyeLike, Multinomial)", () => {
        const { graph, builder } = setup();

        const start = builder.createConstant("start", int64Vec([0]));
        const limit = builder.createConstant("limit", int64Vec([10]));
        const delta = builder.createConstant("delta", int64Vec([2]));
        const [rangeOut] = builder.createOp("Range", [start, limit, delta]);
        expect(rangeOut.shape.length).toBe(1);

        const shapeConst = builder.createConstant("shape", int64Vec([3, 4]));
        const [cosOut] = builder.createOp("ConstantOfShape", [shapeConst]);
        expect(cosOut.shape).toEqual([3, 4]);

        const [rnOut] = builder.createOp("RandomNormal", [], { shape: [2, 5] });
        expect(rnOut.shape).toEqual([2, 5]);

        const [ruOut] = builder.createOp("RandomUniform", [], {
            shape: [3, 3],
            low: 0.0,
            high: 1.0,
        });
        expect(ruOut.shape).toEqual([3, 3]);

        const x = graph
            .addNode("genX")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 4], "input"))
            .as(TensorNode);
        const [rnlOut] = builder.createOp("RandomNormalLike", [x]);
        expect(rnlOut.shape).toEqual([4, 4]);

        const [rulOut] = builder.createOp("RandomUniformLike", [x]);
        expect(rulOut.shape).toEqual([4, 4]);

        const [eyeOut] = builder.createOp("EyeLike", [x], { k: 1 });
        expect(eyeOut.shape).toEqual([4, 4]);

        const multiIn = graph
            .addNode("multi_in")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 6], "input"))
            .as(TensorNode);
        const [multiOut] = builder.createOp("Multinomial", [multiIn], { sample_size: 4 });
        expect(multiOut.shape).toEqual([2, 4]);
    });

    it("infers shapes for Quantization ops (QuantizeLinear, DequantizeLinear, DynamicQuantizeLinear, QLinear*, ConvInteger, MatMulInteger)", () => {
        const { graph, builder } = setup();

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const scale = builder.createConstant("scale", int64Vec([1]));
        const zp = builder.createConstant("zp", int64Vec([0]));

        const [qOut] = builder.createOp("QuantizeLinear", [x, scale, zp], { axis: 1 });
        expect(qOut.shape).toEqual([2, 4]);

        const [dqOut] = builder.createOp("DequantizeLinear", [qOut, scale, zp], { axis: 1 });
        expect(dqOut.shape).toEqual([2, 4]);

        const dynOuts = builder.createOp("DynamicQuantizeLinear", [x]);
        expect(dynOuts.length).toBe(3);
        expect(dynOuts[0].shape).toEqual([2, 4]);

        // QLinearMatMul
        const a = graph
            .addNode("qA")
            .init(new TensorNode.Builder(DataType.UINT8, [2, 3], "input"))
            .as(TensorNode);
        const b = graph
            .addNode("qB")
            .init(new TensorNode.Builder(DataType.UINT8, [3, 4], "input"))
            .as(TensorNode);
        const sc = builder.createConstant("sc", int64Vec([1]));
        const [qmmOut] = builder.createOp("QLinearMatMul", [a, sc, sc, b, sc, sc, sc, sc]);
        expect(qmmOut.shape).toEqual([2, 4]);

        // MatMulInteger
        const [mmiOut] = builder.createOp("MatMulInteger", [a, b]);
        expect(mmiOut.shape).toEqual([2, 4]);

        const aZp = builder.createConstant("aZp", int64Vec([0]));
        const bZp = builder.createConstant("bZp", int64Vec([0]));
        const [mmiZpOut] = builder.createOp("MatMulInteger", [a, b, aZp, bZp]);
        expect(mmiZpOut.shape).toEqual([2, 4]);

        // ConvInteger
        const ciX = graph
            .addNode("ciX")
            .init(new TensorNode.Builder(DataType.UINT8, [1, 3, 16, 16], "input"))
            .as(TensorNode);
        const ciW = graph
            .addNode("ciW")
            .init(new TensorNode.Builder(DataType.INT8, [8, 3, 3, 3], "input"))
            .as(TensorNode);
        const [ciOut] = builder.createOp("ConvInteger", [ciX, ciW], {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
        });
        expect(ciOut.shape).toEqual([1, 3, 16, 16]);

        // QLinearConv
        const [qlcOut] = builder.createOp("QLinearConv", [ciX, sc, sc, ciW, sc, sc, sc, sc], {
            strides: [1, 1],
            pads: [0, 0, 0, 0],
        });
        expect(qlcOut.shape).toEqual([1, 8, 14, 14]);
    });

    it("infers shapes for RNN ops (RNN, LSTM, GRU)", () => {
        const { graph, builder } = setup();

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [5, 2, 8], "input"))
            .as(TensorNode);
        const w = graph
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 64, 8], "input"))
            .as(TensorNode);
        const r = graph
            .addNode("R")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 64, 16], "input"))
            .as(TensorNode);

        const lstmOuts = builder.createOp("LSTM", [x, w, r], { hidden_size: 16 });
        expect(lstmOuts.length).toBeGreaterThan(0);
        expect(lstmOuts[0].shape[0]).toBe(5);
        expect(lstmOuts[0].shape[1]).toBe(1);
        expect(lstmOuts[0].shape[2]).toBe(2);
        expect(lstmOuts[0].shape[3]).toBe(16);
    });

    it("infers shapes for NeuralNet ops (Conv, MaxPool, AveragePool, Global*, ConvTranspose, LRN, Dropout)", () => {
        const { graph, builder } = setup();

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 4, 32, 32], "input"))
            .as(TensorNode);
        const wConv = graph
            .addNode("wConv")
            .init(new TensorNode.Builder(DataType.FLOAT, [8, 2, 3, 3], "input"))
            .as(TensorNode);

        // Conv with group = 2 and auto_pad
        const [convGroup] = builder.createOp("Conv", [x, wConv], {
            group: 2,
            kernel_shape: [3, 3],
            auto_pad: "SAME_UPPER",
        });
        expect(convGroup.shape[0]).toBe(1);
        expect(convGroup.shape[1]).toBe(8);

        // MaxPool with ceil_mode and dual outputs
        const mpOuts = builder.createOp(
            "MaxPool",
            [x],
            { kernel_shape: [2, 2], strides: [2, 2], ceil_mode: 1 },
            [{ type: DataType.FLOAT }, { type: DataType.INT64 }],
        );
        expect(mpOuts.length).toBe(2);
        expect(mpOuts[0].shape).toEqual([1, 4, 16, 16]);

        // AveragePool with count_include_pad
        const [apOut] = builder.createOp("AveragePool", [x], {
            kernel_shape: [2, 2],
            strides: [2, 2],
            count_include_pad: 1,
        });
        expect(apOut.shape).toEqual([1, 4, 16, 16]);

        const [gapOut] = builder.createOp("GlobalAveragePool", [x]);
        expect(gapOut.shape).toEqual([1, 4, 1, 1]);

        const [gmpOut] = builder.createOp("GlobalMaxPool", [x]);
        expect(gmpOut.shape).toEqual([1, 4, 1, 1]);

        const [glpOut] = builder.createOp("GlobalLpPool", [x], { p: 2 });
        expect(glpOut.shape).toEqual([1, 4, 1, 1]);

        // ConvTranspose
        const wTrans = graph
            .addNode("wTrans")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 8, 3, 3], "input"))
            .as(TensorNode);
        const [ctOut] = builder.createOp("ConvTranspose", [x, wTrans], {
            strides: [1, 1],
            pads: [1, 1, 1, 1],
            output_padding: [0, 0],
        });
        expect(ctOut.shape[0]).toBe(1);
        expect(ctOut.shape[1]).toBe(8);

        const [lrnOut] = builder.createOp("LRN", [x], { size: 5 });
        expect(lrnOut.shape).toEqual([1, 4, 32, 32]);

        // Dropout
        const dropOuts = builder.createOp("Dropout", [x], { seed: 10 }, [
            { type: DataType.FLOAT },
            { type: DataType.BOOL },
        ]);
        expect(dropOuts[0].shape).toEqual([1, 4, 32, 32]);
    });

    it("infers shapes for Reduction ops with axes and keepdims", () => {
        const { graph, builder } = setup();

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3, 4], "input"))
            .as(TensorNode);

        const [argMax] = builder.createOp("ArgMax", [x], { axis: 1, keepdims: 0 });
        expect(argMax.shape).toEqual([2, 4]);

        const [argMin] = builder.createOp("ArgMin", [x], { axis: 2, keepdims: 1 });
        expect(argMin.shape).toEqual([2, 3, 1]);

        const axes = builder.createConstant("axes", int64Vec([0, 1]));
        const [sumOut] = builder.createOp("ReduceSum", [x, axes], { keepdims: 0 });
        expect(sumOut.shape).toEqual([4]);

        const [l2Out] = builder.createOp("ReduceL2", [x, axes], { keepdims: 1 });
        expect(l2Out.shape).toEqual([1, 1, 4]);
    });

    it("infers shapes for Activation and ControlFlow ops (If, Loop, Activations with attributes)", () => {
        const { graph, builder } = setup();

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"))
            .as(TensorNode);

        // Softmax with explicit and default axis fallback
        const [smOut] = builder.createOp("Softmax", [x], { axis: -1 });
        expect(smOut.shape).toEqual([2, 5]);
        const [smDef] = builder.createOp("Softmax", [x]);
        expect(smDef.shape).toEqual([2, 5]);

        const x1D = graph
            .addNode("x1D")
            .init(new TensorNode.Builder(DataType.FLOAT, [5], "input"))
            .as(TensorNode);
        const [sm1D] = builder.createOp("Softmax", [x1D]);
        expect(sm1D.shape).toEqual([5]);

        // LogSoftmax with explicit and default axis
        const [logSmOut] = builder.createOp("LogSoftmax", [x], { axis: 1 });
        expect(logSmOut.shape).toEqual([2, 5]);
        const [logSmDef] = builder.createOp("LogSoftmax", [x]);
        expect(logSmDef.shape).toEqual([2, 5]);

        // Gelu (covers lines 21-24)
        const [geluOut] = builder.createOp("Gelu", [x]);
        expect(geluOut.shape).toEqual([2, 5]);
        const [geluTanh] = builder.createOp("Gelu", [x], { approximate: "tanh" });
        expect(geluTanh.shape).toEqual([2, 5]);

        // PRelu (covers lines 120-127 with scalar and 1D broadcasted slopes)
        const slopeScalar = builder.createConstant("slope_scalar", floatVec([0.25]));
        const [prelu1] = builder.createOp("PRelu", [x, slopeScalar]);
        expect(prelu1.shape).toEqual([2, 5]);

        const slope1D = builder.createConstant("slope_1d", floatVec([0.1, 0.2, 0.3, 0.4, 0.5]));
        const [prelu2] = builder.createOp("PRelu", [x, slope1D]);
        expect(prelu2.shape).toEqual([2, 5]);

        // LeakyRelu with and without alpha
        const [lreluDef] = builder.createOp("LeakyRelu", [x]);
        expect(lreluDef.shape).toEqual([2, 5]);
        const [lreluAttr] = builder.createOp("LeakyRelu", [x], { alpha: 0.1 });
        expect(lreluAttr.shape).toEqual([2, 5]);

        // HardSigmoid & HardSwish (covers lines 156-161)
        const [hsDef] = builder.createOp("HardSigmoid", [x]);
        expect(hsDef.shape).toEqual([2, 5]);
        const [hsAttr] = builder.createOp("HardSigmoid", [x], { alpha: 0.2, beta: 0.5 });
        expect(hsAttr.shape).toEqual([2, 5]);
        const [hswish] = builder.createOp("HardSwish", [x]);
        expect(hswish.shape).toEqual([2, 5]);

        // Parameterized activations: ThresholdedRelu, Celu, Selu, Elu, Shrink
        const [thReluOut] = builder.createOp("ThresholdedRelu", [x], { alpha: 1.0 });
        expect(thReluOut.shape).toEqual([2, 5]);

        const [celuOut] = builder.createOp("Celu", [x], { alpha: 1.5 });
        expect(celuOut.shape).toEqual([2, 5]);

        const [seluOut] = builder.createOp("Selu", [x], { alpha: 1.67, gamma: 1.05 });
        expect(seluOut.shape).toEqual([2, 5]);

        const [eluOut] = builder.createOp("Elu", [x], { alpha: 1.0 });
        expect(eluOut.shape).toEqual([2, 5]);

        const [shrinkOut] = builder.createOp("Shrink", [x], { bias: 0.0, lambd: 0.5 });
        expect(shrinkOut.shape).toEqual([2, 5]);

        // Additional activations
        const [softplus] = builder.createOp("Softplus", [x]);
        expect(softplus.shape).toEqual([2, 5]);
        const [softsign] = builder.createOp("Softsign", [x]);
        expect(softsign.shape).toEqual([2, 5]);
        const [mish] = builder.createOp("Mish", [x]);
        expect(mish.shape).toEqual([2, 5]);
        const [fastGelu] = builder.createOp("FastGelu", [x]);
        expect(fastGelu.shape).toEqual([2, 5]);
        const [quickGelu] = builder.createOp("QuickGelu", [x], { alpha: 1.5 });
        expect(quickGelu.shape).toEqual([2, 5]);

        // ControlFlow: If
        const cond = graph
            .addNode("cond")
            .init(new TensorNode.Builder(DataType.BOOL, [], "input"))
            .as(TensorNode);

        const branchGraph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        branchGraph
            .addNode("out_val")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "output"));

        const ifOp = graph
            .addNode("if_op")
            .init(
                new OperationNode.Builder(
                    "If",
                    [cond],
                    { then_branch: branchGraph, else_branch: branchGraph },
                    [branchGraph, branchGraph],
                ),
            )
            .as(OperationNode);

        const ifOut = graph
            .addNode("if_out")
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);

        builder.addEdge(cond, ifOp, DataType.BOOL, []);
        builder.addEdge(ifOp, ifOut, DataType.FLOAT, []);
        inferNodeShape(ifOp, graph);
        expect(ifOut.shape).toEqual([2, 5]);

        // ControlFlow: Loop
        const loopBody = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        loopBody.addNode("b_iter").init(new TensorNode.Builder(DataType.INT64, [], "input"));
        loopBody.addNode("b_cond").init(new TensorNode.Builder(DataType.BOOL, [], "input"));
        loopBody.addNode("b_carry").init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"));
        loopBody.addNode("b_cond_out").init(new TensorNode.Builder(DataType.BOOL, [], "output"));
        loopBody
            .addNode("b_carry_out")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "output"));

        const loopOp = graph
            .addNode("loop_op")
            .init(new OperationNode.Builder("Loop", [cond, cond, x], {}, [loopBody]))
            .as(OperationNode);
        const loopOut = graph
            .addNode("loop_out")
            .init(new TensorNode.Builder(DataType.FLOAT, [], "intermediate"))
            .as(TensorNode);
        builder.addEdge(loopOp, loopOut, DataType.FLOAT, []);
        inferNodeShape(loopOp, graph);
        expect(loopOut.shape).toEqual([2, 5]);
    });

    it("resolves versioned schemas and infers shapes for Opset 21-28 operators", () => {
        const registry = OpRegistry.getInstance();

        // Multi-version resolution check
        expect(registry.get("QuantizeLinear", 20)?.sinceVersion).toBe(13);
        expect(registry.get("QuantizeLinear", 21)?.sinceVersion).toBe(21);
        expect(registry.get("QuantizeLinear", 28)?.sinceVersion).toBe(21);
        expect(registry.get("GroupNormalization", 19)?.sinceVersion).toBe(18);
        expect(registry.get("GroupNormalization", 21)?.sinceVersion).toBe(21);
        expect(registry.get("Cast", 20)?.sinceVersion).toBe(9);
        expect(registry.get("Cast", 28)?.sinceVersion).toBe(24);
        expect(registry.get("RMSNormalization", 22)).toBeUndefined();
        expect(registry.get("RMSNormalization", 23)?.sinceVersion).toBe(23);

        const { graph, builder } = setup();
        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 8, 16], "input"))
            .as(TensorNode);
        const rmsScale = graph
            .addNode("rms_scale")
            .init(new TensorNode.Builder(DataType.FLOAT, [16], "input"))
            .as(TensorNode);
        const [rmsOut] = builder.createOp("RMSNormalization", [x, rmsScale], {
            axis: -1,
            epsilon: 1e-6,
        });
        expect(rmsOut.shape).toEqual([2, 8, 16]);
        expect(rmsOut.literalType).toBe(DataType.FLOAT);

        // Swish (Opset 24)
        const [swishOut] = builder.createOp("Swish", [x], { alpha: 1.0 });
        expect(swishOut.shape).toEqual([2, 8, 16]);

        // RotaryEmbedding (Opset 23)
        const cosCache = graph
            .addNode("cos")
            .init(new TensorNode.Builder(DataType.FLOAT, [8, 8], "input"))
            .as(TensorNode);
        const sinCache = graph
            .addNode("sin")
            .init(new TensorNode.Builder(DataType.FLOAT, [8, 8], "input"))
            .as(TensorNode);
        const [ropeOut] = builder.createOp("RotaryEmbedding", [x, cosCache, sinCache]);
        expect(ropeOut.shape).toEqual([2, 8, 16]);

        // Attention (Opset 23/24) 4D GQA
        const q4 = graph
            .addNode("Q4")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4, 8, 16], "input"))
            .as(TensorNode);
        const k4 = graph
            .addNode("K4")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2, 12, 16], "input"))
            .as(TensorNode);
        const v4 = graph
            .addNode("V4")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2, 12, 32], "input"))
            .as(TensorNode);
        const attnOuts = builder.createOp("Attention", [q4, k4, v4], {
            q_num_heads: 4,
            kv_num_heads: 2,
            is_causal: 1,
        });
        expect(attnOuts[0].shape).toEqual([2, 4, 8, 32]);
        expect(attnOuts[1].shape).toEqual([2, 2, 12, 16]);
        expect(attnOuts[2].shape).toEqual([2, 2, 12, 32]);

        // TensorScatter (Opset 24)
        const cache = graph
            .addNode("cache")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2, 16, 16], "input"))
            .as(TensorNode);
        const upd = graph
            .addNode("upd")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2, 1, 16], "input"))
            .as(TensorNode);
        const [scatterOut] = builder.createOp("TensorScatter", [cache, upd], { axis: -2 });
        expect(scatterOut.shape).toEqual([2, 2, 16, 16]);

        // SpaceToDepth
        const img = graph
            .addNode("img")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 3, 8, 8], "input"))
            .as(TensorNode);
        const [s2dOut] = builder.createOp("SpaceToDepth", [img], { blocksize: 2 });
        expect(s2dOut.shape).toEqual([1, 12, 4, 4]);
    });

    it("computes minimal compatible export versions and honors explicit IR 14 / Opset 28 overrides", () => {
        // Baseline graph stays at compat (IR 9, Opset 20)
        const base = setup();
        const xBase = base.graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        base.builder.createOp("Relu", [xBase]);
        expect(resolveExportVersions(base.graph)).toEqual({ irVersion: 9, opsetVersion: 20 });

        // Graph with RMSNormalization auto-bumps to Opset 23 / IR 11
        const rmsG = setup();
        const xRms = rmsG.graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const sRms = rmsG.graph
            .addNode("S")
            .init(new TensorNode.Builder(DataType.FLOAT, [4], "input"))
            .as(TensorNode);
        rmsG.builder.createOp("RMSNormalization", [xRms, sRms]);
        expect(resolveExportVersions(rmsG.graph)).toEqual({ irVersion: 11, opsetVersion: 23 });

        // Explicit targetOpset 28 / irVersion 14 override
        expect(resolveExportVersions(base.graph, { targetOpset: 28, irVersion: 14 })).toEqual({
            irVersion: 14,
            opsetVersion: 28,
        });

        // IR 14 dtype triggers IR 14 automatically
        const fp6 = setup();
        fp6.graph.addNode("X6").init(new TensorNode.Builder(DataType.FLOAT6E2M3, [2, 4], "input"));
        expect(resolveExportVersions(fp6.graph).irVersion).toBe(14);
    });
});
