import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import OnnxEdge from "../../../src/Onnx/OnnxEdge.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { Recipes } from "../../../src/Onnx/transformation/recipes.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { int64Vec, makeTensorProto, scalarInt64 } from "../../../src/Onnx/Utils.js";
import { applyRecipe } from "../../helpers/recipeHarness.js";

describe("Canonicalization - Tier 1 & Tier 2 Extended Recipes", () => {
    initializeSchemaRegistry();

    it("decomposes DynamicQuantizeLinear (3 outputs)", () => {
        const { producedOps } = applyRecipe(
            new Recipes.canonicalization.DynamicQuantizeLinear(),
            "DynamicQuantizeLinear",
            [[2, 4]],
            {},
            DataType.FLOAT,
            ["Y", "Y_scale", "Y_zp"],
        );
        expect(producedOps).not.toContain("DynamicQuantizeLinear");
        expect(producedOps).toContain("QuantizeLinear");
    });

    it("decomposes Flatten, Split, Tile, EyeLike, OneHot, and Trilu", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.Flatten(), "Flatten", [[2, 3, 4]], { axis: 1 })
                .producedOps,
        ).toContain("Reshape");

        expect(
            applyRecipe(
                new Recipes.canonicalization.Split(),
                "Split",
                [[4, 4], { constant: int64Vec([2, 2]) }],
                { axis: 0 },
                DataType.FLOAT,
                ["Y0", "Y1"],
            ).producedOps,
        ).toContain("Slice");

        expect(
            applyRecipe(new Recipes.canonicalization.Tile(), "Tile", [
                [2, 3],
                { constant: int64Vec([2, 2]) },
            ]).producedOps,
        ).toContain("Expand");

        expect(
            applyRecipe(new Recipes.canonicalization.EyeLike(), "EyeLike", [[3, 4]], { k: 1 })
                .producedOps,
        ).not.toContain("EyeLike");

        expect(
            applyRecipe(new Recipes.canonicalization.OneHot(), "OneHot", [
                { shape: [2, 3], dataType: DataType.INT64 },
                { constant: scalarInt64(4) },
                { constant: makeTensorProto(DataType.FLOAT, [2], [0, 1]) },
            ]).producedOps,
        ).not.toContain("OneHot");

        expect(
            applyRecipe(
                new Recipes.canonicalization.Trilu(),
                "Trilu",
                [[3, 3], { constant: scalarInt64(0) }],
                { upper: 1 },
            ).producedOps,
        ).not.toContain("Trilu");
    });

    it("decomposes LogSoftmax, Hardmax, and LpNormalization", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.LogSoftmax(), "LogSoftmax", [[2, 4]], {
                axis: -1,
            }).producedOps,
        ).not.toContain("LogSoftmax");
        expect(
            applyRecipe(new Recipes.canonicalization.Hardmax(), "Hardmax", [[2, 4]], { axis: -1 })
                .producedOps,
        ).toContain("ArgMax");
        expect(
            applyRecipe(
                new Recipes.canonicalization.LpNormalization(),
                "LpNormalization",
                [[2, 4]],
                {
                    p: 2,
                    axis: -1,
                },
            ).producedOps,
        ).toContain("ReduceL2");
    });

    it("decomposes GlobalAveragePool, GlobalMaxPool, GlobalLpPool, and LpPool", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.GlobalAveragePool(), "GlobalAveragePool", [
                [1, 3, 4, 4],
            ]).producedOps,
        ).toContain("ReduceMean");
        expect(
            applyRecipe(new Recipes.canonicalization.GlobalMaxPool(), "GlobalMaxPool", [
                [1, 3, 4, 4],
            ]).producedOps,
        ).toContain("ReduceMax");
        expect(
            applyRecipe(
                new Recipes.canonicalization.GlobalLpPool(),
                "GlobalLpPool",
                [[1, 3, 4, 4]],
                { p: 2 },
            ).producedOps,
        ).toContain("ReduceL2");
        expect(
            applyRecipe(new Recipes.canonicalization.LpPool(), "LpPool", [[1, 2, 4, 4]], {
                kernel_shape: [2, 2],
                strides: [2, 2],
                p: 2,
            }).producedOps,
        ).toContain("Conv");
    });

    it("decomposes Quantized MatMul & Conv (MatMulInteger, QLinearMatMul, ConvInteger, QLinearConv)", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.MatMulInteger(), "MatMulInteger", [
                { shape: [2, 4], dataType: DataType.UINT8 },
                { shape: [4, 3], dataType: DataType.UINT8 },
            ]).producedOps,
        ).toContain("MatMul");

        expect(
            applyRecipe(new Recipes.canonicalization.QLinearMatMul(), "QLinearMatMul", [
                { shape: [2, 4], dataType: DataType.UINT8 },
                { shape: [], dataType: DataType.FLOAT },
                { shape: [], dataType: DataType.UINT8 },
                { shape: [4, 3], dataType: DataType.UINT8 },
                { shape: [], dataType: DataType.FLOAT },
                { shape: [], dataType: DataType.UINT8 },
                { shape: [], dataType: DataType.FLOAT },
                { shape: [], dataType: DataType.UINT8 },
            ]).producedOps,
        ).toContain("MatMul");

        expect(
            applyRecipe(
                new Recipes.canonicalization.ConvInteger(),
                "ConvInteger",
                [
                    { shape: [1, 2, 4, 4], dataType: DataType.UINT8 },
                    { shape: [3, 2, 3, 3], dataType: DataType.UINT8 },
                ],
                { kernel_shape: [3, 3] },
            ).producedOps,
        ).toContain("Conv");

        expect(
            applyRecipe(
                new Recipes.canonicalization.QLinearConv(),
                "QLinearConv",
                [
                    { shape: [1, 2, 4, 4], dataType: DataType.UINT8 },
                    { shape: [], dataType: DataType.FLOAT },
                    { shape: [], dataType: DataType.UINT8 },
                    { shape: [3, 2, 3, 3], dataType: DataType.UINT8 },
                    { shape: [], dataType: DataType.FLOAT },
                    { shape: [], dataType: DataType.UINT8 },
                    { shape: [], dataType: DataType.FLOAT },
                    { shape: [], dataType: DataType.UINT8 },
                ],
                { kernel_shape: [3, 3] },
            ).producedOps,
        ).toContain("Conv");
    });

    it("decomposes NegativeLogLikelihoodLoss and SoftmaxCrossEntropyLoss", () => {
        expect(
            applyRecipe(
                new Recipes.canonicalization.NegativeLogLikelihoodLoss(),
                "NegativeLogLikelihoodLoss",
                [
                    { shape: [4, 5], dataType: DataType.FLOAT },
                    { shape: [4], dataType: DataType.INT64 },
                ],
                { reduction: "mean" },
            ).producedOps,
        ).not.toContain("NegativeLogLikelihoodLoss");

        expect(
            applyRecipe(
                new Recipes.canonicalization.SoftmaxCrossEntropyLoss(),
                "SoftmaxCrossEntropyLoss",
                [
                    { shape: [4, 5], dataType: DataType.FLOAT },
                    { shape: [4], dataType: DataType.INT64 },
                ],
                { reduction: "mean" },
                DataType.FLOAT,
                ["loss", "log_prob"],
            ).producedOps,
        ).toContain("LogSoftmax");
    });

    it("decomposes RNN, GRU, and LSTM", () => {
        expect(
            applyRecipe(
                new Recipes.canonicalization.RNN(),
                "RNN",
                [
                    [3, 2, 4],
                    [1, 5, 4],
                    [1, 5, 5],
                ],
                {},
                DataType.FLOAT,
                ["Y", "Y_h"],
            ).producedOps,
        ).toContain("MatMul");

        expect(
            applyRecipe(
                new Recipes.canonicalization.GRU(),
                "GRU",
                [
                    [3, 2, 4],
                    [1, 15, 4],
                    [1, 15, 5],
                ],
                {},
                DataType.FLOAT,
                ["Y", "Y_h"],
            ).producedOps,
        ).toContain("Sigmoid");

        expect(
            applyRecipe(
                new Recipes.canonicalization.LSTM(),
                "LSTM",
                [
                    [3, 2, 4],
                    [1, 20, 4],
                    [1, 20, 5],
                ],
                {},
                DataType.FLOAT,
                ["Y", "Y_h", "Y_c"],
            ).producedOps,
        ).toContain("Tanh");
    });

    it("decomposes Scan into a native ONNX Loop", () => {
        // Build body: (state_in, x_in) -> (state_out = state_in + x_in, y_out = state_out)
        const body = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bodyBuilder = new GraphBuilder(body, "scan_body");
        const sIn = body
            .addNode("s_in")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const xIn = body
            .addNode("x_in")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const [sum] = bodyBuilder.createOp("Add", [sIn, xIn], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        const [sOut] = bodyBuilder.createOp("Identity", [sum], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        const [yOut] = bodyBuilder.createOp("Identity", [sum], {}, [
            { type: DataType.FLOAT, shape: [2] },
        ]);
        body.getNodeById(sOut.id)!.as(TensorNode).setType("output");
        body.getNodeById(yOut.id)!.as(TensorNode).setType("output");

        const outer = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const outerBuilder = new GraphBuilder(outer, "outer");
        const initState = outer
            .addNode("init_s")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const scanIn = outer
            .addNode("scan_x")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
            .as(TensorNode);

        const scanOp = outer
            .addNode("scan_0")
            .init(
                new OperationNode.Builder("Scan", [initState, scanIn], { num_scan_inputs: 1 }, [
                    body,
                ]),
            )
            .as(OperationNode);
        outer.addEdge(initState, scanOp).init(new OnnxEdge.Builder(DataType.FLOAT, [2]));
        outer.addEdge(scanIn, scanOp).init(new OnnxEdge.Builder(DataType.FLOAT, [3, 2]));

        const outS = outer
            .addNode("final_s")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "output"))
            .as(TensorNode);
        const outY = outer
            .addNode("scan_y")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "output"))
            .as(TensorNode);
        outer.addEdge(scanOp, outS).init(new OnnxEdge.Builder(DataType.FLOAT, [2]));
        outer.addEdge(scanOp, outY).init(new OnnxEdge.Builder(DataType.FLOAT, [3, 2]));

        const recipe = new Recipes.canonicalization.Scan();
        expect(recipe.match(scanOp)).toBe(true);
        recipe.apply(scanOp, outerBuilder);

        const ops = outer
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(ops).not.toContain("Scan");
        expect(ops).toContain("Loop");
    });

    it("decomposes ArgMax and ArgMin (first & last tie-breaking, keepdims 0 & 1)", () => {
        expect(
            applyRecipe(new Recipes.canonicalization.ArgMax(), "ArgMax", [[2, 4]], {
                axis: -1,
                keepdims: 1,
                select_last_index: 0,
            }).producedOps,
        ).not.toContain("ArgMax");

        expect(
            applyRecipe(new Recipes.canonicalization.ArgMin(), "ArgMin", [[2, 4]], {
                axis: 0,
                keepdims: 0,
                select_last_index: 1,
            }).producedOps,
        ).not.toContain("ArgMin");
    });

    it("decomposes TopK, Einsum, and Det", () => {
        expect(
            applyRecipe(
                new Recipes.canonicalization.Einsum(),
                "Einsum",
                [
                    [2, 3],
                    [3, 4],
                ],
                { equation: "ij,jk->ik" },
            ).producedOps,
        ).not.toContain("Einsum");

        expect(
            applyRecipe(new Recipes.canonicalization.Det(), "Det", [[3, 3]], {}).producedOps,
        ).not.toContain("Det");
    });

    it("decomposes dynamic / layout=1 / sequence_lens RNN, GRU, and LSTM via Loop recipes", () => {
        // Dynamic seq_len & batch_size RNN
        const rnnRes = applyRecipe(
            new Recipes.canonicalization.RNNLoop(),
            "RNN",
            [
                [-1, -1, 4],
                [1, 5, 4],
                [1, 5, 5],
            ],
            { hidden_size: 5 },
            DataType.FLOAT,
            ["Y", "Y_h"],
        );
        expect(rnnRes.producedOps).not.toContain("RNN");
        expect(rnnRes.producedOps).toContain("Loop");

        // layout=1 (batch_first) bidirectional GRU
        const gruRes = applyRecipe(
            new Recipes.canonicalization.GRULoop(),
            "GRU",
            [
                [2, 3, 4],
                [2, 15, 4],
                [2, 15, 5],
            ],
            { hidden_size: 5, layout: 1, direction: "bidirectional" },
            DataType.FLOAT,
            ["Y", "Y_h"],
        );
        expect(gruRes.producedOps).not.toContain("GRU");
        expect(gruRes.producedOps).toContain("Loop");

        // sequence_lens LSTM
        const lstmRes = applyRecipe(
            new Recipes.canonicalization.LSTMLoop(),
            "LSTM",
            [[4, 2, 3], [1, 16, 3], [1, 16, 4], [1, 32], { shape: [2], dataType: DataType.INT32 }],
            { hidden_size: 4 },
            DataType.FLOAT,
            ["Y", "Y_h", "Y_c"],
        );
        expect(lstmRes.producedOps).not.toContain("LSTM");
        expect(lstmRes.producedOps).toContain("Loop");
    });
    it("decomposes dynamic Split via LowerSplitDynamicRecipe (runtime split & dynamic axis)", () => {
        // Runtime split tensor
        const res1 = applyRecipe(
            new Recipes.canonicalization.SplitDynamic(),
            "Split",
            [[4, 3], { shape: [2], dataType: DataType.INT64 }],
            { axis: 0 },
            DataType.FLOAT,
            ["Y0", "Y1"],
        );
        expect(res1.producedOps).not.toContain("Split");
        expect(res1.producedOps).toContain("CumSum");
        expect(res1.producedOps).toContain("Slice");

        // Dynamic axis equal split (no split tensor)
        const res2 = applyRecipe(
            new Recipes.canonicalization.SplitDynamic(),
            "Split",
            [[-1, 3]],
            { axis: 0, num_outputs: 2 },
            DataType.FLOAT,
            ["Y0", "Y1"],
        );
        expect(res2.producedOps).not.toContain("Split");
        expect(res2.producedOps).toContain("Slice");
    });

    it("decomposes dynamic ArgMax, ArgMin, and TopK via Loop recipes", () => {
        const argMaxRes = applyRecipe(
            new Recipes.canonicalization.ArgMaxLoop(),
            "ArgMax",
            [[2, -1]],
            { axis: 1, keepdims: 1, select_last_index: 0 },
        );
        expect(argMaxRes.producedOps).not.toContain("ArgMax");
        expect(argMaxRes.producedOps).toContain("Loop");

        const argMinRes = applyRecipe(new Recipes.canonicalization.ArgMinLoop(), "ArgMin", [[-1]], {
            axis: 0,
            keepdims: 0,
            select_last_index: 1,
        });
        expect(argMinRes.producedOps).not.toContain("ArgMin");
        expect(argMinRes.producedOps).toContain("Loop");

        const topKRes = applyRecipe(
            new Recipes.canonicalization.TopKLoop(),
            "TopK",
            [[2, -1], { shape: [1], dataType: DataType.INT64 }],
            { axis: 1, largest: 1, sorted: 1 },
            DataType.FLOAT,
            ["Values", "Indices"],
        );
        expect(topKRes.producedOps).not.toContain("TopK");
        expect(topKRes.producedOps).toContain("Loop");
    });
    it("decomposes 2-output MaxPool (Y + Indices) with row-major and col-major storage_order", () => {
        for (const storageOrder of [0, 1]) {
            const res = applyRecipe(
                new Recipes.canonicalization.MaxPoolWithIndices(),
                "MaxPool",
                [[1, 2, 4, 4]],
                { kernel_shape: [2, 2], strides: [2, 2], storage_order: storageOrder },
                DataType.FLOAT,
                ["Y", "Indices"],
            );
            expect(res.producedOps).not.toContain("MaxPool");
            expect(res.producedOps).toContain("Loop");
        }
    });

    it("decomposes general-p LpNormalization and Opset 21 extended/blocked QDQ", () => {
        const lpRes = applyRecipe(
            new Recipes.canonicalization.LpNormalizationGeneral(),
            "LpNormalization",
            [[2, 4]],
            { axis: 1, p: 3 },
        );
        expect(lpRes.producedOps).not.toContain("LpNormalization");
        expect(lpRes.producedOps).toContain("Pow");
        expect(lpRes.producedOps).toContain("ReduceSum");

        const qInt16 = applyRecipe(
            new Recipes.canonicalization.QuantizeLinearExtended(),
            "QuantizeLinear",
            [[2, 4], [], { shape: [], dataType: DataType.INT16 }],
            { axis: 1, block_size: 0 },
            DataType.FLOAT,
            ["Y"],
        );
        expect(qInt16.producedOps).not.toContain("QuantizeLinear");
        expect(qInt16.producedOps).toContain("Clip");
        expect(qInt16.producedOps).toContain("Round");

        const qBlocked = applyRecipe(
            new Recipes.canonicalization.QuantizeLinearExtended(),
            "QuantizeLinear",
            [[2, 6], [2, 3], { shape: [2, 3], dataType: DataType.INT8 }],
            { axis: 1, block_size: 2 },
            DataType.FLOAT,
            ["Y"],
        );
        expect(qBlocked.producedOps).not.toContain("QuantizeLinear");
        expect(qBlocked.producedOps).toContain("Reshape");
        expect(qBlocked.producedOps).toContain("Unsqueeze");

        const dqBlocked = applyRecipe(
            new Recipes.canonicalization.DequantizeLinearBlocked(),
            "DequantizeLinear",
            [
                { shape: [2, 6], dataType: DataType.INT8 },
                [2, 3],
                { shape: [2, 3], dataType: DataType.INT8 },
            ],
            { axis: 1, block_size: 2 },
            DataType.FLOAT,
            ["Y"],
        );
        expect(dqBlocked.producedOps).not.toContain("DequantizeLinear");
        expect(dqBlocked.producedOps).toContain("Reshape");
        expect(dqBlocked.producedOps).toContain("Mul");
    });

    it("decomposes dynamic Hardmax, Trilu, EyeLike, and dynamic-batch Det", () => {
        const hm = applyRecipe(
            new Recipes.canonicalization.HardmaxDynamic(),
            "Hardmax",
            [[2, -1]],
            { axis: 1 },
        );
        expect(hm.producedOps).not.toContain("Hardmax");
        expect(hm.producedOps).toContain("ArgMax");

        const tr = applyRecipe(
            new Recipes.canonicalization.TriluDynamic(),
            "Trilu",
            [[-1, -1, -1]],
            { upper: 0 },
        );
        expect(tr.producedOps).not.toContain("Trilu");
        expect(tr.producedOps).toContain("LessOrEqual");

        const eye = applyRecipe(
            new Recipes.canonicalization.EyeLikeDynamic(),
            "EyeLike",
            [[-1, -1]],
            { k: 1 },
        );
        expect(eye.producedOps).not.toContain("EyeLike");
        expect(eye.producedOps).toContain("Equal");

        const det = applyRecipe(new Recipes.canonicalization.DetDynamic(), "Det", [[-1, 4, 4]]);
        expect(det.producedOps).not.toContain("Det");
        expect(det.producedOps).toContain("ConstantOfShape");
    });

    it("decomposes dynamic-shape Einsum and 2-output MaxPool", () => {
        const ein = applyRecipe(
            new Recipes.canonicalization.Einsum(),
            "Einsum",
            [
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            { equation: "bij,bjk->bik" },
        );
        expect(ein.producedOps).not.toContain("Einsum");
        expect(ein.producedOps).toContain("MatMul");

        const einDiag = applyRecipe(
            new Recipes.canonicalization.Einsum(),
            "Einsum",
            [
                [-1, -1, -1],
                [-1, -1],
            ],
            { equation: "iij,jk->ik" },
        );
        expect(einDiag.producedOps).not.toContain("Einsum");
        expect(einDiag.producedOps).toContain("Range");

        const mpIdx = applyRecipe(
            new Recipes.canonicalization.MaxPoolWithIndices(),
            "MaxPool",
            [[-1, 2, -1, -1]],
            { kernel_shape: [2, 2], strides: [2, 2], storage_order: 1 },
            DataType.FLOAT,
            ["Y", "Indices"],
        );
        expect(mpIdx.producedOps).not.toContain("MaxPool");
        expect(mpIdx.producedOps).toContain("Loop");
        expect(mpIdx.producedOps).toContain("Expand");
    });
});
