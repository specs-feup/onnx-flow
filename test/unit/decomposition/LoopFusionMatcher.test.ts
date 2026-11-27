import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";
import { LoopFusionMatcher } from "../../../src/Onnx/transformation/loop-lowering/LoopFusionMatcher.js";
import { LoopLoweringPass } from "../../../src/Onnx/transformation/loop-lowering/LoopLoweringPass.js";
import { LowerSoftmaxLoopRecipe } from "../../../src/Onnx/transformation/loop-lowering/recipes/LowerSoftmaxLoopRecipe.js";
import { LowerLayerNormLoopRecipe } from "../../../src/Onnx/transformation/loop-lowering/recipes/LowerLayerNormLoopRecipe.js";
import { LowerLogSoftmaxLoopRecipe } from "../../../src/Onnx/transformation/loop-lowering/recipes/LowerLogSoftmaxLoopRecipe.js";
import { LowerRMSNormLoopRecipe } from "../../../src/Onnx/transformation/loop-lowering/recipes/LowerRMSNormLoopRecipe.js";
import { LowerDetLoopRecipe } from "../../../src/Onnx/transformation/loop-lowering/recipes/LowerDetLoopRecipe.js";
import { LowerTopKScalarLoopRecipe } from "../../../src/Onnx/transformation/loop-lowering/recipes/LowerTopKScalarLoopRecipe.js";
import OnnxGraphTransformer from "../../../src/Onnx/transformation/index.js";
import { Schedule } from "../../../src/Onnx/transformation/Schedule.js";
import { LowerMinRecipe } from "../../../src/Onnx/transformation/canonicalization/recipes/LowerMinRecipe.js";
import { LowerMaxRecipe } from "../../../src/Onnx/transformation/canonicalization/recipes/LowerMaxRecipe.js";
import { HistoryManager } from "../../../src/Onnx/transformation/tracking/HistoryManager.js";
import { floatVec, scalarFloat, scalarInt64 } from "../../../src/Onnx/Utils.js";

describe("LoopFusionMatcher", () => {
    initializeSchemaRegistry();

    function buildChain(shape: number[] = [2, 3]) {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph, "fuse");
        const a = graph
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, shape, "input"))
            .as(TensorNode);
        const b = graph
            .addNode("B")
            .init(new TensorNode.Builder(DataType.FLOAT, shape, "input"))
            .as(TensorNode);
        const [addOut] = builder.createOp("Add", [a, b], {}, [{ type: DataType.FLOAT, shape }]);
        const [reluOut] = builder.createOp("Relu", [addOut], {}, [{ type: DataType.FLOAT, shape }]);
        if (reluOut.is(TensorNode)) reluOut.as(TensorNode).setType("output");
        return { graph, builder, a, b, addOut, reluOut };
    }

    it("fuses producer and elementwise consumer into a single opportunity when fuse=true", () => {
        const { graph } = buildChain();
        const matcher = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        });

        const opps = matcher.findOpportunities(graph);
        expect(opps.length).toBe(1);
        expect(opps[0].targetNodeId.split(",").length).toBe(2);
    });

    it("emits separate single-op opportunities when fuse=false", () => {
        const { graph } = buildChain();
        const matcher = new LoopFusionMatcher({
            coalesce: true,
            fuse: false,
            indexingMode: IndexingMode.Pointer,
        });

        const opps = matcher.findOpportunities(graph);
        expect(opps.length).toBe(2);
        expect(opps.every((o) => !o.targetNodeId.includes(","))).toBe(true);
    });

    it("stops fusion across multi-consumer forks and self-carry ops (CumSum)", () => {
        // 1. Fork: Add -> Relu and Add -> Neg (2 consumers of addOut -> cannot fuse)
        const { graph, builder, addOut } = buildChain();
        const [negOut] = builder.createOp("Neg", [addOut], {}, [
            { type: DataType.FLOAT, shape: [2, 3] },
        ]);
        if (negOut.is(TensorNode)) negOut.as(TensorNode).setType("output");

        const forkOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(graph);
        expect(forkOpps.length).toBe(3);

        // 2. Self-carry barrier: CumSum -> Relu cannot fuse
        const gCs = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bCs = new GraphBuilder(gCs, "cs");
        const x = gCs
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const ax = bCs.createConstant("ax", scalarInt64(1));
        const [csOut] = bCs.createOp("CumSum", [x, ax], { exclusive: 0, reverse: 0 }, [
            { type: DataType.FLOAT, shape: [2, 4] },
        ]);
        const [csRelu] = bCs.createOp("Relu", [csOut], {}, [
            { type: DataType.FLOAT, shape: [2, 4] },
        ]);
        if (csRelu.is(TensorNode)) csRelu.as(TensorNode).setType("output");

        const csOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(gCs);
        expect(csOpps.length).toBe(2);
    });

    it("selects CoalescedMatMul vs standard MatMul recipe based on coalesce flag", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "mm");
        const a = g
            .addNode("A")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const w = g
            .addNode("W")
            .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
            .as(TensorNode);
        b.createOp("MatMul", [a, w], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);

        for (const coalesce of [true, false]) {
            const clone = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const cb = new GraphBuilder(clone, "mm");
            const ca = clone
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
                .as(TensorNode);
            const cw = clone
                .addNode("W")
                .init(new TensorNode.Builder(DataType.FLOAT, [3, 2], "input"))
                .as(TensorNode);
            cb.createOp("MatMul", [ca, cw], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);

            const opps = new LoopFusionMatcher({
                coalesce,
                fuse: false,
                indexingMode: IndexingMode.Pointer,
            }).findOpportunities(clone);
            expect(opps.length).toBe(1);
            opps[0].apply(cb);
            const loopOp = clone
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Loop")!;
            const innerOps = loopOp.regions[0]
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            if (coalesce) {
                // Coalesced scalar MAC uses accumulator carries rather than vector ReduceSum
                expect(innerOps).not.toContain("ReduceSum");
            } else {
                expect(innerOps).toContain("ReduceSum");
            }
        }
        expect(g.getOperationNodes().length).toBe(1);
    });

    it("fuses convex single-output diamond DAGs into 1 Loop (including dynamic shapes)", () => {
        // Static diamond: X -> {Relu, Neg} -> Add
        for (const shape of [
            [2, 3],
            [-1, 4],
        ]) {
            const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const b = new GraphBuilder(g, "diam");
            const x = g
                .addNode("X")
                .init(new TensorNode.Builder(DataType.FLOAT, shape, "input"))
                .as(TensorNode);
            const [r] = b.createOp("Relu", [x], {}, [{ type: DataType.FLOAT, shape }]);
            const [n] = b.createOp("Neg", [x], {}, [{ type: DataType.FLOAT, shape }]);
            const [out] = b.createOp("Add", [r, n], {}, [{ type: DataType.FLOAT, shape }]);
            if (out.is(TensorNode)) out.as(TensorNode).setType("output");

            const opps = new LoopFusionMatcher({
                coalesce: true,
                fuse: true,
                indexingMode: IndexingMode.Pointer,
            }).findOpportunities(g);
            expect(opps.length).toBe(1);
            expect(opps[0].targetNodeId.split(",").length).toBe(3);

            opps[0].apply(b);
            const topOps = g
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            expect(topOps.filter((t) => t === "Loop").length).toBe(1);
            expect(topOps).not.toContain("Relu");
            expect(topOps).not.toContain("Neg");
            expect(topOps).not.toContain("Add");
        }
    });

    it("emits scan_outputs without ScatterElements when useScanOutputs=true, preserving carry default when false", () => {
        for (const useScanOutputs of [false, true]) {
            const { graph } = buildChain([2, 3]);
            const history = new HistoryManager(graph);
            new LoopLoweringPass({
                coalesce: true,
                fuse: true,
                indexingMode: IndexingMode.Pointer,
                useScanOutputs,
            }).run(graph, history);

            const loopOp = graph
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Loop")!;
            expect(loopOp).toBeDefined();
            const innerOps = loopOp.regions[0]
                .getOperationNodes()
                .toArray()
                .map((o) => o.type);
            if (useScanOutputs) {
                expect(innerOps).not.toContain("ScatterElements");
            } else {
                expect(innerOps).toContain("ScatterElements");
            }
        }
    });

    it("uses 0D scalar accumulator carries without carry Gather for trailing-axis ReduceSum", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "redacc");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const ax = b.createConstant("ax", scalarInt64(1));
        const [out] = b.createOp("ReduceSum", [x, ax], { keepdims: 1 }, [
            { type: DataType.FLOAT, shape: [2, 1] },
        ]);
        if (out.is(TensorNode)) out.as(TensorNode).setType("output");

        const history = new HistoryManager(g);
        new LoopLoweringPass({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).run(g, history);

        const loopOp = g
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Loop")!;
        const innerOps = loopOp.regions[0].getOperationNodes().toArray();
        const gathers = innerOps.filter((o) => o.type === "Gather");
        // Only the input scalar gather remains; accumulator read uses scalar carry directly
        expect(gathers.length).toBe(1);
        expect(loopOp.getInputs()!.length).toBeGreaterThanOrEqual(4);
    });

    it("lowers Softmax and LayerNormalization into fused nested Loops via LoopLowering recipes", () => {
        const gSm = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bSm = new GraphBuilder(gSm, "sm");
        const xSm = gSm
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
            .as(TensorNode);
        const [ySm] = bSm.createOp("Softmax", [xSm], { axis: -1 }, [
            { type: DataType.FLOAT, shape: [-1, 4] },
        ]);
        if (ySm.is(TensorNode)) ySm.as(TensorNode).setType("output");

        const smRecipe = new LowerSoftmaxLoopRecipe();
        const smOp = gSm.getOperationNodes().toArray()[0];
        expect(smRecipe.match(smOp)).toBe(true);
        const smOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(gSm);
        expect(smOpps.length).toBe(1);
        expect(smOpps[0].recipeName).toBe("LoopLowering");
        expect(smOpps[0].description).toContain("Softmax");
        smOpps[0].apply(bSm);
        const outerSmLoops = gSm
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(outerSmLoops.length).toBe(1);
        const innerSmOps = outerSmLoops[0].regions[0]
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(innerSmOps).toContain("Loop");
        expect(innerSmOps).not.toContain("ReduceMax");

        const gLn = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bLn = new GraphBuilder(gLn, "ln");
        const xLn = gLn
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
            .as(TensorNode);
        const scLn = bLn.createConstant("S", floatVec([1, 1, 1, 1]));
        const bLnC = bLn.createConstant("B", floatVec([0, 0, 0, 0]));
        const [yLn] = bLn.createOp("LayerNormalization", [xLn, scLn, bLnC], { axis: -1 }, [
            { type: DataType.FLOAT, shape: [-1, 4] },
        ]);
        if (yLn.is(TensorNode)) yLn.as(TensorNode).setType("output");

        const lnRecipe = new LowerLayerNormLoopRecipe();
        const lnOp = gLn.getOperationNodes().toArray()[0];
        expect(lnRecipe.match(lnOp)).toBe(true);
        const lnOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(gLn);
        expect(lnOpps.length).toBe(1);
        expect(lnOpps[0].recipeName).toBe("LoopLowering");
        expect(lnOpps[0].description).toContain("LayerNormalization");
        lnOpps[0].apply(bLn);
        const outerLnLoops = gLn
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(outerLnLoops.length).toBe(1);
        const innerLnOps = outerLnLoops[0].regions[0]
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(innerLnOps).toContain("Loop");
        expect(innerLnOps).not.toContain("ReduceMean");
        void scalarFloat;
    });

    it("lowers LogSoftmax, RMSNormalization, Det (N>=4), and TopK via nested standalone loop recipes and defers AveragePool without outer Div", () => {
        // 1. LowerLogSoftmaxLoop & LowerRMSNormLoop matching and structure
        const gLsm = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bLsm = new GraphBuilder(gLsm, "lsm");
        const xLsm = gLsm
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
            .as(TensorNode);
        const [yLsm] = bLsm.createOp("LogSoftmax", [xLsm], { axis: -1 }, [
            { type: DataType.FLOAT, shape: [-1, 4] },
        ]);
        if (yLsm.is(TensorNode)) yLsm.as(TensorNode).setType("output");
        const lsmRecipe = new LowerLogSoftmaxLoopRecipe();
        expect(lsmRecipe.match(gLsm.getOperationNodes().toArray()[0])).toBe(true);
        const lsmOpps = new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        }).findOpportunities(gLsm);
        lsmOpps[0].apply(bLsm);
        const lsmLoops = gLsm
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(lsmLoops.length).toBe(1);
        expect(
            lsmLoops[0].regions[0]
                .getOperationNodes()
                .toArray()
                .map((o) => o.type),
        ).toContain("Log");

        const gRms = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bRms = new GraphBuilder(gRms, "rms");
        const xRms = gRms
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4], "input"))
            .as(TensorNode);
        const sRms = bRms.createConstant("S", floatVec([1, 1, 1, 1]));
        const [yRms] = bRms.createOp("RMSNormalization", [xRms, sRms], { axis: -1 }, [
            { type: DataType.FLOAT, shape: [-1, 4] },
        ]);
        if (yRms.is(TensorNode)) yRms.as(TensorNode).setType("output");
        const rmsRecipe = new LowerRMSNormLoopRecipe();
        expect(rmsRecipe.match(gRms.getOperationNodes().toArray()[0])).toBe(true);
        new LoopFusionMatcher({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
        })
            .findOpportunities(gRms)[0]
            .apply(bRms);
        expect(
            gRms
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop").length,
        ).toBe(1);

        // 2. Det (4x4) and TopK defer to LoopLowering (O(1) Loops) in full pipeline
        const gDet = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bDet = new GraphBuilder(gDet, "detl");
        const xDet = gDet
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [-1, 4, 4], "input"))
            .as(TensorNode);
        const [yDet] = bDet.createOp("Det", [xDet], {}, [{ type: DataType.FLOAT, shape: [-1] }]);
        if (yDet.is(TensorNode)) yDet.as(TensorNode).setType("output");
        expect(new LowerDetLoopRecipe().match(gDet.getOperationNodes().toArray()[0])).toBe(true);
        const outDet = new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            optimize: true,
        }).apply(gDet, false);
        expect(
            outDet
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop").length,
        ).toBe(1);

        const gTopK = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bTopK = new GraphBuilder(gTopK, "topkl");
        const xTopK = gTopK
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 5], "input"))
            .as(TensorNode);
        const kTopK = bTopK.createConstant("K", scalarInt64(3));
        const [vTopK, iTopK] = bTopK.createOp("TopK", [xTopK, kTopK], { axis: -1, largest: 1 }, [
            { type: DataType.FLOAT, shape: [2, 3] },
            { type: DataType.INT64, shape: [2, 3] },
        ]);
        if (vTopK.is(TensorNode)) vTopK.as(TensorNode).setType("output");
        if (iTopK.is(TensorNode)) iTopK.as(TensorNode).setType("output");
        expect(new LowerTopKScalarLoopRecipe().match(gTopK.getOperationNodes().toArray()[0])).toBe(
            true,
        );
        const outTopK = new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            optimize: true,
        }).apply(gTopK, false);
        const topKTopOps = outTopK
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(topKTopOps.filter((t) => t === "Loop").length).toBe(1);
        expect(topKTopOps).not.toContain("ArgMax");

        // 3. AveragePool defers to single Loop without outer Div or Conv
        const gAp = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bAp = new GraphBuilder(gAp, "ap");
        const xAp = gAp
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [1, 2, 8, 8], "input"))
            .as(TensorNode);
        const [yAp] = bAp.createOp(
            "AveragePool",
            [xAp],
            { kernel_shape: [2, 2], strides: [1, 1], pads: [0, 0, 0, 0] },
            [{ type: DataType.FLOAT, shape: [1, 2, 7, 7] }],
        );
        if (yAp.is(TensorNode)) yAp.as(TensorNode).setType("output");
        const outAp = new OnnxGraphTransformer({
            canonicalize: true,
            loopLowering: true,
            fuse: true,
            optimize: true,
        }).apply(gAp, false);
        const apTopOps = outAp
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(apTopOps.filter((t) => t === "Loop").length).toBe(1);
        expect(apTopOps).not.toContain("Div");
        expect(apTopOps).not.toContain("Conv");

        // 4. Non-self-carry carry-tensor loop reuses Loop iter without redundant out_ptr carry
        // Inputs are [M, cond, outCarry, inPtr] (4 instead of 5)
        const { graph: gRelu } = buildChain([2, 3]);
        new LoopLoweringPass({
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
            useScanOutputs: false,
        }).run(gRelu, new HistoryManager(gRelu));
        const reluLoop = gRelu
            .getOperationNodes()
            .toArray()
            .find((o) => o.type === "Loop")!;
        expect(reluLoop.getInputs()!.length).toBe(4);
        expect(
            reluLoop.regions[0]
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Add").length,
        ).toBe(2); // 1 compute Add + 1 input AGU Add (no out_ptr Add)
    });

    it("supports Schedule presets (sfu-loops, flat-loops, cgra-tiled) and emits BOOL predicates in LowerMin/LowerMax", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "preset_sm");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const [smOut] = b.createOp("Softmax", [x], { axis: -1 }, [
            { type: DataType.FLOAT, shape: [2, 4] },
        ]);
        if (smOut.is(TensorNode)) smOut.as(TensorNode).setType("output");

        const sfuCfg = Schedule.preset("sfu-loops").toConfig();
        expect(sfuCfg.recipeOverrides?.["Softmax"]).toBe("LowerSoftmaxLoop");
        expect(sfuCfg.recipeOverrides?.["LogSoftmax"]).toBe("LowerLogSoftmaxLoop");
        expect(sfuCfg.recipeOverrides?.["Det"]).toBe("LowerDetLoop");
        const flatCfg = Schedule.preset("flat-loops").toConfig();
        expect(flatCfg.recipeOverrides?.["Softmax"]).toBe("LowerSoftmax");
        const tiledCfg = Schedule.preset("cgra-tiled").toConfig();
        expect(tiledCfg.splits?.length).toBeGreaterThanOrEqual(2);

        const sfuOut = Schedule.preset("sfu-loops").apply(g);
        expect(
            sfuOut
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop").length,
        ).toBe(1);

        // LowerMin and LowerMax must emit BOOL predicates for Where
        for (const [opName, rec] of [
            ["Min", new LowerMinRecipe()],
            ["Max", new LowerMaxRecipe()],
        ] as const) {
            const gMM = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
            const bMM = new GraphBuilder(gMM, "mm_bool");
            const aMM = gMM
                .addNode("A")
                .init(new TensorNode.Builder(DataType.FLOAT, [], "input"))
                .as(TensorNode);
            const bIn = gMM
                .addNode("B")
                .init(new TensorNode.Builder(DataType.FLOAT, [], "input"))
                .as(TensorNode);
            const [yMM] = bMM.createOp(opName, [aMM, bIn], {}, [
                { type: DataType.FLOAT, shape: [] },
            ]);
            if (yMM.is(TensorNode)) yMM.as(TensorNode).setType("output");
            rec.apply(gMM.getOperationNodes().toArray()[0], bMM);
            const cmp = gMM
                .getOperationNodes()
                .toArray()
                .find((o) => o.type === "Less" || o.type === "Greater")!;
            expect(cmp.getOutgoers.targets.filterIs(TensorNode).first()!.literalType).toBe(
                DataType.BOOL,
            );
        }
    });
});
