import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/onnx/OnnxGraph.js";
import OperationNode from "../../../src/onnx/OperationNode.js";
import TensorNode from "../../../src/onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/onnx/GraphBuilder.js";
import { DataType } from "../../../src/onnx/OnnxTypes.js";
import { initializeSchemaRegistry } from "../../../src/onnx/schema/index.js";
import { IndexingMode } from "../../../src/DecompositionOptions.js";
import { Schedule } from "../../../src/onnx/transformation/Schedule.js";

describe("Schedule API & Multi-Output Loops", () => {
    initializeSchemaRegistry();

    function buildForkGraph() {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "fork");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const y = g
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const [addT] = b.createOp("Add", [x, y], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
        const [reluT] = b.createOp("Relu", [addT], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
        const [negT] = b.createOp("Neg", [addT], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
        if (reluT.is(TensorNode)) reluT.as(TensorNode).setType("output");
        if (negT.is(TensorNode)) negT.as(TensorNode).setType("output");

        const addOp = addT.incomers.sources.filterIs(OperationNode).first()!;
        const reluOp = reluT.incomers.sources.filterIs(OperationNode).first()!;
        const negOp = negT.incomers.sources.filterIs(OperationNode).first()!;
        return { g, addOp, reluOp, negOp };
    }

    it("selects recipes with useRecipe and strips/tiles per-op Loops with split", () => {
        const gSm = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bSm = new GraphBuilder(gSm, "sm");
        const xSm = gSm
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 5], "input"))
            .as(TensorNode);
        const [outSm] = bSm.createOp("Softmax", [xSm], { axis: -1 }, [
            { type: DataType.FLOAT, shape: [4, 5] },
        ]);
        if (outSm.is(TensorNode)) outSm.as(TensorNode).setType("output");

        new Schedule()
            .useRecipe("Softmax", "LowerSoftmaxLoop")
            .vectorize("Softmax", 4)
            .apply(gSm, { optimize: true, indexingMode: IndexingMode.Pointer });

        const smLoops = gSm
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(smLoops.length).toBe(1);
        expect(smLoops[0].getMetadata("vectorWidth")).toBe(4);
        const smInnerLoops = smLoops[0].regions[0]
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(smInnerLoops.length).toBe(2);

        // Per-op strip-mining split + vectorize through canonicalization (Relu -> Max)
        const gTile = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const bTile = new GraphBuilder(gTile, "sp");
        const xT = gTile
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [4, 5], "input"))
            .as(TensorNode);
        const [rT] = bTile.createOp("Relu", [xT], {}, [{ type: DataType.FLOAT, shape: [4, 5] }]);
        if (rT.is(TensorNode)) rT.as(TensorNode).setType("output");

        new Schedule()
            .split("Relu", 4)
            .vectorize("Relu", 4)
            .apply(gTile, { optimize: true, indexingMode: IndexingMode.Pointer });

        const tiledLoops = gTile
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(tiledLoops.length).toBe(1);
        expect(tiledLoops[0].getMetadata("tiled")).toBe(true);
        expect(tiledLoops[0].getMetadata("vectorWidth")).toBe(4);
        const tiledInner = tiledLoops[0].regions[0]
            .getOperationNodes()
            .toArray()
            .filter((o) => o.type === "Loop");
        expect(tiledInner.length).toBe(1);
        expect(tiledInner[0].getMetadata("vectorWidth")).toBe(4);
    });

    it("honors stopAt('none') and stopAt('canonicalization') per node", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "stop");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const [r] = b.createOp("Relu", [x], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
        const [s] = b.createOp("Sigmoid", [r], {}, [{ type: DataType.FLOAT, shape: [2, 3] }]);
        if (s.is(TensorNode)) s.as(TensorNode).setType("output");

        const reluId = r.incomers.sources.filterIs(OperationNode).first()!.id;
        new Schedule().stopAt(reluId, "none").apply(g, { optimize: true });

        const topOps = g
            .getOperationNodes()
            .toArray()
            .map((o) => o.type);
        expect(topOps).toContain("Relu");
        expect(topOps).not.toContain("Sigmoid");
    });

    it("fuses multi-output fork graphs via Schedule.fuse and fuseMultiOutput option", () => {
        for (const mode of ["schedule", "option", "scan"] as const) {
            const { g, addOp, reluOp, negOp } = buildForkGraph();
            if (mode === "schedule") {
                new Schedule().fuse([addOp.id, reluOp.id, negOp.id]).apply(g, { optimize: true });
            } else {
                new Schedule().apply(g, {
                    optimize: true,
                    fuseMultiOutput: true,
                    useScanOutputs: mode === "scan",
                });
            }
            const loops = g
                .getOperationNodes()
                .toArray()
                .filter((o) => o.type === "Loop");
            expect(loops.length).toBe(1);
            expect(loops[0].getOutputs().length).toBeGreaterThanOrEqual(2);
            expect(g.getOutputTensorNodes().length).toBe(2);
            const outProducers = g
                .getOutputTensorNodes()
                .toArray()
                .map((t) => t.incomers.sources.filterIs(OperationNode).first());
            expect(
                outProducers.every(
                    (p) =>
                        p?.type === "Reshape" &&
                        p.getInputs()?.[0].incomers.sources.first()?.id === loops[0].id,
                ),
            ).toBe(true);
        }
    });

    it("supports computeAt, reorder, unroll, and cacheRead", () => {
        const g = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const b = new GraphBuilder(g, "unr");
        const x = g
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const y = g
            .addNode("Y")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 2], "input"))
            .as(TensorNode);
        const [addT] = b.createOp("Add", [x, y], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);
        const [reluT] = b.createOp("Relu", [addT], {}, [{ type: DataType.FLOAT, shape: [2, 2] }]);
        if (reluT.is(TensorNode)) reluT.as(TensorNode).setType("output");

        const addOp = addT.incomers.sources.filterIs(OperationNode).first()!;
        const reluOp = reluT.incomers.sources.filterIs(OperationNode).first()!;

        new Schedule()
            .cacheRead("X", reluOp.id) // no-op consumer filter test
            .cacheRead("X", addOp.id)
            .computeAt(addOp.id, reluOp.id)
            .reorder(reluOp.id, [1, 0])
            .unroll(reluOp.id, 8)
            .apply(g, { optimize: true, useScanOutputs: true });

        const ops = g.getOperationNodes().toArray();
        expect(ops.some((o) => o.type === "Loop")).toBe(false);
        expect(ops.some((o) => o.type === "Identity" && o.getMetadata("cacheRead") === true)).toBe(
            true,
        );
    });
});
