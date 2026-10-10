import { describe, it, expect } from "vitest";
import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import OperationNode from "../../../src/Onnx/OperationNode.js";
import RegionArgumentNode from "../../../src/Onnx/RegionArgumentNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import { DataType } from "../../../src/Onnx/OnnxTypes.js";
import {
    scalarOfType,
    scalarZeroOfType,
    constI64,
    constF32,
    tensorOnesConst,
    makeValueScalar1,
    makeI64ShapeConst,
    constBuilder,
    getProducer,
    getConsumers,
    shapeOf,
    editShapeDim,
} from "../../../src/Onnx/utils/GraphFactories.js";
import {
    formatId,
    addEdge,
    toArrayLike,
    removeInitializerByName,
    maybeRemoveOrphanConstant,
    findTensorByOnnxName,
    findConstantProducerAsTensor,
    isValueNode,
    asValueNode,
    tryAsValueNode,
    isConcreteValueNode,
    asConcreteValueNode,
    tryAsConcreteValueNode,
    isOnnxNode,
    asOnnxNode,
    tryAsOnnxNode,
    chunkTensor,
    topologicalSortOperationNodesSubset,
} from "../../../src/Onnx/utils/GraphQueries.js";
import {
    toScalar,
    toNumShape,
    isKnownDim,
    isNumeric,
    resolveShapeToNumbers,
    getAttr,
    getIntAttr,
    getFloatAttr,
    getStringAttr,
    getIntsAttr,
    getFloatsAttr,
    getSmallestRankShape,
    getLargestRankShape,
    unsqueezeIdx,
    as1D,
} from "../../../src/Onnx/utils/ShapeMath.js";
import { initializeSchemaRegistry } from "../../../src/Onnx/Schema/index.js";

describe("GraphFactories, GraphQueries & ShapeMath", () => {
    initializeSchemaRegistry();

    it("creates constant nodes through GraphFactories helpers", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        const sType = scalarOfType(graph, "s_type", 7, DataType.INT32);
        expect(sType.literalType).toBe(DataType.INT32);

        const sZero = scalarZeroOfType(graph, "s_zero", DataType.FLOAT);
        expect(sZero.literalType).toBe(DataType.FLOAT);

        const c64 = constI64(graph, "c64", [10, 20]);
        expect(c64.shape).toEqual([2]);

        const cF32 = constF32(graph, "cF32", [1.1, 2.2]);
        expect(cF32.literalType).toBe(DataType.FLOAT);

        const ones = tensorOnesConst(graph, "ones", DataType.FLOAT, [2, 3]);
        expect(ones.shape).toEqual([2, 3]);

        const s1 = makeValueScalar1(graph, "s1", DataType.INT64, 42);
        expect(s1.shape).toEqual([1]);

        const shp = makeI64ShapeConst(graph, "shp", [4, 4]);
        expect(shp.shape).toEqual([2]);

        const builderInstance = constBuilder(99);
        expect(builderInstance).toBeDefined();
    });

    it("inspects graph producers, consumers, shapeOf, and editShapeDim", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 4], "input"))
            .as(TensorNode);
        const [reluOut] = builder.createOp("Relu", [x]);

        expect(getProducer(reluOut)?.type).toBe("Relu");
        expect(getConsumers(x).length).toBe(1);

        const shpNode = shapeOf(graph, x, "shape_x");
        expect(shpNode.literalType).toBe(DataType.INT64);

        const updateVal = constI64(graph, "dim_upd", [8]);
        const edited = editShapeDim(graph, shpNode, 0, updateVal, "edit_test");
        expect(edited).toBeDefined();
    });

    it("verifies GraphQueries helpers, guards, and casts", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        expect(formatId("prefix", "node")).toBe("prefix_node");

        expect(toArrayLike([1, 2])).toEqual([1, 2]);
        expect(toArrayLike({ toArray: () => [3, 4] })).toEqual([3, 4]);
        expect(toArrayLike({})).toEqual([]);

        const tNode = graph
            .addNode("t1")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const cNode = constI64(graph, "c1", [1]);
        const opNode = graph
            .addNode("op1")
            .init(new OperationNode.Builder("Relu", [tNode]))
            .as(OperationNode);
        const raNode = graph
            .addNode("ra1")
            .init(new RegionArgumentNode.Builder(0, "orig_t1", DataType.FLOAT, [2]))
            .as(RegionArgumentNode);

        addEdge(graph, opNode, tNode, DataType.FLOAT);

        expect(isValueNode(tNode)).toBe(true);
        expect(isValueNode(cNode)).toBe(true);
        expect(isValueNode(raNode)).toBe(true);
        expect(isValueNode(opNode)).toBe(false);

        expect(asValueNode(tNode).id).toBe(tNode.id);
        expect(asValueNode(tNode).is(TensorNode)).toBe(true);
        expect(() => asValueNode(opNode)).toThrow(/Expected a ValueNode/);
        expect(tryAsValueNode(opNode)).toBeUndefined();
        expect(tryAsValueNode(tNode)?.id).toBe(tNode.id);

        expect(isConcreteValueNode(tNode)).toBe(true);
        expect(isConcreteValueNode(raNode)).toBe(false);
        expect(asConcreteValueNode(cNode).id).toBe(cNode.id);
        expect(() => asConcreteValueNode(raNode)).toThrow(/Expected a ConcreteValueNode/);
        expect(tryAsConcreteValueNode(raNode)).toBeUndefined();
        expect(tryAsConcreteValueNode(cNode)?.id).toBe(cNode.id);

        expect(isOnnxNode(opNode)).toBe(true);
        expect(asOnnxNode(opNode).id).toBe(opNode.id);
        expect(tryAsOnnxNode(undefined)).toBeUndefined();
        expect(tryAsOnnxNode(opNode)?.id).toBe(opNode.id);

        expect(findTensorByOnnxName(graph, "t1")?.id).toBe(tNode.id);
        expect(findTensorByOnnxName(graph, "c1")?.id).toBe(cNode.id);
        expect(findConstantProducerAsTensor(graph, "c1")?.id).toBe(cNode.id);

        const orphanConst = constI64(graph, "orphan", [9]);
        maybeRemoveOrphanConstant(graph, orphanConst);
        expect(graph.hasNode("orphan")).toBe(false);

        removeInitializerByName(graph, "unused");
    });

    it("slices tensors into 1D chunks using chunkTensor", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const mat = graph
            .addNode("Matrix")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);

        const chunks = chunkTensor(builder, mat, 0);
        expect(chunks.length).toBe(2);
    });

    it("exercises ShapeMath attribute extraction, scalar coercions, and rank comparisons", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);

        expect(isNumeric(DataType.FLOAT)).toBe(true);
        expect(isNumeric(DataType.STRING)).toBe(false);
        expect(isNumeric(DataType.BOOL)).toBe(false);

        expect(toNumShape([1, "2", "sym"])).toEqual([1, 2, undefined]);
        expect(toNumShape(undefined)).toBeUndefined();

        expect(isKnownDim(5)).toBe(true);
        expect(isKnownDim(-1)).toBe(false);
        expect(isKnownDim(undefined)).toBe(false);

        const t2D = graph
            .addNode("t2D")
            .init(new TensorNode.Builder(DataType.FLOAT, [2, 3], "input"))
            .as(TensorNode);
        const scalarNode = toScalar(graph, t2D, "to_scalar_test");
        expect(scalarNode.shape).toEqual([]);
        expect(toScalar(graph, scalarNode, "noop")).toBe(scalarNode);

        expect(resolveShapeToNumbers(t2D)).toEqual([2, 3]);

        const op = graph
            .addNode("op_attrs")
            .init(
                new OperationNode.Builder("CustomOp", [], {
                    iVal: 10,
                    fVal: 2.5,
                    sVal: "hello",
                    intsVal: [1, 2, 3],
                    floatsVal: [0.1, 0.2],
                }),
            )
            .as(OperationNode);

        expect(getAttr(op, "missing", "default")).toBe("default");
        expect(getIntAttr(op, "iVal", 0)).toBe(10);
        expect(getFloatAttr(op, "fVal", 0)).toBe(2.5);
        expect(getStringAttr(op, "sVal", "")).toBe("hello");
        expect(getIntsAttr(op, "intsVal", [])).toEqual([1, 2, 3]);
        expect(getFloatsAttr(op, "floatsVal", [])).toEqual([0.1, 0.2]);

        const tRank1 = graph
            .addNode("r1")
            .init(new TensorNode.Builder(DataType.FLOAT, [5], "input"))
            .as(TensorNode);
        expect(getSmallestRankShape([t2D, tRank1])).toEqual([5]);
        expect(getSmallestRankShape([])).toEqual([]);
        expect(getLargestRankShape([t2D, tRank1])).toEqual([2, 3]);
        expect(getLargestRankShape([])).toEqual([]);

        const unsq = unsqueezeIdx(graph, tRank1, constI64(graph, "ax", [0]), "unsq_tag");
        expect(unsq.shape).toEqual([1]);

        const oneD = as1D(graph, "as1d_tag", scalarNode);
        expect(oneD.shape).toEqual([1]);
    });

    it("sorts subsets of operation nodes handling dependencies and cycles", () => {
        const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
        const builder = new GraphBuilder(graph);

        const x = graph
            .addNode("X")
            .init(new TensorNode.Builder(DataType.FLOAT, [2], "input"))
            .as(TensorNode);
        const [r1] = builder.createOp("Relu", [x]);
        const [_r2] = builder.createOp("Abs", [r1]);

        const ops = graph.getOperationNodes().toArray();
        const sorted = topologicalSortOperationNodesSubset(graph, [ops[1], ops[0]]);
        expect(sorted[0].type).toBe("Relu");
        expect(sorted[1].type).toBe("Abs");
    });
});
