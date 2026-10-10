import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import type { AttributeMap } from "../../../src/Onnx/OnnxTypes.js";
import { DataType, type KnownShape } from "../../../src/Onnx/OnnxTypes.js";
import type { DecompositionRecipe } from "../../../src/Onnx/transformation/Recipe.js";
import { expect } from "vitest";

export type TestInputSpec = KnownShape | { shape: KnownShape; dataType?: DataType };

export function applyRecipe(
    recipe: DecompositionRecipe,
    opType: string,
    inputSpecs: TestInputSpec[] = [[2, 2]],
    attributes: AttributeMap = {},
    defaultDataType: DataType = DataType.FLOAT,
): { graph: OnnxGraph.Class; producedOps: string[] } {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const builder = new GraphBuilder(graph, "test");

    const inputs = inputSpecs.map((spec, i) => {
        const isObj = !Array.isArray(spec) && typeof spec === "object";
        const shape = isObj ? spec.shape : (spec as KnownShape);
        const dtype = isObj && spec.dataType !== undefined ? spec.dataType : defaultDataType;
        return graph
            .addNode(`in_${i}`)
            .init(new TensorNode.Builder(dtype, shape, "input"))
            .as(TensorNode);
    });

    builder.createOp(opType, inputs, attributes);
    const targetOp = graph
        .getOperationNodes()
        .toArray()
        .find((o) => o.type === opType)!;

    expect(recipe.match(targetOp)).toBe(true);
    recipe.apply(targetOp, builder);

    const producedOps = graph
        .getOperationNodes()
        .toArray()
        .map((o) => o.type);
    return { graph, producedOps };
}
