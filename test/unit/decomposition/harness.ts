import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../../src/Onnx/OnnxGraph.js";
import TensorNode from "../../../src/Onnx/TensorNode.js";
import { GraphBuilder } from "../../../src/Onnx/GraphBuilder.js";
import type { AttributeMap, ConcreteValueNode } from "../../../src/Onnx/OnnxTypes.js";
import { DataType, type KnownShape } from "../../../src/Onnx/OnnxTypes.js";
import type { DecompositionRecipe } from "../../../src/Onnx/transformation/Recipe.js";
import { expect } from "vitest";

export function testRecipeApplication(
    recipe: DecompositionRecipe,
    opType: string,
    inputShapes: KnownShape[],
    attributes: AttributeMap = {},
): { graph: OnnxGraph.Class; outTensor: ConcreteValueNode } {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const builder = new GraphBuilder(graph, "test_scope");

    const inputs = inputShapes.map((shape, i) => {
        return graph
            .addNode(`in_${i}`)
            .init(new TensorNode.Builder(DataType.FLOAT, shape, "input"))
            .as(TensorNode);
    });

    const [outTensor] = builder.createOp(opType, inputs, attributes);
    const targetOp = graph
        .getOperationNodes()
        .toArray()
        .find((o) => o.type === opType)!;

    expect(recipe.match(targetOp)).toBe(true);

    // Apply rewrite
    recipe.apply(targetOp, builder);

    return { graph, outTensor };
}
