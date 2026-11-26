import Graph from "@specs-feup/flow/graph/Graph";
import OnnxGraph from "../../src/onnx/ir/OnnxGraph.js";
import TensorNode from "../../src/onnx/ir/TensorNode.js";
import { GraphBuilder } from "../../src/onnx/ir/GraphBuilder.js";
import type { AttributeMap, TensorProto, ConcreteValueNode } from "../../src/onnx/ir/OnnxTypes.js";
import { DataType, type KnownShape } from "../../src/onnx/ir/OnnxTypes.js";
import type { DecompositionRecipe } from "../../src/onnx/transformation/Recipe.js";
import inferShapes from "../../src/onnx/schema/InferShapes.js";
import { expect } from "vitest";
import validateGraph from "../../src/onnx/validation/ValidateGraph.js";

export type TestInputSpec =
    | KnownShape
    | { shape: KnownShape; dataType?: DataType }
    | { constant: TensorProto };

export function applyRecipe(
    recipe: DecompositionRecipe,
    opType: string,
    inputSpecs: TestInputSpec[] = [[2, 2]],
    attributes: AttributeMap = {},
    defaultDataType: DataType = DataType.FLOAT,
    outputNames?: string[],
): { graph: OnnxGraph.Class; producedOps: string[] } {
    const graph = Graph.create().init(new OnnxGraph.Builder()).as(OnnxGraph);
    const builder = new GraphBuilder(graph, "test");

    const inputs: ConcreteValueNode[] = inputSpecs.map((spec, i) => {
        if (!Array.isArray(spec) && "constant" in spec) {
            return builder.createConstant(`const_${i}`, spec.constant);
        }
        const isObj = !Array.isArray(spec) && typeof spec === "object";
        const shape = isObj ? spec.shape : (spec as KnownShape);
        const dtype = isObj && spec.dataType !== undefined ? spec.dataType : defaultDataType;
        return graph
            .addNode(`in_${i}`)
            .init(new TensorNode.Builder(dtype, shape, "input"))
            .as(TensorNode);
    });

    if (outputNames && outputNames.length > 0) {
        builder.createOpWithExact("target_op", opType, inputs, outputNames, attributes);
    } else {
        builder.createOp(opType, inputs, attributes);
    }

    try {
        inferShapes(graph);
    } catch {
        // Non-fatal for ops with partial synthetic metadata
    }

    const targetOp = graph
        .getOperationNodes()
        .toArray()
        .find((o) => o.type === opType)!;

    for (const out of targetOp.getOutputs()) {
        if (out.is(TensorNode)) {
            out.as(TensorNode).setType("output");
        }
    }

    expect(recipe.match(targetOp)).toBe(true);
    recipe.apply(targetOp, builder);
    expect(() => validateGraph(graph)).not.toThrow();

    const producedOps = graph
        .getOperationNodes()
        .toArray()
        .map((o) => o.type);
    return { graph, producedOps };
}
