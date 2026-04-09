import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type OnnxGraph from "./OnnxGraph.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import RegionArgumentNode from "./RegionArgumentNode.js";
import type { Shape, ValueNode } from "./OnnxTypes.js";
import { DataType } from "./OnnxTypes.js";
import {
    decodeIntegerVectorFromTensorProto,
    readTensorData,
    topologicalSortOperationNodes,
    UNKNOWN_SHAPE,
} from "./Utils.js";
import OnnxEdge from "./OnnxEdge.js";
import type OperationNode from "./OperationNode.js";
import { OpRegistry } from "./Schema/OpRegistry.js";

/** Helper: resolve a tensor's shape from its node or incoming edge */
function resolveTensorShape(t: BaseNode.Class): Shape {
    if (t.is(ConstantNode)) {
        return t.as(ConstantNode).shape;
    }
    if (t.is(RegionArgumentNode)) {
        return t.as(RegionArgumentNode).shape;
    }
    if (t.is(TensorNode)) {
        const tn = t.as(TensorNode);
        return tn.shape;
    }
    return [];
}

/** Helper: resolve a tensor's type */
function resolveLiteralType(t: BaseNode.Class): DataType {
    if (t.is(ConstantNode)) return t.as(ConstantNode).literalType;
    if (t.is(RegionArgumentNode)) return t.as(RegionArgumentNode).literalType;
    if (t.is(TensorNode)) {
        const tn = t.as(TensorNode);
        if (tn.literalType !== DataType.UNDEFINED) return tn.literalType;
        const interEdge = tn.getIncomers.first() as OnnxEdge.Class | undefined;
        if (interEdge?.literalType !== undefined) return interEdge.literalType;
    }
    return DataType.UNDEFINED;
}

/** * Recursive Propagator for Subgraphs
 * This pushes outer variable info into the inner graph's RegionArgumentNodes.
 */
function propagateToRegion(outerGraph: OnnxGraph.Class, region: OnnxGraph.Class) {
    const regionNodes = region.getNodes();
    for (const node of regionNodes) {
        if (node.is(RegionArgumentNode)) {
            const arg = node.as(RegionArgumentNode);
            const outerNode = outerGraph.getNodeById(arg.originalName);

            if (outerNode) {
                const shape = resolveTensorShape(outerNode);
                const type = resolveLiteralType(outerNode);

                arg.setShape(shape);
                arg.setLiteralType(type);
            }
        }
    }
}

/** Shape inference for a node */
export function inferNodeShape(node: OperationNode.Class, graph: OnnxGraph.Class): void {
    const inputs: (ValueNode | undefined)[] = node.getInputs() ?? [];

    const infos = inputs.map((inp: ValueNode | undefined) => {
        let shape: Shape = [];
        let dtype = DataType.UNDEFINED;
        let constantValue: number[] | undefined = undefined;

        if (inp !== undefined && inp.is(ConstantNode)) {
            const cn = inp.as(ConstantNode);
            shape = cn.shape;
            dtype = cn.literalType;

            // Only decode small tensors (like shapes, axes, pads) to avoid
            // OOM crashes and buffer overruns on massive weight matrices.
            const numElements = shape.reduce(
                (a, b) => (a as number) * (typeof b === "number" ? b : 1),
                1,
            ) as number;
            if (numElements <= 32) {
                constantValue = readTensorData(cn);
            }
        } else if (inp !== undefined && inp.is(RegionArgumentNode)) {
            const ra = inp.as(RegionArgumentNode);
            shape = ra.shape;
            dtype = ra.literalType;
        } else if (inp !== undefined && inp.is(TensorNode)) {
            const tns = inp.as(TensorNode);
            const directEdge = graph.getEdge(inp.id, node.id)?.tryAs(OnnxEdge);

            shape = directEdge?.shape ?? (tns.shape as Shape | undefined) ?? [-1];
            dtype = directEdge?.literalType ?? tns.literalType;
        }

        return { shape, dtype, constantValue };
    });

    let outShape: Shape = [];
    let outDtype = infos[0]?.dtype ?? DataType.UNDEFINED;

    // --- 1. Subgraph / Control Flow Interception ---
    // (We must handle these here because schemas cannot easily traverse inner regions)
    if (node.type === "Loop") {
        const body = node.regions[0];
        propagateToRegion(graph, body);

        const loopInputs = inputs;
        const bodyInputs = body.getInputTensorNodes().toArray();

        for (let i = 0; i < loopInputs.length - 2; i++) {
            const vInit = loopInputs[i + 2];
            const vBody = bodyInputs[i + 2];
            if (vInit !== undefined) vBody.setShape(resolveTensorShape(vInit));
            if (vInit !== undefined) vBody.setLiteralType(resolveLiteralType(vInit));
        }

        inferShapes(body);

        const bodyOutputs = body.getOutputTensorNodes().toArray();
        const loopOutputs = node.getOutgoers.targets.filterIs(TensorNode).toArray();

        for (let i = 0; i < loopOutputs.length; i++) {
            const bOut = bodyOutputs[i + 1];
            const lOut = loopOutputs[i];

            if (i < loopInputs.length - 2) {
                lOut.setShape(bOut.shape);
                lOut.setLiteralType(bOut.literalType);
            } else {
                const tripCnt =
                    inputs[0] !== undefined && inputs[0].is(ConstantNode)
                        ? decodeIntegerVectorFromTensorProto(
                              inputs[0].as(ConstantNode).constantValue,
                          )![0]
                        : undefined;
                const dim0 = tripCnt !== undefined ? tripCnt : UNKNOWN_SHAPE[0];
                lOut.setShape([dim0, ...bOut.shape]);
                lOut.setLiteralType(bOut.literalType);
            }
        }
        return;
    }

    if (node.type === "If") {
        for (const region of node.regions) {
            propagateToRegion(graph, region);
            inferShapes(region);
        }

        const thenGraph = node.regions[0];
        const thenOutputs = thenGraph.getOutputTensorNodes().toArray();
        const ifOutputs = node.getOutgoers.targets.filterIs(TensorNode).toArray();

        for (let i = 0; i < ifOutputs.length; i++) {
            ifOutputs[i].setShape(thenOutputs[i].shape);
            ifOutputs[i].setLiteralType(thenOutputs[i].literalType);
        }
        return;
    }

    // --- 2. Delegate to the Schema ---
    const registry = OpRegistry.getInstance();
    // Defaulting to opset 19: ideally, we should pass the model's actual opset version here if tracked
    const schema = registry.get(node.type, 19);

    if (schema && schema.inferShape) {
        const results = schema.inferShape(infos, node.getAttributes());
        if (results.length > 0) {
            outShape = results[0].shape;
            outDtype = results[0].dtype;
        }
    } else {
        // Fallback for completely unknown/custom ops
        const first = infos.find((i) => typeof i.shape !== "undefined");
        if (first) {
            outShape = first.shape;
            outDtype = first.dtype;
        }
    }

    // --- 3. Apply the results to the graph ---
    const outputs = node.getOutgoers.targets;
    const outputTensors = outputs.filter((t) => t.is(TensorNode));

    node.getOutgoers.forEach((e) => graph.getEdgeById(e.id)?.remove());

    for (const output of outputs) {
        graph.addEdge(node, output).init(new OnnxEdge.Builder(outDtype, outShape));
    }

    if (Array.isArray(outShape) && outShape.length > 0) {
        for (const out of outputTensors) {
            const tn = out.tryAs(TensorNode);
            if (!tn) continue;
            tn.setShape(outShape);
            tn.setLiteralType(outDtype);
        }
    }
}

/** Main shape inference */
export default function inferShapes(graph: OnnxGraph.Class): void {
    const ops = topologicalSortOperationNodes(graph);

    for (const node of ops) {
        inferNodeShape(node, graph);
    }
}
