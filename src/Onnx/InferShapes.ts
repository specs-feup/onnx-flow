import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type OnnxGraph from "./OnnxGraph.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import RegionArgumentNode from "./RegionArgumentNode.js";
import type { Shape, ValueNode } from "./OnnxTypes.js";
import { DataType } from "./OnnxTypes.js";
import { readTensorData, topologicalSortOperationNodes } from "./Utils.js";
import OnnxEdge from "./OnnxEdge.js";
import type OperationNode from "./OperationNode.js";
import { OpRegistry } from "./Schema/OpRegistry.js";
import type { TensorInfo } from "./Schema/OpSchema.js";

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
export function propagateToRegion(outerGraph: OnnxGraph.Class, region: OnnxGraph.Class): void {
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

    // --- Delegate to the Schema ---
    const registry = OpRegistry.getInstance();
    const schema = registry.get(node.type, 19);

    let results: TensorInfo[] = [];

    if (schema && schema.inferShape) {
        // Pass the context and the inferShapes callback to handle subgraphs dynamically!
        results = schema.inferShape(infos, node.getAttributes(), node, graph, inferShapes);
    } else {
        // Fallback for completely unknown/custom ops
        const first = infos.find((i) => typeof i.shape !== "undefined");
        if (first) {
            results = [{ shape: first.shape, dtype: first.dtype }];
        }
    }

    // --- Apply the results to the graph ---
    const outputs = node.getOutgoers.targets.toArray();
    node.getOutgoers.forEach((e) => graph.getEdgeById(e.id)?.remove());

    for (let i = 0; i < outputs.length; i++) {
        const output = outputs[i];
        // Safely map each result to its corresponding output port, fallback to [0] if missing
        let res: TensorInfo = { shape: [], dtype: DataType.UNDEFINED };
        if (i in results) {
            res = results[i];
        } else if (results.length > 0) {
            res = results[0];
        }

        graph.addEdge(node, output).init(new OnnxEdge.Builder(res.dtype, res.shape));

        if (output.is(TensorNode)) {
            const tn = output.as(TensorNode);
            if (Array.isArray(res.shape) && res.shape.length > 0) tn.setShape(res.shape);
            if (res.dtype !== DataType.UNDEFINED) tn.setLiteralType(res.dtype);
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
