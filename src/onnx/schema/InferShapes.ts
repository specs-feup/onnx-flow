import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type OnnxGraph from "../ir/OnnxGraph.js";
import TensorNode from "../ir/TensorNode.js";
import ConstantNode from "../ir/ConstantNode.js";
import RegionArgumentNode from "../ir/RegionArgumentNode.js";
import type { Shape, ValueNode } from "../ir/OnnxTypes.js";
import { DataType } from "../ir/OnnxTypes.js";
import { readTensorData, topologicalSortOperationNodes } from "../utils/index.js";
import OnnxEdge from "../ir/OnnxEdge.js";
import type OperationNode from "../ir/OperationNode.js";
import { OpRegistry } from "./OpRegistry.js";
import type { TensorInfo } from "./OpSchema.js";

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
    const schema = registry.get(node.type, graph.getTargetOpset());

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
    // Update existing outgoing edges in-place so edge/node iteration order is preserved.
    const existingEdges = node.getOutgoers.toArray() as OnnxEdge.Class[];
    const outputs = existingEdges.map((e) => e.target);

    for (let i = 0; i < outputs.length; i++) {
        const output = outputs[i];
        // Safely map each result to its corresponding output port, fallback to [0] if missing
        let res: TensorInfo = { shape: [], dtype: DataType.UNDEFINED };
        if (i in results) {
            res = results[i];
        } else if (results.length > 0) {
            res = results[0];
        }

        const edge = existingEdges[i];
        edge.setLiteralType(res.dtype);
        edge.setShape(res.shape);

        if (output.is(TensorNode)) {
            const tn = output.as(TensorNode);
            const isUnknownFallback =
                Array.isArray(res.shape) && res.shape.length === 1 && res.shape[0] === -1;
            const hasMoreSpecificExisting =
                Array.isArray(tn.shape) &&
                (tn.shape.length > 1 ||
                    (tn.shape.length === 1 && typeof tn.shape[0] === "number" && tn.shape[0] >= 0));
            if (
                Array.isArray(res.shape) &&
                res.shape.length > 0 &&
                !(isUnknownFallback && hasMoreSpecificExisting)
            ) {
                tn.setShape(res.shape);
            }
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
