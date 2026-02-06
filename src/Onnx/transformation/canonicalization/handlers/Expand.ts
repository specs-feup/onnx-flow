import OnnxGraph from "../../../OnnxGraph.js";
import OperationNode from "../../../OperationNode.js";
import TensorNode from "../../../TensorNode.js";
import { DataType } from "../../../OnnxTypes.js";
import { uniq, addEdge, toArrayLike } from "../../../Utils.js";

export default function expandHandler(g: OnnxGraph.Class, op: OperationNode.Class): boolean {
    if (op.type !== "Expand") return false;

    const ins = op.getInputs?.() ?? [];
    if (ins.length !== 2) return false;

    const xIn = ins[0];
    const shapeIn = ins[1];
    if (!xIn?.is?.(TensorNode) || !shapeIn?.is?.(TensorNode)) return false;

    const X = xIn.as(TensorNode);
    const shape = shapeIn.as(TensorNode);

    const outs = toArrayLike<TensorNode.Class>(op.getOutgoers?.targets?.filterIs?.(TensorNode));
    if (outs.length !== 1) return false;
    const Y = outs[0];

    // Expand preserves X's dtype.
    const dt = (X.literalType as DataType | undefined) ?? (Y.literalType as DataType | undefined);
    if (dt == null) {
        return false;
    }

    // Only handle numeric / bool-ish types we know how to zero-fill.
    switch (dt) {
        case DataType.FLOAT:
        case DataType.FLOAT16:
        case DataType.BFLOAT16:
        case DataType.DOUBLE:
        case DataType.INT8:
        case DataType.UINT8:
        case DataType.INT16:
        case DataType.UINT16:
        case DataType.INT32:
        case DataType.UINT32:
        case DataType.INT64:
        case DataType.UINT64:
        case DataType.BOOL:
            break;
        default:
            return false;
    }

    // Pick a reasonable meta-shape for the zeros/add result.
    // This is for graph typing only; runtime shape still comes from ConstantOfShape(shape).
    let outShape: Array<number | string> | undefined;

    if (Array.isArray(Y.shape) && Y.shape.length > 0) {
        outShape = [...Y.shape];
    } else if (Array.isArray(X.shape) && X.shape.length > 0) {
        // We at least know the rank; dims may be unknown.
        outShape = new Array(X.shape.length).fill(undefined);
    } else {
        // Fallback: leave shape unknown; ONNX IR allows this.
        outShape = undefined;
    }

    // 1) zeros_f = ConstantOfShape(shape)  (defaults to FLOAT 0.0)
    const cosOp = g
        .addNode(uniq(g, `${op.id}_expand_fill`))
        .init(new OperationNode.Builder("ConstantOfShape", [shape], {}))
        .as(OperationNode);

    const zerosF = g
        .addNode(uniq(g, `${op.id}_expand_fill_out`))
        .init(new TensorNode.Builder(DataType.FLOAT, outShape as any, "intermediate"))
        .as(TensorNode);

    addEdge(g, cosOp, zerosF, DataType.FLOAT, outShape);

    // 2) If needed, Cast zeros to X's dtype
    let zeros = zerosF;
    if (dt !== DataType.FLOAT) {
        const castOp = g
            .addNode(uniq(g, `${op.id}_expand_cast`))
            .init(new OperationNode.Builder("Cast", [zerosF], { to: dt }))
            .as(OperationNode);

        const zerosCast = g
            .addNode(uniq(g, `${op.id}_expand_cast_out`))
            .init(new TensorNode.Builder(dt, outShape as any, "intermediate"))
            .as(TensorNode);

        addEdge(g, castOp, zerosCast, dt, outShape);
        zeros = zerosCast;
    }

    // 3) Y = Add(X, zeros)  (broadcast does the Expand)
    const addOp = g
        .addNode(uniq(g, `${op.id}_expand_add`))
        .init(new OperationNode.Builder("Add", [X, zeros], {}))
        .as(OperationNode);

    addEdge(g, addOp, Y, dt, outShape ?? Y.shape);

    g.getNodeById(op.id)?.remove();

    return true;
}
