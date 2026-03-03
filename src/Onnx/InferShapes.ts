import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import type OnnxGraph from "./OnnxGraph.js";
import TensorNode from "./TensorNode.js";
import ConstantNode from "./ConstantNode.js";
import RegionArgumentNode from "./RegionArgumentNode.js";
import type { Shape, TensorProto, ValueNode } from "./OnnxTypes.js";
import { DataType } from "./OnnxTypes.js";
import {
    broadcastShapes,
    decodeIntegerVectorFromTensorProto,
    getAttr,
    inferPoolDim,
    normalizeAxis,
    toNum,
    topologicalSortOperationNodes,
    toStaticShape,
    UNKOWN_SHAPE,
} from "./Utils.js";
import OnnxEdge from "./OnnxEdge.js";
import OperationNode from "./OperationNode.js";

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
            // Look up the original node in the outer graph
            // Note: This assumes unique IDs across the entire model or correct originalName references.
            const outerNode = outerGraph.getNodeById(arg.originalName);

            if (outerNode) {
                // Propagate Type and Shape
                const shape = resolveTensorShape(outerNode);
                const type = resolveLiteralType(outerNode);

                arg.setShape(shape);
                arg.setLiteralType(type);
            }
        }
    }
}

/** Main shape inference */
export default function inferShapes(graph: OnnxGraph.Class): void {
    const ops = topologicalSortOperationNodes(graph);

    for (const node of ops) {
        const inputs: (ValueNode | undefined)[] = node.getInputs() ?? [];

        const infos = inputs.map((inp: ValueNode | undefined) => {
            let shape: Shape = [];
            let dtype = DataType.UNDEFINED;

            if (inp !== undefined && inp.is(ConstantNode)) {
                const cn = inp.as(ConstantNode);
                shape = cn.shape;
                dtype = cn.literalType;
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

            return {
                shape,
                dtype,
            };
        });

        let outShape: Shape = [];
        let outDtype = infos[0]?.dtype ?? DataType.UNDEFINED;

        if (node.type === "Loop") {
            const body = node.regions[0];
            // 1. Propagate Captures (Outer -> Inner)
            propagateToRegion(graph, body);

            // 2. Propagate Loop Inputs -> Body Inputs
            // Loop inputs: [trip, cond, v_init_1, v_init_2...]
            // Body inputs: [iter, cond, v_in_1, v_in_2...]
            const loopInputs = inputs; // skip trip/cond logic for shape?
            // Usually v_init starts at index 2
            const bodyInputs = body.getInputTensorNodes().toArray();
            // Body inputs are strictly ordered: iter, cond, loop_vars...

            // Set shapes for loop variables in body based on v_initial
            for (let i = 0; i < loopInputs.length - 2; i++) {
                const vInit = loopInputs[i + 2];
                const vBody = bodyInputs[i + 2]; // skip iter, cond
                if (vInit !== undefined) vBody.setShape(resolveTensorShape(vInit));
                if (vInit !== undefined) vBody.setLiteralType(resolveLiteralType(vInit));
            }

            // 3. Recurse
            inferShapes(body);

            // 4. Propagate Body Outputs -> Loop Outputs
            // Body outputs: [cond_out, v_out_1... v_out_N, scan_1... scan_M]
            // Loop outputs: [v_final_1... v_final_N, scan_1... scan_M]
            // (Scan outputs gain a dimension)

            const bodyOutputs = body.getOutputTensorNodes().toArray();
            const loopOutputs = node.getOutgoers.targets.filterIs(TensorNode).toArray();

            // Loop outputs map to body outputs starting at index 1 (skip cond_out)
            for (let i = 0; i < loopOutputs.length; i++) {
                const bOut = bodyOutputs[i + 1]; // skip cond_out
                const lOut = loopOutputs[i];

                if (i < loopInputs.length - 2) {
                    // Carried variable (shape preserved)
                    lOut.setShape(bOut.shape);
                    lOut.setLiteralType(bOut.literalType);
                } else {
                    // Scan output (prepend dimension, usually trip count or symbolic)
                    // Ideally we find 'trip_count' from input[0]
                    const tripCnt =
                        inputs[0] !== undefined && inputs[0].is(ConstantNode)
                            ? decodeIntegerVectorFromTensorProto(
                                  inputs[0].as(ConstantNode).constantValue,
                              )![0]
                            : undefined;
                    const dim0 = tripCnt !== undefined ? tripCnt : UNKOWN_SHAPE[0];
                    lOut.setShape([dim0, ...bOut.shape]);
                    lOut.setLiteralType(bOut.literalType);
                }
            }
            continue; // Skip standard switch, we handled it
        }

        if (node.type === "If") {
            // Propagate to both branches
            for (const region of node.regions) {
                propagateToRegion(graph, region);
                inferShapes(region);
            }

            // Output shape is union/broadcast of branches (usually identical)
            // Just take from 'then' branch for simplicity
            const thenGraph = node.regions[0];
            const thenOutputs = thenGraph.getOutputTensorNodes().toArray();
            const ifOutputs = node.getOutgoers.targets.filterIs(TensorNode).toArray();

            for (let i = 0; i < ifOutputs.length; i++) {
                ifOutputs[i].setShape(thenOutputs[i].shape);
                ifOutputs[i].setLiteralType(thenOutputs[i].literalType);
            }
            continue;
        }

        switch (node.type) {
            /** ───── Elementwise (broadcast) ───── */
            case "Add":
            case "Sub":
            case "Mul":
            case "Div":
            case "Pow":
            case "Min":
            case "Max":
            case "And":
            case "Or":
            case "Xor":
            case "Greater":
            case "Less":
            case "GreaterOrEqual":
            case "LessOrEqual":
            case "Equal":
            case "NotEqual": {
                const shapes = infos.map((i) => toStaticShape(i.shape));
                outShape = broadcastShapes(...shapes);
                if (
                    [
                        "Greater",
                        "Less",
                        "GreaterOrEqual",
                        "LessOrEqual",
                        "Equal",
                        "NotEqual",
                        "And",
                        "Or",
                        "Xor",
                    ].includes(node.type)
                ) {
                    outDtype = DataType.BOOL;
                }
                break;
            }

            /** ───── Unary activations (shape preserved) ───── */
            case "Relu":
            case "LeakyRelu":
            case "Sigmoid":
            case "Tanh":
            case "Exp":
            case "Sqrt":
            case "Abs":
            case "Neg":
            case "Clip": {
                outShape = infos[0]?.shape ?? [];
                break;
            }

            /** ───── Where ───── */
            case "Where": {
                const sc = infos[0]?.shape ?? [];
                const sx = infos[1]?.shape ?? [];
                const sy = infos[2]?.shape ?? [];
                outShape = broadcastShapes(toStaticShape(sc), toStaticShape(sx), toStaticShape(sy));
                outDtype = infos[1]?.dtype ?? infos[2]?.dtype;
                if (infos[0]?.dtype !== DataType.BOOL) {
                    console.warn("Where: condition input is not BOOL.");
                }
                break;
            }

            /** ───── MatMul (simple 2D) ───── */
            case "MatMul": {
                if (infos.length >= 2) {
                    const [a, b] = infos;
                    if (a.shape.length === 2 && b.shape.length === 2) {
                        outShape = [a.shape[0], b.shape[1]];
                    } else {
                        console.warn("MatMul with non-2D tensors:", a.shape, b.shape);
                        outShape = [];
                    }
                }
                break;
            }

            /** ───── Gemm (2D MatMul + broadcast add) ───── */
            case "Gemm": {
                const a = infos[0]?.shape ?? [];
                const b = infos[1]?.shape ?? [];
                if (a.length === 2 && b.length === 2) {
                    const mm: Shape = [a[0], b[1]];
                    const c = infos[2]?.shape ?? [];
                    outShape = c.length ? broadcastShapes(toStaticShape(mm), toStaticShape(c)) : mm;
                } else {
                    outShape = [];
                    console.warn("Gemm with non-2D inputs; inference skipped.");
                }
                break;
            }

            /** ───── Transpose (with perm attr, default = reverse) ───── */
            case "Transpose": {
                const inputShape = infos[0]?.shape ?? [];
                const perm = getAttr(
                    node,
                    "perm",
                    inputShape.map((_, i) => i).reverse(),
                ) as number[];
                outShape = perm.map((p) => inputShape[p] ?? 1);
                break;
            }

            /** ───── Reshape (ONNX rules: 0 / -1, product preserved) ───── */
            case "Reshape": {
                const inputShape = infos[0]?.shape ?? [];
                const shapeInput: ValueNode | undefined = inputs[1];
                let target: number[] = [];

                if (shapeInput !== undefined && shapeInput.is(ConstantNode)) {
                    target =
                        decodeIntegerVectorFromTensorProto(
                            shapeInput.as(ConstantNode).constantValue,
                        ) ?? [];
                }

                if (target.length > 0 && inputShape.length > 0) {
                    const inNums = inputShape.map((d) => toNum(d) ?? 1);
                    const prodIn = inNums.reduce((a, b) => a * (b || 1), 1) || 1;

                    let inferIndex = -1;
                    let knownProd = 1;
                    const resolved = target.slice();

                    // 0 → copy from input
                    resolved.forEach((d, i) => {
                        if (d === 0) {
                            resolved[i] = inNums[i] ?? 1;
                        }
                    });

                    // -1 → infer from remaining product
                    resolved.forEach((d, i) => {
                        if (d === -1) {
                            if (inferIndex !== -1) {
                                throw new Error("Reshape: multiple -1 in target shape not allowed");
                            }
                            inferIndex = i;
                        } else {
                            knownProd *= d || 1;
                        }
                    });

                    if (inferIndex !== -1) {
                        const missing = prodIn / (knownProd || 1);
                        resolved[inferIndex] = missing;
                    }

                    outShape = resolved;
                } else {
                    outShape = inputShape.slice();
                }
                break;
            }

            /** ───── Unsqueeze / Squeeze ───── */
            case "Unsqueeze": {
                const tensorShape = infos[0]?.shape ?? [];
                const axesNode: ValueNode | undefined = inputs[1];

                let raw: number[] = [];
                if (axesNode !== undefined && axesNode.is(ConstantNode)) {
                    raw =
                        decodeIntegerVectorFromTensorProto(
                            axesNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }

                const axes = [...raw].sort((a, b) => a - b);
                outShape = [...tensorShape];
                for (const ax of axes) outShape.splice(ax, 0, 1);
                break;
            }

            case "Squeeze": {
                const inputShape = infos[0]?.shape ?? [];
                const axesNode: ValueNode | undefined = inputs[1];

                let axes: number[] = [];
                if (axesNode !== undefined && axesNode.is(ConstantNode)) {
                    axes =
                        decodeIntegerVectorFromTensorProto(
                            axesNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }

                if (axes.length === 0) {
                    outShape = inputShape.filter((d) => d !== 1);
                } else {
                    const rank = inputShape.length;
                    const norm = new Set(axes.map((a) => normalizeAxis(a, rank)));
                    outShape = inputShape.filter((dim, idx) => !norm.has(idx) || dim !== 1);
                }
                break;
            }

            /** ───── Gather / GatherElements / Scatter(Elements) ───── */
            case "Gather": {
                const dataShape = infos[0]?.shape ?? [];
                const indicesShape = infos[1]?.shape ?? [];
                const axisRaw = getAttr(node, "axis", 0) as number;
                const axis = normalizeAxis(axisRaw, dataShape.length);
                outShape = [
                    ...dataShape.slice(0, axis),
                    ...indicesShape,
                    ...dataShape.slice(axis + 1),
                ];
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            case "Scan": {
                const outs = node.getOutgoers.targets;

                const firstOutT = outs
                    .filter((t) => t.is(TensorNode))
                    .first()
                    ?.as(TensorNode);

                if (firstOutT !== undefined) {
                    outDtype = firstOutT.literalType;
                }
                outShape = [];
                break;
            }

            case "GatherElements": {
                const indicesShape = infos[1]?.shape ?? [];
                outShape = indicesShape.slice();
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            case "ScatterElements": {
                const dataShape = infos[0]?.shape ?? [];
                outShape = dataShape.slice();
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            case "Scatter": {
                const dataShape = infos[0]?.shape ?? [];
                outShape = dataShape.slice();
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── Slice (with starts/ends/axes/steps) ───── */
            case "Slice": {
                const dataShape = infos[0]?.shape ?? [];
                const rank = dataShape.length;
                if (rank === 0) {
                    outShape = [];
                    break;
                }

                const startsNode: ValueNode | undefined = inputs[1];
                const endsNode: ValueNode | undefined = inputs[2];
                const axesNode: ValueNode | undefined = inputs[3];
                const stepsNode: ValueNode | undefined = inputs[4];

                let starts: number[] = [];
                let ends: number[] = [];
                let axes: number[] = [];
                let steps: number[] = [];

                if (startsNode !== undefined && startsNode.is(ConstantNode)) {
                    starts =
                        decodeIntegerVectorFromTensorProto(
                            startsNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }
                if (endsNode !== undefined && endsNode.is(ConstantNode)) {
                    ends =
                        decodeIntegerVectorFromTensorProto(
                            endsNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }
                if (axesNode !== undefined && axesNode.is(ConstantNode)) {
                    axes =
                        decodeIntegerVectorFromTensorProto(
                            axesNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }
                if (stepsNode !== undefined && stepsNode.is(ConstantNode)) {
                    steps =
                        decodeIntegerVectorFromTensorProto(
                            stepsNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }

                const normAxis = (ax: number, r: number) => (r > 0 ? ((ax % r) + r) % r : 0);

                if (!axes.length) {
                    axes = Array.from({ length: starts.length || rank }, (_, i) => i);
                } else {
                    axes = axes.map((a) => normAxis(a, rank));
                }

                const out = dataShape.slice();

                for (let i = 0; i < axes.length; i++) {
                    const ax = axes[i];
                    const len = dataShape[ax] ?? 0;
                    if (len === 0) continue;

                    let s = starts[i] ?? 0;
                    let e = ends[i] ?? len;
                    const step = steps[i] ?? 1;
                    if (step === 0) continue;

                    const normPos = (pos: number) =>
                        pos < 0
                            ? Math.max(0, toNum(len) ?? 0 + pos)
                            : Math.min(toNum(len) ?? Number.MAX_VALUE, pos);

                    s = normPos(s);
                    e = normPos(toNum(e) ?? e);

                    const size = Math.max(0, Math.ceil((e - s) / step));
                    out[ax] = size;
                }

                outShape = out;
                break;
            }

            /** ───── Pad ───── */
            case "Pad": {
                const dataShape = infos[0]?.shape ?? [];
                const padsNode: ValueNode | undefined = inputs[1];

                let pads: number[] = [];
                if (padsNode !== undefined && padsNode.is(ConstantNode)) {
                    pads =
                        decodeIntegerVectorFromTensorProto(
                            padsNode.as(ConstantNode).constantValue,
                        ) ?? [];
                }

                const rank = dataShape.length;
                outShape = dataShape.slice();
                if (pads.length === 2 * rank) {
                    for (let i = 0; i < rank; i++) {
                        outShape[i] =
                            (toNum(outShape[i]) ?? 0) + (pads[i] ?? 0) + (pads[i + rank] ?? 0);
                    }
                }
                break;
            }

            case "ReduceSum":
            case "ReduceMean":
            case "ReduceProd":
            case "ReduceMin":
            case "ReduceMax":
            case "ReduceL1":
            case "ReduceL2":
            case "ReduceLogSum":
            case "ReduceSumSquare": {
                const inShape = infos[0]?.shape ?? [];
                const keepdims = (getAttr(node, "keepdims", 1) as number) !== 0;

                const axesAttr = getAttr(node, "axes", undefined) as number[] | number | undefined;
                let axes: number[] | undefined = Array.isArray(axesAttr)
                    ? axesAttr.map(Number)
                    : typeof axesAttr === "number"
                      ? [Number(axesAttr)]
                      : undefined;

                if (!axes) {
                    const axesNode: ValueNode | undefined = inputs[1];
                    if (axesNode !== undefined && axesNode.is(ConstantNode)) {
                        const raw =
                            decodeIntegerVectorFromTensorProto(
                                axesNode.as(ConstantNode).constantValue,
                            ) ?? [];
                        if (raw.length > 0) {
                            axes = raw;
                        }
                    }
                }

                if (!axes || axes.length === 0) {
                    outShape = keepdims ? inShape.map(() => 1) : [];
                } else {
                    const rank = inShape.length;
                    const norm = new Set(axes.map((a) => normalizeAxis(a, rank)));
                    outShape = keepdims
                        ? inShape.map((d, i) => (norm.has(i) ? 1 : d))
                        : inShape.filter((_, i) => !norm.has(i));
                }
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── ReduceLogSumExp ───── */
            case "ReduceLogSumExp": {
                const dataShape = infos[0]?.shape ?? [];
                const rank = dataShape.length;

                if (rank === 0) {
                    outShape = [];
                    outDtype = infos[0]?.dtype ?? outDtype;
                    break;
                }

                const attrs = node.getAttributes();
                let axes: number[] | undefined = attrs["axes"] as number[] | undefined;

                if (!Array.isArray(axes) || axes.length === 0) {
                    axes = Array.from({ length: rank }, (_, i) => i);
                } else {
                    axes = axes.map((a: number) => (a < 0 ? ((a % rank) + rank) % rank : a));
                }

                const keepdims = Number(attrs["keepdims"] ?? 1);

                if (keepdims) {
                    outShape = dataShape.slice();
                    for (const ax of axes) {
                        outShape[ax] = 1;
                    }
                } else {
                    const axeSet = new Set(axes);
                    outShape = dataShape.filter((_, i) => !axeSet.has(i));
                }

                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── ArgMax / ArgMin ───── */
            case "ArgMax":
            case "ArgMin": {
                const inShape = infos[0]?.shape ?? [];
                const keepdims = (getAttr(node, "keepdims", 1) as number) !== 0;
                const axis = normalizeAxis(getAttr(node, "axis", 0) as number, inShape.length);
                if (keepdims) {
                    outShape = inShape.map((d, i) => (i === axis ? 1 : d));
                } else {
                    outShape = inShape.filter((_, i) => i !== axis);
                }
                outDtype = DataType.INT64;
                break;
            }

            /** ───── Shape (with start/end/ to) ───── */
            case "Shape": {
                const inputShape = infos[0]?.shape ?? [];
                const rank = inputShape.length;
                if (rank === 0) {
                    outShape = [0];
                    outDtype = DataType.INT64;
                    break;
                }

                const attrs = node.getAttributes();
                const hasStart = Object.prototype.hasOwnProperty.call(attrs, "start");
                const hasEnd = Object.prototype.hasOwnProperty.call(attrs, "end");

                let start = hasStart ? Number(attrs["start"]) : 0;
                let end = hasEnd ? Number(attrs["end"]) : rank;

                const norm = (idx: number, r: number) => (r > 0 ? ((idx % r) + r) % r : 0);

                start = norm(start, rank);
                end = norm(end, rank);

                start = Math.max(0, Math.min(start, rank));
                end = Math.max(0, Math.min(end, rank));

                let length = Math.max(0, end - start);

                if (!hasStart && !hasEnd && length === 0 && rank > 0) {
                    length = rank;
                }

                outShape = [length];

                const toAttr = attrs["to"];
                if (typeof toAttr === "number") {
                    outDtype = toAttr as number;
                } else {
                    outDtype = DataType.INT64;
                }
                break;
            }

            /** ───── ConstantOfShape (with Shape(X) fallback) ───── */
            case "ConstantOfShape": {
                const shapeTensor: ValueNode | undefined = inputs[0]; // BaseNode
                let shape: Shape = [];

                if (shapeTensor !== undefined && shapeTensor.is(ConstantNode)) {
                    const arr =
                        decodeIntegerVectorFromTensorProto(
                            shapeTensor.as(ConstantNode).constantValue,
                        ) ?? [];
                    if (arr.length) {
                        shape = arr;
                    }
                }

                if (!shape.length) {
                    const producers = shapeTensor!.incomers.sources;
                    const shapeOp = producers
                        .filterIs(OperationNode)
                        .filter((op) => op.type === "Shape")
                        .first();

                    if (shapeOp) {
                        const shapeInputs = shapeOp.getInputs() ?? [];
                        const xTensor = shapeInputs[0];
                        const xShape = resolveTensorShape(xTensor);
                        if (xShape.length) shape = xShape.slice();
                    }
                }

                // if we still don't know the shape, reuse the existing output shape
                if (!shape.length) {
                    const outs = node.getOutgoers.targets;
                    const outT = outs
                        .filter((t) => t.is(TensorNode))
                        .first()
                        ?.as(TensorNode);

                    if (outT && Array.isArray(outT.shape) && outT.shape.length > 0) {
                        shape = [...outT.shape];
                    }
                }

                outShape = shape;
                break;
            }

            /** ───── OneHot ───── */
            case "OneHot": {
                const indicesShape = infos[0]?.shape ?? [];

                const depthTensorNode: ValueNode | undefined = inputs[1];
                let depth = 0;

                if (depthTensorNode !== undefined && depthTensorNode.is(ConstantNode)) {
                    const depthArr =
                        decodeIntegerVectorFromTensorProto(
                            depthTensorNode.as(ConstantNode).constantValue,
                        ) ?? [];
                    depth = depthArr[0] ?? 0;
                }

                if (indicesShape.length > 0) {
                    outShape = depth > 0 ? [...indicesShape, depth] : [...indicesShape, 1];
                } else {
                    outShape = depth > 0 ? [depth] : [];
                }

                const valuesTensor: ValueNode | undefined = inputs[2];
                if (valuesTensor !== undefined && valuesTensor.is(ConstantNode)) {
                    outDtype = valuesTensor.as(ConstantNode).literalType;
                } else if (valuesTensor !== undefined && valuesTensor.is(TensorNode)) {
                    outDtype = valuesTensor.as(TensorNode).literalType;
                } else {
                    outDtype = DataType.FLOAT;
                }
                break;
            }

            /** ───── Concat ───── */
            case "Concat": {
                const axis = getAttr(node, "axis", 0) as number;
                const inputShapes = infos.map((i) => i.shape);
                const ref = inputShapes.find((s) => s.length) ?? [];
                outShape = [...ref];
                outShape[axis] = inputShapes.reduce((sum, s) => sum + (toNum(s[axis]) ?? 0), 0);
                break;
            }

            /** ───── Flatten ───── */
            case "Flatten": {
                const inputShape = infos[0]?.shape ?? [];
                const axis = getAttr(node, "axis", 1) as number;
                const d0 = inputShape
                    .slice(0, axis)
                    .reduce((a, b) => (toNum(a) ?? 1) * (toNum(b) ?? 1), 1);
                const d1 = inputShape
                    .slice(axis)
                    .reduce((a, b) => (toNum(a) ?? 1) * (toNum(b) ?? 1), 1);
                outShape = [d0, d1];
                break;
            }

            /** ───── Expand (with Expand(x, Shape(X)) pattern) ───── */
            case "Expand": {
                const dataShape = infos[0]?.shape ?? [];
                const shapeInput: ValueNode | undefined = inputs[1]; // BaseNode
                let targetShape: Shape = [];

                if (shapeInput !== undefined && shapeInput.is(ConstantNode)) {
                    const arr =
                        decodeIntegerVectorFromTensorProto(
                            shapeInput.as(ConstantNode).constantValue,
                        ) ?? [];
                    if (arr.length) {
                        targetShape = arr;
                    }
                }

                const producers = shapeInput!.incomers.sources;
                const shapeOp = producers
                    .filterIs(OperationNode)
                    .filter((op) => op.type === "Shape")
                    .first();

                if (shapeOp) {
                    const shapeInputs = shapeOp.getInputs() ?? [];
                    const xTensor = shapeInputs[0];
                    const xShape = resolveTensorShape(xTensor);
                    if (xShape.length) targetShape = xShape.slice();
                }

                if (targetShape.length > 0) {
                    outShape = targetShape;
                } else if (dataShape.length > 0) {
                    outShape = dataShape.slice();
                } else {
                    outShape = [];
                }
                break;
            }

            /** ───── Conv (NCHW) with pads/auto_pad/dilations ───── */
            case "Conv": {
                const xShape = infos[0]?.shape ?? [];
                const wShape = infos[1]?.shape ?? [];

                if (xShape.length === 0 || wShape.length === 0) {
                    const first = infos.find((i) => i.shape.length);
                    if (first) {
                        outShape = first.shape.slice();
                        outDtype = first.dtype;
                    }
                    break;
                }

                if (xShape.length !== 4 || wShape.length !== 4) {
                    outShape = xShape.slice();
                    outDtype = infos[0]?.dtype ?? outDtype;
                    break;
                }

                const [N, , H, W] = xShape.map(toNum) as number[];
                const [M, , kH, kW] = wShape.map(toNum) as number[];

                const attrs = node.getAttributes();

                let strides = attrs["strides"] as number[] | undefined;
                if (!Array.isArray(strides) || strides.length !== 2) {
                    strides = [1, 1];
                }
                const [sH, sW] = strides.map(Number);

                let dilations = attrs["dilations"] as number[] | undefined;
                if (!Array.isArray(dilations) || dilations.length !== 2) {
                    dilations = [1, 1];
                }
                const [dH, dW] = dilations.map(Number);

                const pads = attrs["pads"] as number[] | undefined;
                let padTop = 0,
                    padLeft = 0,
                    padBottom = 0,
                    padRight = 0;

                const autoPad = (attrs["auto_pad"] ?? "NOTSET") as string;

                if (Array.isArray(pads) && pads.length === 4) {
                    [padTop, padLeft, padBottom, padRight] = pads.map(Number);
                } else if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
                    const kEffH = dH * (kH - 1) + 1;
                    const kEffW = dW * (kW - 1) + 1;

                    const outH = Math.ceil(H / sH);
                    const outW = Math.ceil(W / sW);

                    const totalPadH = Math.max(0, (outH - 1) * sH + kEffH - H);
                    const totalPadW = Math.max(0, (outW - 1) * sW + kEffW - W);

                    if (autoPad === "SAME_UPPER") {
                        padTop = Math.floor(totalPadH / 2);
                        padBottom = totalPadH - padTop;
                        padLeft = Math.floor(totalPadW / 2);
                        padRight = totalPadW - padLeft;
                    } else {
                        padBottom = Math.floor(totalPadH / 2);
                        padTop = totalPadH - padBottom;
                        padRight = Math.floor(totalPadW / 2);
                        padLeft = totalPadW - padRight;
                    }
                }

                const kEffH = dH * (kH - 1) + 1;
                const kEffW = dW * (kW - 1) + 1;

                const H_padded = H + padTop + padBottom;
                const W_padded = W + padLeft + padRight;

                const H_out = Math.floor((H_padded - kEffH) / sH + 1);
                const W_out = Math.floor((W_padded - kEffW) / sW + 1);

                outShape = [N, M, H_out, W_out];
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── MaxPool / AveragePool ───── */
            case "MaxPool":
            case "AveragePool": {
                const x = toStaticShape(infos[0]?.shape);
                const n = x[0],
                    c = x[1],
                    h = x[2],
                    wdim = x[3];
                const kernel =
                    (getAttr(node, "kernel_shape", [1, 1]) as number[] | undefined) ?? [];
                const strides = (getAttr(node, "strides", [1, 1]) as number[] | undefined) ?? [];
                const pads = (getAttr(node, "pads", [0, 0, 0, 0]) as number[] | undefined) ?? [];
                const dil = (getAttr(node, "dilations", [1, 1]) as number[] | undefined) ?? [];

                const Hout = inferPoolDim(
                    h,
                    kernel[0] ?? 1,
                    strides[0] ?? 1,
                    pads[0] ?? 0,
                    pads[2] ?? 0,
                    dil[0] ?? 1,
                );
                const Wout = inferPoolDim(
                    wdim,
                    kernel[1] ?? 1,
                    strides[1] ?? 1,
                    pads[1] ?? 0,
                    pads[3] ?? 0,
                    dil[1] ?? 1,
                );
                outShape = [n, c, Hout, Wout];
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── BatchNormalization (shape preserved) ───── */
            case "BatchNormalization": {
                outShape = infos[0]?.shape ?? [];
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── Cast (dtype changes, shape preserved) ───── */
            case "Cast": {
                outShape = infos[0]?.shape ?? [];
                outDtype = getAttr(node, "to", outDtype) as number;
                break;
            }

            /** ───── Softmax (shape preserved) ───── */
            case "Softmax": {
                outShape = infos[0]?.shape ?? [];
                outDtype = infos[0]?.dtype ?? outDtype;
                break;
            }

            /** ───── Range (1D) ───── */
            case "Range": {
                const getScalar = (inp: ValueNode | undefined): number | undefined => {
                    if (inp !== undefined && inp.is(ConstantNode)) {
                        const proto: TensorProto = inp.as(ConstantNode).constantValue;

                        // 1. Try parsed arrays first
                        if (proto.floatData && proto.floatData.length > 0)
                            return proto.floatData[0];
                        if (proto.doubleData && proto.doubleData.length > 0)
                            return proto.doubleData[0];
                        if (proto.int64Data && proto.int64Data.length > 0)
                            return Number(proto.int64Data[0]);
                        if (proto.int32Data && proto.int32Data.length > 0)
                            return proto.int32Data[0];

                        // 2. Fallback to existing integer decoder
                        const arr = decodeIntegerVectorFromTensorProto(proto);
                        if (arr && arr.length > 0) return arr[0];
                    }
                    return undefined;
                };

                const start = getScalar(inputs[0]);
                const end = getScalar(inputs[1]);
                const step = getScalar(inputs[2]);

                if (start !== undefined && end !== undefined && step !== undefined && step !== 0) {
                    const len = Math.max(0, Math.ceil((end - start) / step));
                    outShape = [len];
                } else {
                    outShape = []; // Fallback to unknown shape
                }
                break;
            }

            /** ───── Loop  ───── */
            case "Loop": {
                const initState = infos[2];
                outShape = initState.shape.slice();
                outDtype = initState.dtype;
                break;
            }

            default: {
                const first = infos.find((i) => typeof i.shape !== "undefined");
                if (first) {
                    outShape = first.shape;
                    outDtype = first.dtype;
                }
            }
        }

        // Rewire edges with updated shapes/dtypes
        const outputs = node.getOutgoers.targets;
        const outputTensors = outputs.filter((t) => t.is(TensorNode));

        node.getOutgoers.forEach((e) => graph.getEdgeById(e.id)?.remove());

        for (const output of outputs) {
            // If we inferred a specific shape above (standard ops), apply it.
            // If it was a Loop/If, we likely already updated the tensor node directly,
            // so we pull that back.
            let finalShape = outShape;
            let finalDtype = outDtype;

            if (node.type === "Loop" || node.type === "If") {
                if (output.is(TensorNode)) {
                    finalShape = output.as(TensorNode).shape;
                    finalDtype = output.as(TensorNode).literalType;
                }
            }

            graph.addEdge(node, output).init(new OnnxEdge.Builder(finalDtype, finalShape));
        }

        if (Array.isArray(outShape) && outShape.length > 0) {
            for (const out of outputTensors) {
                const tn = out.tryAs(TensorNode);
                if (!tn) continue;
                // Only update if we actually calculated something new or if it's intermediate
                if (node.type !== "Loop" && node.type !== "If") {
                    tn.setShape(outShape);
                    tn.setLiteralType(outDtype);
                }
            }
        }
    }
}
