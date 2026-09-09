import OnnxEdge from "../OnnxEdge.js";
import type OnnxGraph from "../OnnxGraph.js";
import type { Dim, ValueNode, Shape, StaticShape, ConcreteValueNode } from "../OnnxTypes.js";
import { DataType } from "../OnnxTypes.js";
import OperationNode from "../OperationNode.js";
import TensorNode from "../TensorNode.js";
import { UNKNOWN_SHAPE } from "./Constants.js";
import { makeTensorConst, uniq } from "./GraphFactories.js";
import { int64Vec } from "./TensorData.js";

export function isNumeric(dtype: DataType): boolean {
    return !(dtype === DataType.STRING || dtype === DataType.BOOL);
}

export function toNum(x: Dim): number | undefined {
    if (typeof x === "number") return x;
    if (typeof x === "string") {
        const n = Number(x);
        return isNaN(n) ? undefined : n;
    }
    return undefined;
}

export function toScalar(g: OnnxGraph.Class, t: ValueNode, tag: string): TensorNode.Class {
    if (t.shape.length === 0 && t.is(TensorNode)) return t;
    const shapeConst = makeTensorConst(g, uniq(g, `${tag}_shape`), int64Vec([]));
    const reshape = g
        .addNode(uniq(g, `${tag}_reshape`))
        .init(new OperationNode.Builder("Reshape", [t, shapeConst]))
        .as(OperationNode);
    const out = g
        .addNode(uniq(g, `${tag}_out`))
        .init(new TensorNode.Builder(t.literalType, [], "intermediate"))
        .as(TensorNode);
    g.addEdge(reshape, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
}

export function toNumShape(s?: Array<Dim>): Array<number | undefined> | undefined {
    if (!s) return undefined;
    return s.map(toNum);
}

export function asStaticDims(shape?: Shape): StaticShape {
    if (!shape || !Array.isArray(shape)) return [];
    return shape.map((d) => {
        const n = toNum(d);
        return n !== undefined && n > 0 ? n : 1;
    });
}

export function isKnownDim(d: number | undefined): boolean {
    return typeof d === "number" && Number.isFinite(d) && d > 0;
}

export function toStaticShape(shape?: Shape): StaticShape {
    if (!shape || !Array.isArray(shape)) return [];
    return shape.map((d) => {
        const n = toNum(d);
        return n !== undefined ? n : UNKNOWN_SHAPE[0];
    });
}

/**
 * @brief Checks if two tensor nodes have the same shape.
 *
 * @param tensor1 The first tensor node to compare.
 * @param tensor2 The second tensor node to compare.
 * @returns True if the shapes are equal, false otherwise.
 */
export function shapesEqual(tensor1: ConcreteValueNode, tensor2: ConcreteValueNode): boolean {
    if (tensor1.shape.length !== tensor2.shape.length) {
        return false;
    }

    for (let i = 0; i < tensor1.shape.length; i++) {
        if (tensor1.shape[i] !== tensor2.shape[i]) {
            return false;
        }
    }

    return true;
}

/**
 * Resolves the shape of a node, strictly coercing all dimensions to numbers.
 * Handles TensorNode, ConstantNode, and RegionArgumentNode.
 * * - Tries the node's own .shape property first.
 * - If empty/missing, tries incoming/outgoing edges (for TensorNodes).
 * - Converts all dimensions to numbers.
 * - Non-finite or <= 0 values (like "batch" or -1) are coerced to 1 for safety in loop bounds.
 */
export function resolveShapeToNumbers(t: ValueNode): StaticShape {
    let rawShape: Shape = [];

    // 1. Try internal shape
    if (t.shape.length > 0) {
        rawShape = t.shape;
    }
    // 2. Try edges if it's a TensorNode
    else if (t.is(TensorNode)) {
        const tn = t.as(TensorNode);
        const incs = tn.getIncomers;
        // Try incoming edges
        for (const e of incs) {
            if (e.shape.length) {
                rawShape = e.shape;
                tn.setShape(rawShape); // Cache it
                break;
            }
        }
        // Fallback: try outgoing edges
        if (!rawShape.length) {
            const outs = tn.getOutgoers;
            for (const e of outs) {
                if (e.shape.length) {
                    rawShape = e.shape;
                    tn.setShape(rawShape); // Cache it
                    break;
                }
            }
        }
    }

    // 3. Strict conversion
    return rawShape.map((d) => {
        const n = Number(d);
        // Treat NaN/strings/<=0 as 1 for safety
        return Number.isFinite(n) && n > 0 ? n : 1;
    });
}

export function prodSafe(dims: StaticShape): number {
    return dims.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

export function computeStrides(dims: StaticShape): StaticShape {
    const n = dims.length;
    const strides = new Array(n);
    let acc = 1;
    for (let i = n - 1; i >= 0; --i) {
        const d = dims[i] > 0 ? dims[i] : 1;
        strides[i] = acc;
        acc *= d;
    }
    return strides;
}

export function normalizeAxis(axis: number, rank: number): number {
    if (rank <= 0) return axis;
    return ((axis % rank) + rank) % rank;
}

export function broadcastTwoShapes(a: StaticShape, b: StaticShape): StaticShape {
    const ra = a.length,
        rb = b.length;
    const r = Math.max(ra, rb);
    const out = new Array<number>(r);
    for (let i = 0; i < r; i++) {
        const da = a[ra - 1 - i] ?? 1;
        const db = b[rb - 1 - i] ?? 1;
        if (da === 1) out[r - 1 - i] = db;
        else if (db === 1) out[r - 1 - i] = da;
        else if (da === db) out[r - 1 - i] = da;
        else out[r - 1 - i] = Math.max(da, db);
    }
    return out;
}

export function broadcastShapes(...shapes: number[][]): number[] {
    return shapes.reduce((acc, s) => broadcastTwoShapes(acc, s), []);
}

/**
 * Infers the output dimension for a single spatial axis in a pooling operation.
 */
export function inferPoolDim(
    inDim: number,
    k: number,
    stride: number,
    padHead: number,
    padTail: number,
    dil: number,
    ceilMode: number = 0, // New parameter
): number {
    const effectiveK = dil * (k - 1) + 1;
    const value = (inDim + padHead + padTail - effectiveK) / stride + 1;

    // Ensure the output dimension is at least 1 if the input exists
    const outDim = ceilMode === 1 ? Math.ceil(value) : Math.floor(value);

    // Safety check: if ceil_mode is 1, the last pooling window
    // must start within the input + padding range.
    if (ceilMode === 1 && (outDim - 1) * stride >= inDim + padHead) {
        return outDim - 1;
    }

    return Math.max(0, outDim);
}

export function inferConvDim(
    inDim: number,
    k: number,
    stride: number,
    padHead: number,
    padTail: number,
    dil: number,
): number {
    return inferPoolDim(inDim, k, stride, padHead, padTail, dil);
}

export function getAttr(node: unknown, name: string, def?: unknown): unknown {
    const n = node as {
        getAttributes?: () => Record<string, unknown>;
        attributes?: Record<string, unknown>;
    };
    const v = n.getAttributes?.()[name] ?? n.attributes?.[name];
    return v === undefined ? def : v;
}

export function getIntAttr(node: OperationNode.Class, name: string, def: number): number {
    const val = getAttr(node, name, def);
    return typeof val === "number" ? val : def;
}

export function getFloatAttr(node: OperationNode.Class, name: string, def: number): number {
    const val = getAttr(node, name, def);
    return typeof val === "number" ? val : def; // ONNX floats parse as JS numbers
}

export function getStringAttr(node: OperationNode.Class, name: string, def: string): string {
    const val = getAttr(node, name, def);
    return typeof val === "string" ? val : def;
}

export function getIntsAttr(node: OperationNode.Class, name: string, def: number[]): number[] {
    const val = getAttr(node, name, def);
    return Array.isArray(val) ? (val as number[]) : def;
}

export function getFloatsAttr(node: OperationNode.Class, name: string, def: number[]): number[] {
    const val = getAttr(node, name, def);
    return Array.isArray(val) ? (val as number[]) : def;
}

export function getSmallestRankShape(tensors: TensorNode.Class[]): Shape {
    if (tensors.length === 0) return [];
    let smallest = tensors[0].shape;
    for (let i = 1; i < tensors.length; i++) {
        if (tensors[i].shape.length < smallest.length) smallest = tensors[i].shape;
    }
    return smallest;
}

export function getLargestRankShape(tensors: ValueNode[]): Shape {
    if (tensors.length === 0) return [];
    let largest = tensors[0].shape;
    for (let i = 1; i < tensors.length; i++) {
        if (tensors[i].shape.length > largest.length) largest = tensors[i].shape;
    }
    return largest;
}

export function unsqueezeIdx(
    g: OnnxGraph.Class,
    idx: ConcreteValueNode,
    axes: ConcreteValueNode,
    tag: string,
): TensorNode.Class {
    const unsq = g
        .addNode(uniq(g, tag))
        .init(new OperationNode.Builder("Unsqueeze", [idx, axes]))
        .as(OperationNode);
    const out = g
        .addNode(uniq(g, `${tag}_out`))
        .init(new TensorNode.Builder(idx.literalType, [1], "intermediate"))
        .as(TensorNode);
    g.addEdge(unsq, out).init(new OnnxEdge.Builder(out.literalType, out.shape)).as(OnnxEdge);
    return out;
}

export function as1D(
    g: OnnxGraph.Class,
    name: string,
    scalarI64T: ConcreteValueNode,
): TensorNode.Class {
    const axes = makeTensorConst(g, `axes_${name}`, int64Vec([0]));
    return unsqueezeIdx(g, scalarI64T, axes, `${name}_u`); // 1-D [1] from scalar
}
