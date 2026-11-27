import type { Dim, ValueNode, Shape, StaticShape, ConcreteValueNode } from "../ir/OnnxTypes.js";
import { DataType } from "../ir/OnnxTypes.js";
import type OperationNode from "../ir/OperationNode.js";
import TensorNode from "../ir/TensorNode.js";
import { UNKNOWN_SHAPE } from "./Constants.js";

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

/** Checks whether two concrete value nodes have identical shapes element-by-element. */
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
 * Resolves the shape of a value node, strictly coercing all dimensions to positive numbers
 * (treating symbolic or non-positive dims as 1 for safe loop bounds).
 */
export function resolveShapeToNumbers(t: ValueNode): StaticShape {
    let rawShape: Shape = [];

    if (t.shape.length > 0) {
        rawShape = t.shape;
    } else if (t.is(TensorNode)) {
        const tn = t.as(TensorNode);
        for (const e of tn.getIncomers) {
            if (e.shape.length > 0) {
                rawShape = e.shape;
                tn.setShape(rawShape);
                break;
            }
        }
        if (rawShape.length === 0) {
            for (const e of tn.getOutgoers) {
                if (e.shape.length > 0) {
                    rawShape = e.shape;
                    tn.setShape(rawShape);
                    break;
                }
            }
        }
    }

    return rawShape.map((d) => {
        const n = Number(d);
        return Number.isFinite(n) && n > 0 ? n : 1;
    });
}

export function prodSafe(dims: StaticShape): number {
    return dims.reduce((a, b) => a * (b > 0 ? b : 1), 1);
}

export function computeStrides(dims: StaticShape): StaticShape {
    const n = dims.length;
    const strides = new Array<number>(n);
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
    const ra = a.length;
    const rb = b.length;
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

/** Infers the output dimension for a single spatial axis in a pooling or convolution op. */
export function inferPoolDim(
    inDim: number,
    k: number,
    stride: number,
    padHead: number,
    padTail: number,
    dil: number,
    ceilMode: number = 0,
): number {
    const effectiveK = dil * (k - 1) + 1;
    const value = (inDim + padHead + padTail - effectiveK) / stride + 1;
    const outDim = ceilMode === 1 ? Math.ceil(value) : Math.floor(value);

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
    return typeof val === "number" ? val : def;
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
