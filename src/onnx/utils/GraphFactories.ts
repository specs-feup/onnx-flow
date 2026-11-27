import ConstantNode from "../ir/ConstantNode.js";
import type OnnxGraph from "../ir/OnnxGraph.js";
import type { StaticShape, TensorProto, ValueNode } from "../ir/OnnxTypes.js";
import { DataType } from "../ir/OnnxTypes.js";
import OperationNode from "../ir/OperationNode.js";
import { makeTensorProto } from "./TensorData.js";

const uniqNextIdxCache = new WeakMap<OnnxGraph.Class, Map<string, number>>();
const regionScopeCache = new WeakMap<OnnxGraph.Class, string>();
let regionScopeCounter = 0;

/** Assigns (or retrieves) a unique scope prefix tag for a nested subgraph region. */
export function assignRegionScopeTag(g: OnnxGraph.Class, prefix: string = "canon"): string {
    let tag = regionScopeCache.get(g);
    if (tag === undefined) {
        tag = `${prefix}_r${++regionScopeCounter}`;
        regionScopeCache.set(g, tag);
    }
    return tag;
}

/** Returns the scope tag previously assigned to `g`, or `fallback` if untagged. */
export function getRegionScopeTag(g: OnnxGraph.Class, fallback: string): string {
    return regionScopeCache.get(g) ?? fallback;
}

/** Generates a node ID guaranteed to be unique within `g`, starting from `base`. */
export function uniq(g: OnnxGraph.Class, base: string): string {
    if (!g.hasNode(base)) return base;
    let cache = uniqNextIdxCache.get(g);
    if (!cache) {
        cache = new Map<string, number>();
        uniqNextIdxCache.set(g, cache);
    }
    let i = cache.get(base) ?? 0;
    let id = i > 0 ? `${base}_${i}` : base;
    while (g.hasNode(id)) id = `${base}_${++i}`;
    cache.set(base, i + 1);
    return id;
}

/** Create a ConstantNode for a rank-0 scalar of given type. */
export function scalarOfType(
    g: OnnxGraph.Class,
    name: string,
    v: number,
    dtype: DataType,
): ConstantNode.Class {
    const proto = makeTensorProto(dtype, [], [v]);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Create a ConstantNode for a rank-0 INT64 scalar. */
export function scalarI64(g: OnnxGraph.Class, name: string, v: number): ConstantNode.Class {
    return scalarOfType(g, name, v, DataType.INT64);
}

/** Create a ConstantNode for a rank-0 scalar zero. */
export function scalarZeroOfType(
    g: OnnxGraph.Class,
    name: string,
    dtype: DataType,
): ConstantNode.Class {
    return scalarOfType(g, name, 0, dtype);
}

/** Create a ConstantNode for a 1D INT64 vector. */
export function constI64(g: OnnxGraph.Class, name: string, vals: number[]): ConstantNode.Class {
    const proto = makeTensorProto(DataType.INT64, [vals.length], vals);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Create a ConstantNode for a 1D FLOAT vector. */
export function constF32(g: OnnxGraph.Class, name: string, vals: number[]): ConstantNode.Class {
    const proto = makeTensorProto(DataType.FLOAT, [vals.length], vals);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Create a ConstantNode filled with ones. */
export function tensorOnesConst(
    g: OnnxGraph.Class,
    name: string,
    dtype: DataType,
    shape: StaticShape,
): ConstantNode.Class {
    const size = shape.reduce((a, b) => a * b, 1);
    const ones = new Array<number>(size).fill(1);
    const proto = makeTensorProto(dtype, shape, ones);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Generic helper to create a ConstantNode from a TensorProto. */
export function makeTensorConst(
    g: OnnxGraph.Class,
    id: string,
    proto: TensorProto,
): ConstantNode.Class {
    return g.addNode(uniq(g, id)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

/** Gets the OperationNode that produced this ValueNode, if any. */
export function getProducer(node: ValueNode): OperationNode.Class | undefined {
    return node.incomers.sources.first()?.tryAs(OperationNode);
}

/** Gets all OperationNodes that consume this ValueNode. */
export function getConsumers(node: ValueNode): OperationNode.Class[] {
    return node.outgoers.targets.filterIs(OperationNode).toArray();
}
