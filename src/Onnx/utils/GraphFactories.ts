import ConstantNode from "../ConstantNode.js";
import type OnnxGraph from "../OnnxGraph.js";
import type { StaticShape, TensorProto, ValueNode, ConcreteValueNode } from "../OnnxTypes.js";
import { DataType } from "../OnnxTypes.js";
import OperationNode from "../OperationNode.js";
import TensorNode from "../TensorNode.js";
import { addEdge } from "./GraphQueries.js";
import { makeTensorProto } from "./TensorData.js";

export function uniq(g: OnnxGraph.Class, base: string): string {
    let i = 0,
        id = base;
    while (g.hasNode(id)) id = `${base}_${++i}`;
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

export function makeValueScalar1(
    g: OnnxGraph.Class,
    name: string,
    dtype: DataType,
    v: number,
): ConstantNode.Class {
    const proto = makeTensorProto(dtype, [1], [v]);
    return g.addNode(uniq(g, name)).init(new ConstantNode.Builder(proto)).as(ConstantNode);
}

export function makeI64ShapeConst(
    g: OnnxGraph.Class,
    name: string,
    vals: number[],
): ConstantNode.Class {
    return constI64(g, name, vals);
}

// Helper to create a scalar ConstantNode Builder
export const constBuilder = (val: number): ConstantNode.Builder => {
    return new ConstantNode.Builder(makeTensorProto(DataType.INT64, [], [val]));
};

/** Gets the OperationNode that produced this ValueNode, if any. */
export function getProducer(node: ValueNode): OperationNode.Class | undefined {
    return node.incomers.sources.first()?.tryAs(OperationNode);
}

/** Gets all OperationNodes that consume this ValueNode. */
export function getConsumers(node: ValueNode): OperationNode.Class[] {
    return node.outgoers.targets.filterIs(OperationNode).toArray();
}

/** Creates a Shape op + intermediate output tensor. */
export function shapeOf(g: OnnxGraph.Class, x: ConcreteValueNode, name: string): TensorNode.Class {
    const sop = g
        .addNode(uniq(g, `${name}_op`))
        .init(new OperationNode.Builder("Shape", [x], {}))
        .as(OperationNode);
    const s = g
        .addNode(uniq(g, `${name}`))
        .init(new TensorNode.Builder(DataType.INT64, [x.shape.length], "intermediate"))
        .as(TensorNode);
    addEdge(g, sop, s, DataType.INT64, [x.shape.length]);
    return s;
}

/** Creates ScatterElements to edit a specific dimension of a shape tensor. */
export function editShapeDim(
    g: OnnxGraph.Class,
    baseShape: TensorNode.Class,
    axis: number,
    size1D: ConcreteValueNode,
    name: string,
): TensorNode.Class {
    const idx = makeI64ShapeConst(g, `${name}_idx`, [axis]);

    const shapeOne = makeI64ShapeConst(g, `${name}_vec_shape`, [1]);

    const reshapeOp = g
        .addNode(uniq(g, `${name}_ensure_vec_op`))
        .init(new OperationNode.Builder("Reshape", [size1D, shapeOne]))
        .as(OperationNode);

    const updateVec = g
        .addNode(uniq(g, `${name}_ensure_vec`))
        .init(new TensorNode.Builder(DataType.INT64, [1], "intermediate"))
        .as(TensorNode);

    addEdge(g, reshapeOp, updateVec, DataType.INT64, [1]);

    const sc = g
        .addNode(uniq(g, `${name}_sc`))
        .init(
            new OperationNode.Builder("ScatterElements", [baseShape, idx, updateVec], { axis: 0 }),
        )
        .as(OperationNode);
    const out = g
        .addNode(uniq(g, `${name}_out`))
        .init(
            new TensorNode.Builder(DataType.INT64, [baseShape.shape[0] as number], "intermediate"),
        )
        .as(TensorNode);
    addEdge(g, sc, out, DataType.INT64, [baseShape.shape[0] as number]);
    return out;
}
