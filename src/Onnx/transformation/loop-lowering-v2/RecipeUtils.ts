import type OnnxGraph from "../../OnnxGraph.js";
import type { ValueNode, ConcreteValueNode, KnownShape, Shape } from "../../OnnxTypes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../../ConstantNode.js";
import TensorNode from "../../TensorNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";
import {
    toStaticShape,
    makeTensorConst,
    scalarInt64,
    computeStrides,
    uniq,
    int64Vec,
} from "../../Utils.js";
import { GraphBuilder } from "../../GraphBuilder.js";

/**
 * Decodes a linear iteration index into a multi-index based on output radices.
 */
export function decodeMixedRadix(
    builder: GraphBuilder,
    iter: ConcreteValueNode,
    dims: number[],
    tag: string,
): ConcreteValueNode[] {
    const dd = dims.map((d) => (d > 0 ? d : 1));
    const out: ConcreteValueNode[] = [];
    let rem = iter;

    for (let k = dd.length - 1; k >= 0; k--) {
        const dConst = builder.createConstant(`mr_dim_${tag}_${k}`, scalarInt64(dd[k]));

        // modOut = rem % dConst
        const [modOut] = builder.createOp("Mod", [rem, dConst]);
        out.unshift(modOut);

        // nextRem = rem / dConst
        const [divOut] = builder.createOp("Div", [rem, dConst]);
        rem = divOut;
    }
    return out;
}

/**
 * Builds a linear index from a multi-index and strides.
 */
export function buildLinearIndex(
    builder: GraphBuilder,
    idx: ConcreteValueNode[],
    strides: number[],
    tag: string,
): ConcreteValueNode {
    let acc: ConcreteValueNode = builder.createConstant(`lin_zero_${tag}`, scalarInt64(0));

    for (let i = 0; i < idx.length; i++) {
        const sConst = builder.createConstant(`lin_stride_${tag}_${i}`, scalarInt64(strides[i]));
        const [mulOut] = builder.createOp("Mul", [idx[i], sConst]);
        const [addOut] = builder.createOp("Add", [acc, mulOut]);
        acc = addOut;
    }
    return acc;
}

/**
 * Ensures an input is 1D by reshaping it if necessary.
 */
export function ensureFlatInput(builder: GraphBuilder, t: ValueNode): ValueNode {
    if (t.shape.length <= 1) return t;

    const shapeConst = builder.createConstant(`flat_shape_${t.id}`, int64Vec([-1]));
    const [flat] = builder.createOp("Reshape", [t, shapeConst]);
    return flat;
}

/**
 * Wraps an outer node as a RegionArgument (implicit capture).
 */
export function createCapturedInput(
    body: OnnxGraph.Class,
    outerNode: ValueNode,
): RegionArgumentNode.Class {
    if (body.hasNode(outerNode.id)) {
        const existing = body.getNodeById(outerNode.id);
        if (existing?.is(RegionArgumentNode)) return existing.as(RegionArgumentNode);
    }

    let originalName = outerNode.id;
    if (outerNode.is(RegionArgumentNode)) {
        originalName = outerNode.as(RegionArgumentNode).originalName;
    }

    return body
        .addNode(outerNode.id)
        .init(
            new RegionArgumentNode.Builder(0, originalName, outerNode.literalType, outerNode.shape),
        )
        .as(RegionArgumentNode);
}

/**
 * Standard resolve input logic for recipes, using GraphBuilder for all internal wiring.
 */
export function resolveRecipeInput(
    builder: GraphBuilder,
    input: BaseNode.Class,
    valueMap: Map<string, ValueNode>,
    iter: ConcreteValueNode,
    axes: ConcreteValueNode,
    outShape: KnownShape,
    flatten: boolean = true,
    returnGather: boolean = true,
): ValueNode {
    if (valueMap.has(input.id)) {
        return valueMap.get(input.id)!;
    }

    let tOuter: ValueNode;
    if (input.is(TensorNode)) tOuter = input.as(TensorNode);
    else if (input.is(ConstantNode)) tOuter = input.as(ConstantNode);
    else if (input.is(RegionArgumentNode)) tOuter = input.as(RegionArgumentNode);
    else throw new Error(`Unhandled input case for ${input.id}`);

    let tInner: ValueNode;
    if (builder.graph.hasNode(tOuter.id)) {
        tInner = builder.graph.getNodeById(tOuter.id) as ValueNode;
    } else {
        if (tOuter.is(ConstantNode)) {
            const c = tOuter.as(ConstantNode);
            tInner = builder.createConstant(c.id, c.constantValue);
        } else {
            tInner = createCapturedInput(builder.graph, tOuter);
        }
    }

    if (!returnGather || (tInner.shape !== undefined && tInner.shape.length === 0)) {
        return flatten ? ensureFlatInput(builder, tInner) : tInner;
    }

    const inDimsStatic = toStaticShape(tInner.shape as Shape);
    const outDimsStatic = toStaticShape(outShape as Shape);

    const shapeKey = `__idx_cache_${inDimsStatic.join(",")}_to_${outDimsStatic.join(",")}`;
    let linU: ConcreteValueNode;

    if (valueMap.has(shapeKey)) {
        linU = valueMap.get(shapeKey) as ConcreteValueNode;
    } else {
        const rO = outDimsStatic.length;
        const rI = inDimsStatic.length;
        const outRadix = outDimsStatic.map((d) => (d > 0 ? d : 1));
        const inRadix = inDimsStatic.map((d) => (d > 0 ? d : 1));

        const oDigits = decodeMixedRadix(builder, iter, outRadix, `gb_out_${shapeKey}`);
        const iDigits: ConcreteValueNode[] = [];

        for (let k = 0; k < rI; k++) {
            const inDim = inRadix[k];
            const outPos = rO - rI + k;
            if (outPos < 0 || inDim === 1) {
                iDigits.push(builder.createConstant(`gb_zero_${shapeKey}_${k}`, scalarInt64(0)));
            } else {
                iDigits.push(oDigits[outPos]);
            }
        }

        const strides = computeStrides(inRadix);
        const linScalar = buildLinearIndex(builder, iDigits, strides, `gb_lin_${shapeKey}`);
        [linU] = builder.createOp("Unsqueeze", [linScalar, axes]);

        valueMap.set(shapeKey, linU);
    }

    const flatT = ensureFlatInput(builder, tInner);
    const [gathered] = builder.createOp("Gather", [flatT, linU], { axis: 0 });
    return squeezeIfLen1(builder, gathered, axes, `gb_sq_${tInner.id}`);
}

export function squeezeIfLen1(
    builder: GraphBuilder,
    t: ValueNode,
    axes: ConcreteValueNode,
    tag: string,
): ValueNode {
    if (t.shape.length === 1 && t.shape[0] === 1) {
        const [out] = builder.createOp("Squeeze", [t, axes]);
        return out;
    }
    return t;
}
