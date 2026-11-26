import type OnnxGraph from "../../ir/OnnxGraph.js";
import type { ValueNode, ConcreteValueNode, KnownShape, Shape } from "../../ir/OnnxTypes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../../ir/ConstantNode.js";
import RegionArgumentNode from "../../ir/RegionArgumentNode.js";
import { asValueNode, int64Vec } from "../../utils/index.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import type { LoopCarryContext } from "./LoopLoweringRecipe.js";
import { resolveGatherScalarIndex } from "./LoopIndexing.js";

// Re-export cohesive sub-modules so all loop-lowering recipes and public API keep a single import point.
export {
    extractMixedDims,
    buildMixedStrides,
    mulMixed,
    productMixed,
    addMixed,
    packShapeVector,
    broadcastMixedShapes,
    computeMatMulOutputDimsMixed,
    captureMixedList,
    resolvePoolShapeAndPadsMixed,
} from "./MixedShapeMath.js";
export {
    decodeMixedRadix,
    buildLinearIndex,
    getOrCreateIterationSpaceCounters,
    buildAguPointer,
    type AguSpaceCounters,
} from "./LoopIndexing.js";

/**
 * Ensures an input from an outer graph is available inside `builder.graph` as a `ValueNode`
 * (reusing existing local nodes, cloning constants, or creating a `RegionArgumentNode` capture).
 */
export function ensureInnerValueNode(builder: GraphBuilder, d: ValueNode): ValueNode {
    if (builder.graph.hasNode(d.id)) {
        return asValueNode(builder.graph.getNodeById(d.id)!);
    }
    if (d.is(ConstantNode)) {
        const c = d.as(ConstantNode);
        return builder.createConstant(c.id, c.constantValue);
    }
    return createCapturedInput(builder.graph, d);
}

/**
 * Reads a single scalar (0D) element from a 1D tensor at a scalar 0D index,
 * avoiding Unsqueeze/Squeeze wrapper ops around `Gather`.
 */
export function gatherScalar(
    builder: GraphBuilder,
    flatTensor: ValueNode,
    scalarIdx: ConcreteValueNode,
): ConcreteValueNode {
    return builder.createOp("Gather", [flatTensor, scalarIdx], { axis: 0 })[0];
}

/** Ensures a tensor input is 1D by reshaping it to `[-1]` if its rank exceeds 1. */
export function ensureFlatInput(builder: GraphBuilder, t: ValueNode): ValueNode {
    if (!(t.shape as Shape | KnownShape | undefined) || t.shape.length <= 1) return t;
    const shapeConst = builder.createConstant(`flat_shape_${t.id}`, int64Vec([-1]));
    const [flat] = builder.createOp("Reshape", [t, shapeConst]);
    return flat;
}

/** Wraps an outer-scope value node as a `RegionArgumentNode` capture inside `body`. */
export function createCapturedInput(
    body: OnnxGraph.Class,
    outerNode: ValueNode,
): RegionArgumentNode.Class {
    if (body.hasNode(outerNode.id)) {
        const existing = body.getNodeById(outerNode.id);
        if (existing !== undefined && existing.is(RegionArgumentNode)) {
            return existing.as(RegionArgumentNode);
        }
    }

    const originalName = outerNode.is(RegionArgumentNode)
        ? outerNode.as(RegionArgumentNode).originalName
        : outerNode.id;

    return body
        .addNode(outerNode.id)
        .init(
            new RegionArgumentNode.Builder(0, originalName, outerNode.literalType, outerNode.shape),
        )
        .as(RegionArgumentNode);
}

/**
 * Resolves an op input inside a lowered `Loop` body, optionally hoisting its flattening
 * outside the loop and gathering the element for the current iteration (Pointer AGU or Radix).
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
    targetShapeNode?: ValueNode,
    loopCtx?: LoopCarryContext,
): ValueNode {
    if (valueMap.has(input.id)) {
        return valueMap.get(input.id)!;
    }

    const tOuter = asValueNode(input);
    const wantsFlat =
        flatten || (returnGather && ((tOuter.shape as Shape | undefined)?.length ?? 0) > 0);

    if (
        wantsFlat &&
        loopCtx &&
        !tOuter.is(RegionArgumentNode) &&
        ((tOuter.shape as Shape | undefined)?.length ?? 0) > 1 &&
        loopCtx.outerBuilder.graph.hasNode(tOuter.id)
    ) {
        const flatOuterKey = `__flat_outer_${tOuter.id}`;
        let flatOuter = valueMap.get(flatOuterKey);
        if (!flatOuter) {
            flatOuter = ensureFlatInput(loopCtx.outerBuilder, tOuter);
            valueMap.set(flatOuterKey, flatOuter);
        }
        const tInnerFlat = builder.graph.hasNode(flatOuter.id)
            ? asValueNode(builder.graph.getNodeById(flatOuter.id)!)
            : createCapturedInput(builder.graph, flatOuter);
        if (!returnGather) {
            valueMap.set(input.id, tInnerFlat);
            return tInnerFlat;
        }

        const linScalar = resolveGatherScalarIndex(
            builder,
            tOuter,
            valueMap,
            iter,
            axes,
            outShape,
            targetShapeNode,
            loopCtx,
        );
        const finalScalar = gatherScalar(builder, tInnerFlat, linScalar);
        valueMap.set(input.id, finalScalar);
        return finalScalar;
    }

    const tInner = ensureInnerValueNode(builder, tOuter);

    if (
        !returnGather ||
        ((tInner.shape as Shape | KnownShape | undefined) && tInner.shape.length === 0)
    ) {
        const res = flatten ? ensureFlatInput(builder, tInner) : tInner;
        valueMap.set(input.id, res);
        return res;
    }

    const linScalar = resolveGatherScalarIndex(
        builder,
        tInner,
        valueMap,
        iter,
        axes,
        outShape,
        targetShapeNode,
        loopCtx,
    );
    const flatT = ensureFlatInput(builder, tInner);
    const finalScalar = gatherScalar(builder, flatT, linScalar);

    valueMap.set(input.id, finalScalar);
    return finalScalar;
}

/** Squeezes a 1-element 1D tensor `[1]` down to a 0D scalar if applicable. */
export function squeezeIfLen1(
    builder: GraphBuilder,
    t: ValueNode,
    axes: ConcreteValueNode,
    _tag: string,
): ValueNode {
    if ((t.shape as Shape | KnownShape | undefined) && t.shape.length === 1 && t.shape[0] === 1) {
        const [out] = builder.createOp("Squeeze", [t, axes]);
        return out;
    }
    return t;
}
