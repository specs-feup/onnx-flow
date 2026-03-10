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
import type { GraphBuilder } from "../../GraphBuilder.js";

/**
 * Decodes a linear iteration index into a multi-index based on output radices.
 */
export function decodeMixedRadix(
    builder: GraphBuilder,
    iter: ConcreteValueNode,
    dims: (number | ValueNode)[], // <-- CHANGED: Accept ValueNode
    tag: string,
): ConcreteValueNode[] {
    const out: ConcreteValueNode[] = [];
    let rem = iter;

    for (let k = dims.length - 1; k >= 0; k--) {
        const dim = dims[k];
        let dConst: ValueNode;

        // --- CHANGED: Handle dynamic ValueNode dims ---
        if (typeof dim === "number") {
            const safeDim = dim > 0 ? dim : 1;
            dConst = builder.createConstant(`mr_dim_${tag}_${k}`, scalarInt64(safeDim));
        } else {
            dConst = dim; // Use the dynamic scalar node directly
        }
        // ----------------------------------------------

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
    strides: (number | ValueNode)[], // <-- CHANGED: Accept ValueNode
    tag: string,
): ConcreteValueNode {
    let acc: ConcreteValueNode = builder.createConstant(`lin_zero_${tag}`, scalarInt64(0));

    for (let i = 0; i < idx.length; i++) {
        // --- CHANGED: Handle dynamic ValueNode strides ---
        let sNode: ValueNode;
        if (typeof strides[i] === "number") {
            sNode = builder.createConstant(
                `lin_stride_${tag}_${i}`,
                scalarInt64(strides[i] as number),
            );
        } else {
            sNode = strides[i] as ValueNode;
        }
        // -------------------------------------------------

        const [mulOut] = builder.createOp("Mul", [idx[i], sNode]);
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

    // If shapes are purely static, use a string key. If dynamic, fallback to node ID to avoid cache collisions.
    const isDynamic = inDimsStatic.includes(-1) || outDimsStatic.includes(-1);
    const shapeKey = isDynamic
        ? `__idx_cache_dyn_${tInner.id}_to_out`
        : `__idx_cache_${inDimsStatic.join(",")}_to_${outDimsStatic.join(",")}`;

    let linU: ConcreteValueNode;

    if (valueMap.has(shapeKey)) {
        linU = valueMap.get(shapeKey) as ConcreteValueNode;
    } else {
        const rO = outDimsStatic.length;
        const rI = inDimsStatic.length;

        // --- NEW DYNAMIC SUPPORT ---
        let outDims: (number | ValueNode)[] = outDimsStatic;
        let inDims: (number | ValueNode)[] = inDimsStatic;

        if (isDynamic) {
            const extractDims = (tensor: ValueNode, staticDims: number[], isOut: boolean) => {
                const dims: (number | ValueNode)[] = [];
                let shapeNode: ValueNode | null = null;

                for (let i = 0; i < staticDims.length; i++) {
                    if (staticDims[i] !== -1) {
                        dims.push(staticDims[i]);
                    } else {
                        if (!shapeNode) {
                            shapeNode = builder.createOp("Shape", [tensor])[0];
                        }
                        const iConst = builder.createConstant(
                            `dim_idx_${isOut ? "out" : "in"}_${i}`,
                            int64Vec([i]),
                        );
                        const [dimRaw] = builder.createOp("Gather", [shapeNode, iConst], {
                            axis: 0,
                        });
                        dims.push(squeezeIfLen1(builder, dimRaw, axes, `sq_dim`));
                    }
                }
                return dims;
            };

            inDims = extractDims(tInner, inDimsStatic, false);
            // Note: For outShape, we assume the outer builder already placed the final Loop output shape
            // logic somewhere, but usually in element-wise ops, outShape comes from the broadcasted result.
            // If outShape is dynamic, we assume the pass provided it correctly or it mirrors the largest input.
            // If we don't have a direct tensor for outShape, we map it to -1 and handle it appropriately.
        }

        const outRadix = outDims.map((d) => {
            if (typeof d === "number") return d > 0 ? d : 1;
            // For dynamic radix: Max(d, 1)
            const oneConst = builder.createConstant(`one`, scalarInt64(1));
            return builder.createOp("Max", [d, oneConst])[0];
        });

        const inRadix = inDims.map((d) => {
            if (typeof d === "number") return d > 0 ? d : 1;
            const oneConst = builder.createConstant(`one`, scalarInt64(1));
            return builder.createOp("Max", [d, oneConst])[0];
        });
        // ---------------------------

        const oDigits = decodeMixedRadix(builder, iter, outRadix, `gb_out_${tInner.id}`);
        const iDigits: ConcreteValueNode[] = [];

        for (let k = 0; k < rI; k++) {
            const inDim = inRadix[k];
            const outPos = rO - rI + k;

            if (outPos < 0) {
                iDigits.push(builder.createConstant(`gb_zero_${tInner.id}_${k}`, scalarInt64(0)));
            } else if (typeof inDim === "number") {
                if (inDim === 1) {
                    iDigits.push(
                        builder.createConstant(`gb_zero_${tInner.id}_${k}`, scalarInt64(0)),
                    );
                } else {
                    iDigits.push(oDigits[outPos]);
                }
            } else {
                // Dynamic Broadcast Check: Where(Equal(inDim, 1), 0, oDigits[outPos])
                const oneConst = builder.createConstant(`gb_one_${tInner.id}_${k}`, scalarInt64(1));
                const zeroConst = builder.createConstant(
                    `gb_zero_${tInner.id}_${k}`,
                    scalarInt64(0),
                );

                const [isOne] = builder.createOp("Equal", [inDim, oneConst]);
                const [bcastIdx] = builder.createOp("Where", [isOne, zeroConst, oDigits[outPos]]);
                iDigits.push(bcastIdx as ConcreteValueNode);
            }
        }

        // --- DYNAMIC STRIDES CALCULATION ---
        let strides: (number | ValueNode)[] = new Array(rI).fill(1);
        let currentStride: ValueNode | number = 1;
        for (let i = rI - 1; i >= 0; i--) {
            strides[i] = currentStride;
            if (i > 0) {
                const dim = inRadix[i];
                if (typeof currentStride === "number" && typeof dim === "number") {
                    currentStride = currentStride * dim;
                } else {
                    const cNode =
                        typeof currentStride === "number"
                            ? builder.createConstant(`cs_${i}`, scalarInt64(currentStride))
                            : currentStride;
                    const dNode =
                        typeof dim === "number"
                            ? builder.createConstant(`cd_${i}`, scalarInt64(dim))
                            : (dim as ValueNode);
                    [currentStride] = builder.createOp("Mul", [cNode, dNode]);
                }
            }
        }
        // -----------------------------------

        const linScalar = buildLinearIndex(builder, iDigits, strides, `gb_lin_${tInner.id}`);
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
