import type OnnxGraph from "../../OnnxGraph.js";
import type { ValueNode, ConcreteValueNode, KnownShape, Shape } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../../ConstantNode.js";
import TensorNode from "../../TensorNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";
import { toStaticShape, scalarInt64, int64Vec } from "../../Utils.js";
import type { GraphBuilder } from "../../GraphBuilder.js";
import type { LoopCarryContext } from "./LoopLoweringRecipe.js";
import { IndexingMode } from "../../../DecompositionOptions.js";

/**
 * Decodes a linear iteration index into a multi-index based on output radices (Radix Mode).
 */
export function decodeMixedRadix(
    builder: GraphBuilder,
    iter: ConcreteValueNode,
    dims: (number | ValueNode)[],
    tag: string,
): ConcreteValueNode[] {
    const out: ConcreteValueNode[] = [];
    let rem = iter;

    for (let k = dims.length - 1; k >= 0; k--) {
        const dim = dims[k];
        let dConst: ValueNode;

        if (typeof dim === "number") {
            const safeDim = dim > 0 ? dim : 1;
            dConst = builder.createConstant(`mr_dim_${tag}_${k}`, scalarInt64(safeDim));
        } else {
            dConst = dim;
        }

        const [modOut] = builder.createOp("Mod", [rem, dConst]);
        out.unshift(modOut);

        const [divOut] = builder.createOp("Div", [rem, dConst]);
        rem = divOut;
    }
    return out;
}

/**
 * Builds a linear index from a multi-index and strides (Radix Mode).
 */
export function buildLinearIndex(
    builder: GraphBuilder,
    idx: ConcreteValueNode[],
    strides: (number | ValueNode)[],
    tag: string,
): ConcreteValueNode {
    let acc: ConcreteValueNode = builder.createConstant(`lin_zero_${tag}`, scalarInt64(0));
    for (let i = 0; i < idx.length; i++) {
        let sNode: ValueNode;
        if (typeof strides[i] === "number") {
            sNode = builder.createConstant(
                `lin_stride_${tag}_${i}`,
                scalarInt64(strides[i] as number),
            );
        } else {
            sNode = strides[i] as ValueNode;
        }

        const [mulOut] = builder.createOp("Mul", [idx[i], sNode]);
        const [addOut] = builder.createOp("Add", [acc, mulOut]);
        acc = addOut;
    }
    return acc;
}

/**
 * Ensures an input in outerGraph is available inside innerGraph as a ValueNode.
 */
export function ensureInnerValueNode(builder: GraphBuilder, d: ValueNode): ValueNode {
    if (builder.graph.hasNode(d.id)) {
        return builder.graph.getNodeById(d.id) as ValueNode;
    }
    if (d.is(ConstantNode)) {
        const c = d.as(ConstantNode);
        return builder.createConstant(c.id, c.constantValue);
    }
    return createCapturedInput(builder.graph, d);
}

export interface AguSpaceCounters {
    counters: ConcreteValueNode[];
    steps: ConcreteValueNode[];
    wraps: ConcreteValueNode[];
}

/**
 * Creates or retrieves shared loop-carried dimension counters [c_0, ..., c_{R-1}]
 * and wrap signals [wrap_0, ..., wrap_{R-1}] for an iteration space of dimensions `dims`.
 * Uses exclusively Add, Equal, And, Where (zero Div/Mod).
 */
export function getOrCreateIterationSpaceCounters(
    builder: GraphBuilder,
    loopCtx: LoopCarryContext,
    valueMap: Map<string, ValueNode>,
    dims: (number | ValueNode)[],
    tag: string,
): AguSpaceCounters {
    const dimsKey = dims.map((d) => (typeof d === "number" ? d.toString() : d.id)).join(",");
    const cacheKey = `__agu_space_${dimsKey}`;

    const cached = (valueMap as unknown as Map<string, unknown>).get(cacheKey) as
        | AguSpaceCounters
        | undefined;
    if (cached) return cached;

    const R = dims.length;
    const counters: ConcreteValueNode[] = new Array(R);
    const steps: ConcreteValueNode[] = new Array(R);
    const wraps: ConcreteValueNode[] = new Array(R);

    const zeroInner = builder.createConstant(`agu_zero_${tag}`, scalarInt64(0));
    const oneInner = builder.createConstant(`agu_one_${tag}`, scalarInt64(1));

    for (let k = R - 1; k >= 0; k--) {
        const dim = dims[k];
        const isInnermost = k === R - 1;
        const stepCond = isInnermost ? undefined : wraps[k + 1];

        // Static size-1 dimension never changes from 0 and always wraps when stepped
        if (typeof dim === "number" && dim <= 1) {
            counters[k] = zeroInner;
            const trueConst =
                stepCond ??
                builder.createConstant(`agu_true_${tag}_${k}`, {
                    dataType: DataType.BOOL,
                    dims: [],
                    int32Data: [1],
                });
            steps[k] = trueConst;
            wraps[k] = trueConst;
            continue;
        }

        const initZeroOuter = loopCtx.outerBuilder.createConstant(
            `init_cnt_${tag}_${k}`,
            scalarInt64(0),
        );
        const cIn = loopCtx.addCarry(`cnt_${tag}_${k}`, initZeroOuter, DataType.INT64, []);
        counters[k] = cIn;

        const boundNode =
            typeof dim === "number"
                ? builder.createConstant(`agu_bound_${tag}_${k}`, scalarInt64(dim))
                : ensureInnerValueNode(builder, dim);

        const [cInc] = builder.createOp("Add", [cIn, oneInner]);
        const [atEnd] = builder.createOp("Equal", [cInc, boundNode]);

        let wrapK: ConcreteValueNode;
        let nextC: ConcreteValueNode;

        if (!stepCond) {
            wrapK = atEnd;
            [nextC] = builder.createOp("Where", [wrapK, zeroInner, cInc]);
            steps[k] = oneInner;
        } else {
            [wrapK] = builder.createOp("And", [stepCond, atEnd]);
            const [steppedC] = builder.createOp("Where", [stepCond, cInc, cIn]);
            [nextC] = builder.createOp("Where", [wrapK, zeroInner, steppedC]);
            steps[k] = stepCond;
        }

        wraps[k] = wrapK;
        loopCtx.setNextState(cIn, nextC);
    }

    const result: AguSpaceCounters = { counters, steps, wraps };
    (valueMap as unknown as Map<string, unknown>).set(cacheKey, result);
    return result;
}

/**
 * Builds an AGU pointer variable that traverses `dims` with `strides`.
 * Uses net wrap-delta folding so contiguous dimensions collapse to a single Add op.
 */
export function buildAguPointer(
    builder: GraphBuilder,
    loopCtx: LoopCarryContext,
    valueMap: Map<string, ValueNode>,
    dims: (number | ValueNode)[],
    strides: (number | ValueNode)[],
    tag: string,
    initialOffset: number | ValueNode = 0,
): ConcreteValueNode {
    const R = dims.length;
    const isAllStatic =
        dims.every((d) => typeof d === "number") && strides.every((s) => typeof s === "number");

    // 1. All-Zero Stride Fast-Path (Scalar / Full Broadcast)
    if (isAllStatic && (strides as number[]).every((s) => s === 0) && initialOffset === 0) {
        return builder.createConstant(`agu_const_zero_${tag}`, scalarInt64(0));
    }

    // 2. Allocate Carried Pointer
    const initOuter =
        typeof initialOffset === "number"
            ? loopCtx.outerBuilder.createConstant(`init_ptr_${tag}`, scalarInt64(initialOffset))
            : initialOffset;
    const ptrIn = loopCtx.addCarry(`ptr_${tag}`, initOuter, DataType.INT64, []);

    if (R === 0) {
        loopCtx.setNextState(ptrIn, ptrIn);
        return ptrIn;
    }

    // 3. Static Dimensions & Strides: Precompute net wrap-deltas at compile time
    if (isAllStatic) {
        const sDims = (dims as number[]).map((d) => (d > 0 ? d : 1));
        const sStrides = strides as number[];

        // delta[k] = strides[k] - sum_{m = k+1}^{R-1} (dims[m] - 1) * strides[m]
        const deltas: number[] = new Array(R).fill(0);
        let cumulativeRewind = 0;
        for (let k = R - 1; k >= 0; k--) {
            deltas[k] = sStrides[k] - cumulativeRewind;
            cumulativeRewind += (sDims[k] - 1) * sStrides[k];
        }

        // Contiguous Fast-Path: if all active dimensions have the exact same net delta,
        // no dimension counters or Where multiplexers are needed at all!
        const firstDelta = deltas[R - 1];
        const isUniformStep = deltas.every((d, k) => sDims[k] === 1 || d === firstDelta);
        if (isUniformStep) {
            if (firstDelta === 0) {
                loopCtx.setNextState(ptrIn, ptrIn);
            } else {
                const stepConst = builder.createConstant(
                    `agu_step_${tag}`,
                    scalarInt64(firstDelta),
                );
                const [nextPtr] = builder.createOp("Add", [ptrIn, stepConst]);
                loopCtx.setNextState(ptrIn, nextPtr);
            }
            return ptrIn;
        }

        // General Static AGU Cascade
        const { wraps } = getOrCreateIterationSpaceCounters(builder, loopCtx, valueMap, dims, tag);
        const deltaCache = new Map<number, ConcreteValueNode>();
        const getSteppedPtr = (delta: number): ConcreteValueNode => {
            if (delta === 0) return ptrIn;
            if (deltaCache.has(delta)) return deltaCache.get(delta)!;
            const dConst = builder.createConstant(`agu_delta_${tag}_${delta}`, scalarInt64(delta));
            const [p] = builder.createOp("Add", [ptrIn, dConst]);
            deltaCache.set(delta, p);
            return p;
        };

        let nextPtr = getSteppedPtr(deltas[R - 1]);
        let prevDelta = deltas[R - 1];

        for (let k = R - 2; k >= 0; k--) {
            if (sDims[k] <= 1) continue;
            const curDelta = deltas[k];
            // Only emit a Where mux if the net delta actually changes at this dimension boundary
            if (curDelta !== prevDelta) {
                const candidatePtr = getSteppedPtr(curDelta);
                [nextPtr] = builder.createOp("Where", [wraps[k + 1], candidatePtr, nextPtr]);
                prevDelta = curDelta;
            }
        }

        loopCtx.setNextState(ptrIn, nextPtr);
        return ptrIn;
    }

    // 4. Dynamic Dimensions / Strides AGU Cascade (Zero Div/Mod)
    const { wraps } = getOrCreateIterationSpaceCounters(builder, loopCtx, valueMap, dims, tag);
    const oneConst = builder.createConstant(`agu_dyn_one_${tag}`, scalarInt64(1));
    let rewindNode: ConcreteValueNode | null = null;
    const deltaNodes: ConcreteValueNode[] = new Array(R);

    for (let k = R - 1; k >= 0; k--) {
        const sNode =
            typeof strides[k] === "number"
                ? builder.createConstant(`agu_s_${tag}_${k}`, scalarInt64(strides[k] as number))
                : (ensureInnerValueNode(builder, strides[k] as ValueNode) as ConcreteValueNode);
        const dNode =
            typeof dims[k] === "number"
                ? builder.createConstant(`agu_d_${tag}_${k}`, scalarInt64(dims[k] as number))
                : (ensureInnerValueNode(builder, dims[k] as ValueNode) as ConcreteValueNode);

        if (!rewindNode) {
            deltaNodes[k] = sNode;
        } else {
            [deltaNodes[k]] = builder.createOp("Sub", [sNode, rewindNode]);
        }

        if (k > 0) {
            const [dMinus1] = builder.createOp("Sub", [dNode, oneConst]);
            const [dimRewind] = builder.createOp("Mul", [dMinus1, sNode]);
            rewindNode = rewindNode
                ? builder.createOp("Add", [rewindNode, dimRewind])[0]
                : dimRewind;
        }
    }

    let [nextPtr] = builder.createOp("Add", [ptrIn, deltaNodes[R - 1]]);
    for (let k = R - 2; k >= 0; k--) {
        const [candPtr] = builder.createOp("Add", [ptrIn, deltaNodes[k]]);
        [nextPtr] = builder.createOp("Where", [wraps[k + 1], candPtr, nextPtr]);
    }

    loopCtx.setNextState(ptrIn, nextPtr);
    return ptrIn;
}

/**
 * Ensures an input is 1D by reshaping it if necessary.
 */
export function ensureFlatInput(builder: GraphBuilder, t: ValueNode): ValueNode {
    if (!(t.shape as Shape | KnownShape | undefined) || t.shape.length <= 1) return t;
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
        if (existing !== undefined && existing.is(RegionArgumentNode))
            return existing.as(RegionArgumentNode);
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
 * Standard resolve input logic for recipes, supporting both Pointer (AGU) and Radix modes.
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

    let tOuter: ValueNode;
    if (input.is(TensorNode)) tOuter = input.as(TensorNode);
    else if (input.is(ConstantNode)) tOuter = input.as(ConstantNode);
    else if (input.is(RegionArgumentNode)) tOuter = input.as(RegionArgumentNode);
    else throw new Error(`Unhandled input case for ${input.id}`);

    let tInner: ValueNode;
    if (builder.graph.hasNode(tOuter.id)) {
        tInner = builder.graph.getNodeById(tOuter.id) as ValueNode;
    } else if (tOuter.is(ConstantNode)) {
        const c = tOuter.as(ConstantNode);
        tInner = builder.createConstant(c.id, c.constantValue);
    } else {
        tInner = createCapturedInput(builder.graph, tOuter);
    }

    if (
        !returnGather ||
        ((tInner.shape as Shape | KnownShape | undefined) && tInner.shape.length === 0)
    ) {
        const res = flatten ? ensureFlatInput(builder, tInner) : tInner;
        valueMap.set(input.id, res);
        return res;
    }

    const inDimsStatic = toStaticShape(tInner.shape as Shape);
    const outDimsStatic = toStaticShape(outShape as Shape);
    const isDynamic = inDimsStatic.includes(-1) || outDimsStatic.includes(-1);
    const modeTag = loopCtx?.indexingMode ?? IndexingMode.Radix;
    const shapeKey = isDynamic
        ? `__idx_cache_${modeTag}_dyn_${tInner.id}_to_out`
        : `__idx_cache_${modeTag}_${inDimsStatic.join(",")}_to_${outDimsStatic.join(",")}`;

    let linU: ConcreteValueNode;

    if (valueMap.has(shapeKey)) {
        linU = valueMap.get(shapeKey) as ConcreteValueNode;
    } else {
        const rO = outDimsStatic.length;
        const rI = inDimsStatic.length;

        const outDims: (number | ValueNode)[] = [...outDimsStatic];
        let inDims: (number | ValueNode)[] = [...inDimsStatic];

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
        }

        const outRadix = outDims.map((d, i) => {
            if (typeof d === "number") {
                if (d > 0) return d;
                if (targetShapeNode) {
                    const innerTarget = ensureInnerValueNode(builder, targetShapeNode);
                    const iConst = builder.createConstant(
                        `dim_idx_out_${tInner.id}_${i}`,
                        int64Vec([i]),
                    );
                    const [dimRaw] = builder.createOp("Gather", [innerTarget, iConst], {
                        axis: 0,
                    });
                    const sqDim = squeezeIfLen1(
                        builder,
                        dimRaw,
                        axes,
                        `sq_dim_out_${tInner.id}_${i}`,
                    );
                    const oneConst = builder.createConstant(
                        `one_max_${tInner.id}_${i}`,
                        scalarInt64(1),
                    );
                    return builder.createOp("Max", [sqDim, oneConst])[0];
                }
                return 1;
            }
            const oneConst = builder.createConstant(`one`, scalarInt64(1));
            return builder.createOp("Max", [d, oneConst])[0];
        });

        const inRadix = inDims.map((d) => {
            if (typeof d === "number") return d > 0 ? d : 1;
            const oneConst = builder.createConstant(`one`, scalarInt64(1));
            return builder.createOp("Max", [d, oneConst])[0];
        });

        // Compute raw input strides
        const inStrides: (number | ValueNode)[] = new Array(rI).fill(1);
        let currentStride: ValueNode | number = 1;
        for (let i = rI - 1; i >= 0; i--) {
            inStrides[i] = currentStride;
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

        let linScalar: ConcreteValueNode;

        // --- BRANCH: POINTER MATH vs RADIX ---
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            // Align input strides to the output iteration space (broadcasted dims get stride 0)
            const alignedStrides: (number | ValueNode)[] = new Array(rO).fill(0);
            for (let k = 0; k < rI; k++) {
                const outPos = rO - rI + k;
                if (outPos < 0) continue;
                const inDim = inRadix[k];
                if (typeof inDim === "number") {
                    alignedStrides[outPos] = inDim === 1 ? 0 : inStrides[k];
                } else {
                    const oneConst = builder.createConstant(
                        `agu_bcast_1_${tInner.id}_${k}`,
                        scalarInt64(1),
                    );
                    const zeroConst = builder.createConstant(
                        `agu_bcast_0_${tInner.id}_${k}`,
                        scalarInt64(0),
                    );
                    const [isOne] = builder.createOp("Equal", [inDim, oneConst]);
                    const sNode =
                        typeof inStrides[k] === "number"
                            ? builder.createConstant(
                                  `agu_s_val_${tInner.id}_${k}`,
                                  scalarInt64(inStrides[k] as number),
                              )
                            : (inStrides[k] as ValueNode);
                    const [effStride] = builder.createOp("Where", [isOne, zeroConst, sNode]);
                    alignedStrides[outPos] = effStride;
                }
            }

            linScalar = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                outRadix,
                alignedStrides,
                `in_${tInner.id}`,
            );
        } else {
            // Legacy Radix Mode
            const oDigits = decodeMixedRadix(builder, iter, outRadix, `gb_out_${tInner.id}`);
            const iDigits: ConcreteValueNode[] = [];
            for (let k = 0; k < rI; k++) {
                const inDim = inRadix[k];
                const outPos = rO - rI + k;
                if (outPos < 0) {
                    iDigits.push(
                        builder.createConstant(`gb_zero_${tInner.id}_${k}`, scalarInt64(0)),
                    );
                } else if (typeof inDim === "number") {
                    if (inDim === 1) {
                        iDigits.push(
                            builder.createConstant(`gb_zero_${tInner.id}_${k}`, scalarInt64(0)),
                        );
                    } else {
                        iDigits.push(oDigits[outPos]);
                    }
                } else {
                    const oneConst = builder.createConstant(
                        `gb_one_${tInner.id}_${k}`,
                        scalarInt64(1),
                    );
                    const zeroConst = builder.createConstant(
                        `gb_zero_${tInner.id}_${k}`,
                        scalarInt64(0),
                    );
                    const [isOne] = builder.createOp("Equal", [inDim, oneConst]);
                    const [bcastIdx] = builder.createOp("Where", [
                        isOne,
                        zeroConst,
                        oDigits[outPos],
                    ]);
                    iDigits.push(bcastIdx as ConcreteValueNode);
                }
            }
            linScalar = buildLinearIndex(builder, iDigits, inStrides, `gb_lin_${tInner.id}`);
        }

        [linU] = builder.createOp("Unsqueeze", [linScalar, axes]);
        valueMap.set(shapeKey, linU);
    }

    const flatT = ensureFlatInput(builder, tInner);
    const [gathered] = builder.createOp("Gather", [flatT, linU], { axis: 0 });
    const finalScalar = squeezeIfLen1(builder, gathered, axes, `gb_sq_${tInner.id}`);

    valueMap.set(input.id, finalScalar);
    return finalScalar;
}

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
