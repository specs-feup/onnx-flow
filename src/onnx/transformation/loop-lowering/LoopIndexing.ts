import type { ValueNode, ConcreteValueNode, KnownShape, Shape } from "../../ir/OnnxTypes.js";
import { DataType } from "../../ir/OnnxTypes.js";
import { toStaticShape, scalarInt64 } from "../../utils/index.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import type { LoopCarryContext } from "./LoopLoweringRecipe.js";
import { IndexingMode } from "../../../DecompositionOptions.js";
import { ensureInnerValueNode, gatherScalar } from "./RecipeUtils.js";

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
        const dConst: ValueNode =
            typeof dim === "number"
                ? builder.createConstant(`mr_dim_${tag}_${k}`, scalarInt64(dim > 0 ? dim : 1))
                : dim;

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
        const sNode: ValueNode =
            typeof strides[i] === "number"
                ? builder.createConstant(
                      `lin_stride_${tag}_${i}`,
                      scalarInt64(strides[i] as number),
                  )
                : (strides[i] as ValueNode);

        const [mulOut] = builder.createOp("Mul", [idx[i], sNode]);
        const [addOut] = builder.createOp("Add", [acc, mulOut]);
        acc = addOut;
    }
    return acc;
}

export interface AguSpaceCounters {
    counters: ConcreteValueNode[];
    steps: ConcreteValueNode[];
    wraps: ConcreteValueNode[];
}

/**
 * Creates or retrieves shared loop-carried dimension counters `[c_0, ..., c_{R-1}]`
 * and wrap signals `[wrap_0, ..., wrap_{R-1}]` for an iteration space of dimensions `dims`.
 * Uses exclusively `Add`, `Equal`, `And`, `Where` (zero `Div`/`Mod`).
 */
export function getOrCreateIterationSpaceCounters(
    builder: GraphBuilder,
    loopCtx: LoopCarryContext,
    valueMap: Map<string, ValueNode>,
    dims: (number | ValueNode)[],
    tag: string,
    stepEnable?: ConcreteValueNode,
): AguSpaceCounters {
    const dimsKey = dims.map((d) => (typeof d === "number" ? d.toString() : d.id)).join(",");
    const cacheKey = `__agu_space_${dimsKey}_${stepEnable ? stepEnable.id : "uncond"}`;

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
        const stepCond = isInnermost ? stepEnable : wraps[k + 1];

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
 * Uses net wrap-delta folding so contiguous dimensions collapse to a single `Add` op.
 */
export function buildAguPointer(
    builder: GraphBuilder,
    loopCtx: LoopCarryContext,
    valueMap: Map<string, ValueNode>,
    dims: (number | ValueNode)[],
    strides: (number | ValueNode)[],
    tag: string,
    initialOffset: number | ValueNode = 0,
    stepEnable?: ConcreteValueNode,
): ConcreteValueNode {
    const R = dims.length;
    const isAllStatic =
        dims.every((d) => typeof d === "number") && strides.every((s) => typeof s === "number");

    if (isAllStatic && (strides as number[]).every((s) => s === 0) && initialOffset === 0) {
        return builder.createConstant(`agu_const_zero_${tag}`, scalarInt64(0));
    }

    const initOuter =
        typeof initialOffset === "number"
            ? loopCtx.outerBuilder.createConstant(`init_ptr_${tag}`, scalarInt64(initialOffset))
            : initialOffset;
    const ptrIn = loopCtx.addCarry(`ptr_${tag}`, initOuter, DataType.INT64, []);

    if (R === 0) {
        loopCtx.setNextState(ptrIn, ptrIn);
        return ptrIn;
    }

    if (isAllStatic) {
        const sDims = (dims as number[]).map((d) => (d > 0 ? d : 1));
        const sStrides = strides as number[];

        const deltas: number[] = new Array(R).fill(0);
        let cumulativeRewind = 0;
        for (let k = R - 1; k >= 0; k--) {
            deltas[k] = sStrides[k] - cumulativeRewind;
            cumulativeRewind += (sDims[k] - 1) * sStrides[k];
        }

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
                const [steppedPtr] = builder.createOp("Add", [ptrIn, stepConst]);
                const nextPtr = stepEnable
                    ? builder.createOp("Where", [stepEnable, steppedPtr, ptrIn])[0]
                    : steppedPtr;
                loopCtx.setNextState(ptrIn, nextPtr);
            }
            return ptrIn;
        }

        const { wraps } = getOrCreateIterationSpaceCounters(
            builder,
            loopCtx,
            valueMap,
            dims,
            tag,
            stepEnable,
        );
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
            if (curDelta !== prevDelta) {
                const candidatePtr = getSteppedPtr(curDelta);
                [nextPtr] = builder.createOp("Where", [wraps[k + 1], candidatePtr, nextPtr]);
                prevDelta = curDelta;
            }
        }

        if (stepEnable) {
            [nextPtr] = builder.createOp("Where", [stepEnable, nextPtr, ptrIn]);
        }

        loopCtx.setNextState(ptrIn, nextPtr);
        return ptrIn;
    }

    const { wraps } = getOrCreateIterationSpaceCounters(
        builder,
        loopCtx,
        valueMap,
        dims,
        tag,
        stepEnable,
    );
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

    if (stepEnable) {
        [nextPtr] = builder.createOp("Where", [stepEnable, nextPtr, ptrIn]);
    }

    loopCtx.setNextState(ptrIn, nextPtr);
    return ptrIn;
}

/**
 * Computes (and caches per shape/mode) the linear gather index for `tRef` within
 * an output iteration space, supporting both Pointer (AGU) and Radix addressing.
 */
export function resolveGatherScalarIndex(
    builder: GraphBuilder,
    tRef: ValueNode,
    valueMap: Map<string, ValueNode>,
    iter: ConcreteValueNode,
    _axes: ConcreteValueNode,
    outShape: KnownShape,
    targetShapeNode?: ValueNode,
    loopCtx?: LoopCarryContext,
): ConcreteValueNode {
    const inDimsStatic = toStaticShape(tRef.shape as Shape);
    const outDimsStatic = toStaticShape(outShape as Shape);
    const isDynamic = inDimsStatic.includes(-1) || outDimsStatic.includes(-1);
    const modeTag = loopCtx?.indexingMode ?? IndexingMode.Radix;
    const shapeKey = isDynamic
        ? `__idx_cache_${modeTag}_dyn_${tRef.id}_to_out`
        : `__idx_cache_${modeTag}_${inDimsStatic.join(",")}_to_${outDimsStatic.join(",")}`;

    if (valueMap.has(shapeKey)) {
        return valueMap.get(shapeKey) as ConcreteValueNode;
    }

    const rO = outDimsStatic.length;
    const rI = inDimsStatic.length;
    const outDims: (number | ValueNode)[] = [...outDimsStatic];
    let inDims: (number | ValueNode)[] = [...inDimsStatic];

    if (isDynamic) {
        const innerTensor = ensureInnerValueNode(builder, tRef);
        const dims: (number | ValueNode)[] = [];
        let shapeNode: ValueNode | null = null;
        for (let i = 0; i < inDimsStatic.length; i++) {
            if (inDimsStatic[i] !== -1) {
                dims.push(inDimsStatic[i]);
            } else {
                if (!shapeNode) {
                    shapeNode = builder.createOp("Shape", [innerTensor])[0];
                }
                const iConst = builder.createConstant(`dim_idx_in_${i}`, scalarInt64(i));
                dims.push(gatherScalar(builder, shapeNode, iConst));
            }
        }
        inDims = dims;
    }

    const outRadix = outDims.map((d, i) => {
        if (typeof d === "number") {
            if (d > 0) return d;
            if (targetShapeNode) {
                const innerTarget = ensureInnerValueNode(builder, targetShapeNode);
                const iConst = builder.createConstant(
                    `dim_idx_out_${tRef.id}_${i}`,
                    scalarInt64(i),
                );
                const sqDim = gatherScalar(builder, innerTarget, iConst);
                const oneConst = builder.createConstant(`one_max_${tRef.id}_${i}`, scalarInt64(1));
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

    if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
        const alignedStrides: (number | ValueNode)[] = new Array(rO).fill(0);
        for (let k = 0; k < rI; k++) {
            const outPos = rO - rI + k;
            if (outPos < 0) continue;
            const inDim = inRadix[k];
            if (typeof inDim === "number") {
                alignedStrides[outPos] = inDim === 1 ? 0 : inStrides[k];
            } else {
                const oneConst = builder.createConstant(
                    `agu_bcast_1_${tRef.id}_${k}`,
                    scalarInt64(1),
                );
                const zeroConst = builder.createConstant(
                    `agu_bcast_0_${tRef.id}_${k}`,
                    scalarInt64(0),
                );
                const [isOne] = builder.createOp("Equal", [inDim, oneConst]);
                const sNode =
                    typeof inStrides[k] === "number"
                        ? builder.createConstant(
                              `agu_s_val_${tRef.id}_${k}`,
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
            `in_${tRef.id}`,
        );
    } else {
        const oDigits = decodeMixedRadix(builder, iter, outRadix, `gb_out_${tRef.id}`);
        const iDigits: ConcreteValueNode[] = [];
        for (let k = 0; k < rI; k++) {
            const inDim = inRadix[k];
            const outPos = rO - rI + k;
            if (outPos < 0) {
                iDigits.push(builder.createConstant(`gb_zero_${tRef.id}_${k}`, scalarInt64(0)));
            } else if (typeof inDim === "number") {
                if (inDim === 1) {
                    iDigits.push(builder.createConstant(`gb_zero_${tRef.id}_${k}`, scalarInt64(0)));
                } else {
                    iDigits.push(oDigits[outPos]);
                }
            } else {
                const oneConst = builder.createConstant(`gb_one_${tRef.id}_${k}`, scalarInt64(1));
                const zeroConst = builder.createConstant(`gb_zero_${tRef.id}_${k}`, scalarInt64(0));
                const [isOne] = builder.createOp("Equal", [inDim, oneConst]);
                const [bcastIdx] = builder.createOp("Where", [isOne, zeroConst, oDigits[outPos]]);
                iDigits.push(bcastIdx as ConcreteValueNode);
            }
        }
        linScalar = buildLinearIndex(builder, iDigits, inStrides, `gb_lin_${tRef.id}`);
    }

    valueMap.set(shapeKey, linScalar);
    return linScalar;
}
