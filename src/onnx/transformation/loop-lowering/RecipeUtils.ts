import type OnnxGraph from "../../OnnxGraph.js";
import type { ValueNode, ConcreteValueNode, KnownShape, Shape } from "../../OnnxTypes.js";
import { DataType } from "../../OnnxTypes.js";
import type BaseNode from "@specs-feup/flow/graph/BaseNode";
import ConstantNode from "../../ConstantNode.js";
import TensorNode from "../../TensorNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";
import {
    inferPoolDim,
    resolveAutoPads,
    toStaticShape,
    scalarInt64,
    int64Vec,
} from "../../utils/index.js";
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

function asValueNode(node: BaseNode.Class): ValueNode {
    if (node.is(TensorNode)) return node.as(TensorNode);
    if (node.is(ConstantNode)) return node.as(ConstantNode);
    if (node.is(RegionArgumentNode)) return node.as(RegionArgumentNode);
    throw new Error(`Expected ValueNode for '${node.id}'`);
}

/**
 * Extracts per-axis dimensions as static numbers where known and scalar 0D INT64 ValueNodes
 * via `Shape` + `Gather` for dynamic (-1) axes. Zero ops are emitted if all dims are static.
 */
export function extractMixedDims(
    builder: GraphBuilder,
    tensor: ValueNode,
    staticDims: number[],
    tag: string,
): (number | ConcreteValueNode)[] {
    if (!staticDims.includes(-1)) return [...staticDims];
    const [shapeVec] = builder.createOp("Shape", [tensor], {}, [
        { type: DataType.INT64, shape: [staticDims.length] },
    ]);
    return staticDims.map((d, axis) => {
        if (d !== -1) return d;
        const idx = builder.createConstant(`${tag}_didx_${axis}`, scalarInt64(axis));
        return gatherScalar(builder, shapeVec, idx);
    });
}

/**
 * Computes row-major strides from mixed static/dynamic dimension values without emitting
 * unnecessary `Mul` nodes for static dimensions or stride-1 tails.
 */
export function buildMixedStrides(
    builder: GraphBuilder,
    dims: (number | ConcreteValueNode)[],
): (number | ConcreteValueNode)[] {
    const rank = dims.length;
    const strides: (number | ConcreteValueNode)[] = new Array(rank).fill(1);
    let cur: number | ConcreteValueNode = 1;
    for (let d = rank - 1; d >= 0; d--) {
        strides[d] = cur;
        if (d > 0) {
            const dim = dims[d];
            if (typeof cur === "number" && typeof dim === "number") {
                cur = cur * dim;
            } else if (cur === 1) {
                cur = dim;
            } else if (dim !== 1) {
                const cNode =
                    typeof cur === "number"
                        ? builder.createConstant(`mstr_${d}`, scalarInt64(cur))
                        : cur;
                const dNode =
                    typeof dim === "number"
                        ? builder.createConstant(`mdim_${d}`, scalarInt64(dim))
                        : dim;
                [cur] = builder.createOp("Mul", [cNode, dNode]);
            }
        }
    }
    return strides;
}

/**
 * Multiplies two mixed static/dynamic INT64 scalar values cleanly without redundant Mul(*1).
 */
export function mulMixed(
    builder: GraphBuilder,
    a: number | ConcreteValueNode,
    b: number | ConcreteValueNode,
    tag: string,
): number | ConcreteValueNode {
    if (typeof a === "number" && typeof b === "number") return a * b;
    if (a === 1) return b;
    if (b === 1) return a;
    if (a === 0 || b === 0) return 0;
    const aNode = typeof a === "number" ? builder.createConstant(`${tag}_a`, scalarInt64(a)) : a;
    const bNode = typeof b === "number" ? builder.createConstant(`${tag}_b`, scalarInt64(b)) : b;
    return builder.createOp("Mul", [aNode, bNode])[0];
}

/**
 * Multiplies a list of mixed static/dynamic INT64 scalar dimensions into a scalar INT64 node or number.
 */
export function productMixed(
    builder: GraphBuilder,
    dims: (number | ConcreteValueNode)[],
    tag: string,
): ConcreteValueNode {
    let acc: number | ConcreteValueNode = 1;
    for (let i = 0; i < dims.length; i++) {
        acc = mulMixed(builder, acc, dims[i], `${tag}_${i}`);
    }
    return typeof acc === "number" ? builder.createConstant(`${tag}_const`, scalarInt64(acc)) : acc;
}

/**
 * Adds two mixed static/dynamic INT64 scalar values without redundant Add(+0).
 */
export function addMixed(
    builder: GraphBuilder,
    a: number | ConcreteValueNode,
    b: number | ConcreteValueNode,
    tag: string,
): number | ConcreteValueNode {
    if (typeof a === "number" && typeof b === "number") return a + b;
    if (a === 0) return b;
    if (b === 0) return a;
    const aNode = typeof a === "number" ? builder.createConstant(`${tag}_a`, scalarInt64(a)) : a;
    const bNode = typeof b === "number" ? builder.createConstant(`${tag}_b`, scalarInt64(b)) : b;
    return builder.createOp("Add", [aNode, bNode])[0];
}

/**
 * Packs a list of mixed static/dynamic scalar INT64 dimensions into a 1D INT64 shape vector.
 * Uses a static base vector + `ScatterElements` for dynamic dims so shape construction
 * stays lightweight without triggering `Concat` lowering in canonicalization.
 */
export function packShapeVector(
    builder: GraphBuilder,
    dims: (number | ConcreteValueNode)[],
    tag: string,
): ConcreteValueNode {
    if (!dims.some((d) => typeof d !== "number")) {
        return builder.createConstant(`${tag}_vec`, int64Vec(dims as number[]));
    }
    const baseStatic = dims.map((d) => (typeof d === "number" ? d : 0));
    let vec: ConcreteValueNode = builder.createConstant(`${tag}_base`, int64Vec(baseStatic));
    const ax0 = builder.createConstant(`${tag}_ax0`, int64Vec([0]));
    const outSpec = [{ type: DataType.INT64, shape: [dims.length] }];
    for (let i = 0; i < dims.length; i++) {
        const d = dims[i];
        if (typeof d === "number") continue;
        const idx1D = builder.createConstant(`${tag}_idx_${i}`, int64Vec([i]));
        const [val1D] = builder.createOp("Unsqueeze", [d, ax0], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);
        [vec] = builder.createOp("ScatterElements", [vec, idx1D, val1D], { axis: 0 }, outSpec);
    }
    return vec;
}

/**
 * Right-aligned NumPy/ONNX broadcast over mixed static/dynamic INT64 dimension vectors
 * in O(rank) ops without allocating data tensors.
 */
export function broadcastMixedShapes(
    builder: GraphBuilder,
    shapes: (number | ConcreteValueNode)[][],
    _tag: string,
): (number | ConcreteValueNode)[] {
    if (shapes.length === 0) return [];
    const maxRank = Math.max(...shapes.map((s) => s.length));
    const result: (number | ConcreteValueNode)[] = [];

    for (let k = 0; k < maxRank; k++) {
        let staticVal = 1;
        const dyns: ConcreteValueNode[] = [];

        for (const sh of shapes) {
            const offset = maxRank - sh.length;
            const d = k >= offset ? sh[k - offset] : 1;
            if (typeof d === "number") {
                if (d === 0) {
                    staticVal = 0;
                } else if (d > 1 && staticVal !== 0) {
                    staticVal = d;
                }
            } else {
                dyns.push(d);
            }
        }

        if (dyns.length === 0 || staticVal === 0) {
            result.push(staticVal);
        } else if (staticVal > 1) {
            result.push(staticVal);
        } else if (dyns.length === 1) {
            result.push(dyns[0]);
        } else {
            let acc: ConcreteValueNode = dyns[0];
            for (let i = 1; i < dyns.length; i++) {
                const [gt] = builder.createOp("Greater", [acc, dyns[i]], {}, [
                    { type: DataType.BOOL, shape: [] },
                ]);
                [acc] = builder.createOp("Where", [gt, acc, dyns[i]], {}, [
                    { type: DataType.INT64, shape: [] },
                ]);
            }
            result.push(acc);
        }
    }

    return result;
}

/**
 * Computes ONNX MatMul output dimensions and contraction length K using O(rank) shape math.
 */
export function computeMatMulOutputDimsMixed(
    builder: GraphBuilder,
    a: ValueNode,
    b: ValueNode,
    tag: string,
): { outDims: (number | ConcreteValueNode)[]; kDim: number | ConcreteValueNode } {
    const sA = toStaticShape(a.shape);
    const sB = toStaticShape(b.shape);
    const dimsA = extractMixedDims(builder, a, sA, `${tag}_a`);
    const dimsB = extractMixedDims(builder, b, sB, `${tag}_b`);

    const isA1D = sA.length === 1;
    const isB1D = sB.length === 1;
    const batchA = isA1D ? [] : dimsA.slice(0, -2);
    const batchB = isB1D ? [] : dimsB.slice(0, -2);
    const batchOut = broadcastMixedShapes(builder, [batchA, batchB], `${tag}_batch`);

    const outDims: (number | ConcreteValueNode)[] = [...batchOut];
    if (!isA1D) outDims.push(dimsA[dimsA.length - 2]);
    if (!isB1D) outDims.push(dimsB[dimsB.length - 1]);

    return { outDims, kDim: dimsA[dimsA.length - 1] };
}

/**
 * Captures any dynamic INT64 scalar nodes from an outer builder list into `innerBuilder`.
 */
export function captureMixedList(
    innerBuilder: GraphBuilder,
    vals: (number | ConcreteValueNode)[],
): (number | ConcreteValueNode)[] {
    return vals.map((v) =>
        typeof v === "number" ? v : (ensureInnerValueNode(innerBuilder, v) as ConcreteValueNode),
    );
}

/**
 * Computes head pads and spatial output dims for Pool ops with static or dynamic spatial dims.
 */
export function resolvePoolShapeAndPadsMixed(
    autoPad: string,
    inSpatial: (number | ConcreteValueNode)[],
    kernel: number[],
    strides: number[],
    dilations: number[],
    explicitPads: number[],
    ceilMode: number,
    builder: GraphBuilder,
    tag: string,
): { pHeads: (number | ConcreteValueNode)[]; outSpatial: (number | ConcreteValueNode)[] } {
    const spatialRank = inSpatial.length;
    if (!inSpatial.some((d) => typeof d !== "number")) {
        const staticIn = inSpatial as number[];
        const pads = resolveAutoPads(autoPad, staticIn, kernel, strides, dilations, explicitPads);
        const outSpatial = staticIn.map((inD, d) =>
            inferPoolDim(
                inD,
                kernel[d],
                strides[d] ?? 1,
                pads[d] ?? 0,
                pads[spatialRank + d] ?? 0,
                dilations[d] ?? 1,
                ceilMode,
            ),
        );
        return { pHeads: pads.slice(0, spatialRank), outSpatial };
    }

    const zeroI64 = builder.createConstant(`${tag}_z`, scalarInt64(0));
    const oneI64 = builder.createConstant(`${tag}_o`, scalarInt64(1));
    const pHeads: (number | ConcreteValueNode)[] = [];
    const outSpatial: (number | ConcreteValueNode)[] = [];

    for (let d = 0; d < spatialRank; d++) {
        const inD = inSpatial[d];
        const k = kernel[d];
        const s = strides[d] ?? 1;
        const dil = dilations[d] ?? 1;
        const effK = (k - 1) * dil + 1;

        if (typeof inD === "number") {
            const singlePads = resolveAutoPads(
                autoPad,
                [inD],
                [k],
                [s],
                [dil],
                explicitPads.length === spatialRank * 2
                    ? [explicitPads[d], explicitPads[spatialRank + d]]
                    : [],
            );
            pHeads.push(singlePads[0]);
            outSpatial.push(inferPoolDim(inD, k, s, singlePads[0], singlePads[1], dil, ceilMode));
            continue;
        }

        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            let outDim: ConcreteValueNode;
            if (s === 1) {
                outDim = inD;
            } else {
                const sMinus1 = builder.createConstant(`${tag}_sm1_${d}`, scalarInt64(s - 1));
                const sConst = builder.createConstant(`${tag}_s_${d}`, scalarInt64(s));
                const [ceilNum] = builder.createOp("Add", [inD, sMinus1]);
                [outDim] = builder.createOp("Div", [ceilNum, sConst]);
            }
            const [outM1] = builder.createOp("Sub", [outDim, oneI64]);
            const stepped =
                s === 1
                    ? outM1
                    : builder.createOp("Mul", [
                          outM1,
                          builder.createConstant(`${tag}_st_${d}`, scalarInt64(s)),
                      ])[0];
            const effKC = builder.createConstant(`${tag}_ek_${d}`, scalarInt64(effK));
            const [needed] = builder.createOp("Add", [stepped, effKC]);
            const [rawPad] = builder.createOp("Sub", [needed, inD]);
            const [totalPad] = builder.createOp("Max", [rawPad, zeroI64]);
            const twoC = builder.createConstant(`${tag}_2_${d}`, scalarInt64(2));
            const padNum =
                autoPad === "SAME_LOWER"
                    ? builder.createOp("Add", [totalPad, oneI64])[0]
                    : totalPad;
            const [pHead] = builder.createOp("Div", [padNum, twoC]);
            pHeads.push(pHead);
            outSpatial.push(outDim);
            continue;
        }

        const pHead = autoPad === "VALID" ? 0 : (explicitPads[d] ?? 0);
        const pTail = autoPad === "VALID" ? 0 : (explicitPads[spatialRank + d] ?? 0);
        pHeads.push(pHead);

        const delta = pHead + pTail - effK;
        const [num] = builder.createOp("Add", [
            inD,
            builder.createConstant(`${tag}_del_${d}`, scalarInt64(delta)),
        ]);

        let rawOut: ConcreteValueNode;
        if (s === 1) {
            [rawOut] = builder.createOp("Add", [num, oneI64]);
        } else if (ceilMode === 1) {
            const sConst = builder.createConstant(`${tag}_sc_${d}`, scalarInt64(s));
            const sM1Const = builder.createConstant(`${tag}_sm1c_${d}`, scalarInt64(s - 1));
            const [ceilNum] = builder.createOp("Add", [num, sM1Const]);
            const [ceilDiv] = builder.createOp("Div", [ceilNum, sConst]);
            const [candOut] = builder.createOp("Add", [ceilDiv, oneI64]);
            const [lastStart] = builder.createOp("Mul", [ceilDiv, sConst]);
            const [inBound] =
                pHead === 0
                    ? [inD]
                    : builder.createOp("Add", [
                          inD,
                          builder.createConstant(`${tag}_ph_${d}`, scalarInt64(pHead)),
                      ]);
            const [validWin] = builder.createOp("Less", [lastStart, inBound]);
            [rawOut] = builder.createOp("Where", [validWin, candOut, ceilDiv]);
        } else {
            const sConst = builder.createConstant(`${tag}_sd_${d}`, scalarInt64(s));
            const [flDiv] = builder.createOp("Div", [num, sConst]);
            [rawOut] = builder.createOp("Add", [flDiv, oneI64]);
        }

        const [clampedOut] = builder.createOp("Max", [rawOut, zeroI64]);
        outSpatial.push(clampedOut);
    }

    return { pHeads, outSpatial };
}

/**
 * Ensures an input in outerGraph is available inside innerGraph as a ValueNode.
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
    stepEnable?: ConcreteValueNode,
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
                const [steppedPtr] = builder.createOp("Add", [ptrIn, stepConst]);
                const nextPtr = stepEnable
                    ? builder.createOp("Where", [stepEnable, steppedPtr, ptrIn])[0]
                    : steppedPtr;
                loopCtx.setNextState(ptrIn, nextPtr);
            }
            return ptrIn;
        }

        // General Static AGU Cascade
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
            // Only emit a Where mux if the net delta actually changes at this dimension boundary
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

    // 4. Dynamic Dimensions / Strides AGU Cascade (Zero Div/Mod)
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
 * Reads a single scalar (0D) element from a 1D tensor at a scalar 0D index,
 * avoiding Unsqueeze/Squeeze wrapper ops around Gather.
 */
export function gatherScalar(
    builder: GraphBuilder,
    flatTensor: ValueNode,
    scalarIdx: ConcreteValueNode,
): ConcreteValueNode {
    return builder.createOp("Gather", [flatTensor, scalarIdx], { axis: 0 })[0];
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
        // Keep shapeKey below using original tOuter dims, while gathering from tInnerFlat
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

    let tInner: ValueNode;
    if (builder.graph.hasNode(tOuter.id)) {
        tInner = asValueNode(builder.graph.getNodeById(tOuter.id)!);
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

function resolveGatherScalarIndex(
    builder: GraphBuilder,
    tRef: ValueNode,
    valueMap: Map<string, ValueNode>,
    iter: ConcreteValueNode,
    axes: ConcreteValueNode,
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
        const extractDims = (tensor: ValueNode, staticDims: number[], isOut: boolean) => {
            const innerTensor = ensureInnerValueNode(builder, tensor);
            const dims: (number | ValueNode)[] = [];
            let shapeNode: ValueNode | null = null;
            for (let i = 0; i < staticDims.length; i++) {
                if (staticDims[i] !== -1) {
                    dims.push(staticDims[i]);
                } else {
                    if (!shapeNode) {
                        shapeNode = builder.createOp("Shape", [innerTensor])[0];
                    }
                    const iConst = builder.createConstant(
                        `dim_idx_${isOut ? "out" : "in"}_${i}`,
                        scalarInt64(i),
                    );
                    dims.push(gatherScalar(builder, shapeNode, iConst));
                }
            }
            return dims;
        };
        inDims = extractDims(tRef, inDimsStatic, false);
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
        // Legacy Radix Mode
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
