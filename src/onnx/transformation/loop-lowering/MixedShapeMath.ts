import type { ValueNode, ConcreteValueNode } from "../../ir/OnnxTypes.js";
import { DataType } from "../../ir/OnnxTypes.js";
import {
    inferPoolDim,
    resolveAutoPads,
    toStaticShape,
    scalarInt64,
    int64Vec,
} from "../../utils/index.js";
import type { GraphBuilder } from "../../ir/GraphBuilder.js";
import { ensureInnerValueNode, gatherScalar } from "./RecipeUtils.js";

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

/** Multiplies two mixed static/dynamic INT64 scalar values without redundant `Mul(*1)`. */
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

/** Multiplies a list of mixed static/dynamic INT64 scalar dimensions into a single INT64 scalar. */
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

/** Adds two mixed static/dynamic INT64 scalar values without redundant `Add(+0)`. */
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
 * Packs a list of mixed static/dynamic scalar INT64 dimensions into a 1D INT64 shape vector
 * using a static base vector + `ScatterElements` for dynamic axes.
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

/** Computes ONNX MatMul output dimensions and contraction length K using O(rank) shape math. */
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

/** Captures any dynamic INT64 scalar nodes from an outer builder list into `innerBuilder`. */
export function captureMixedList(
    innerBuilder: GraphBuilder,
    vals: (number | ConcreteValueNode)[],
): (number | ConcreteValueNode)[] {
    return vals.map((v) =>
        typeof v === "number" ? v : (ensureInnerValueNode(innerBuilder, v) as ConcreteValueNode),
    );
}

/** Computes head pads and spatial output dims for Pool ops with static or dynamic spatial dims. */
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
