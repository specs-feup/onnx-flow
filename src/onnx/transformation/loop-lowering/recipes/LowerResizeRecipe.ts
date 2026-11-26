import type OnnxGraph from "../../../ir/OnnxGraph.js";
import type OperationNode from "../../../ir/OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../ir/OnnxTypes.js";
import {
    getIntAttr,
    getStringAttr,
    int64Vec,
    makeTensorProto,
    readConstIntegerVectorFromTensorNode,
    readVectorFromTensorNode,
    scalarInt64,
    toStaticShape,
} from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    buildMixedStrides,
    decodeMixedRadix,
    ensureFlatInput,
    ensureInnerValueNode,
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../ir/GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

const SUPPORTED_COORD_MODES = new Set([
    "half_pixel",
    "pytorch_half_pixel",
    "align_corners",
    "asymmetric",
]);

export interface ResizeAxisAnalysis {
    canResolve: boolean;
    resizedMask: boolean[];
    staticOut?: number[] | undefined;
}

export function analyzeResizeAxes(op: OperationNode.Class, inDims: number[]): ResizeAxisAnalysis {
    const rank = inDims.length;
    const ins = op.getInputs()!;
    const sizesNode = ins.length >= 4 && ins[3]?.id ? ins[3] : undefined;
    const scalesNode = ins.length >= 3 && ins[2]?.id ? ins[2] : undefined;

    const staticSizes = sizesNode ? readConstIntegerVectorFromTensorNode(sizesNode) : undefined;
    if (staticSizes && staticSizes.length === rank) {
        const resizedMask = inDims.map((d, i) => d === -1 || d !== staticSizes[i]);
        return { canResolve: true, resizedMask, staticOut: staticSizes };
    }
    if (sizesNode && Array.isArray(sizesNode.shape) && sizesNode.shape[0] === rank) {
        return { canResolve: true, resizedMask: new Array<boolean>(rank).fill(true) };
    }

    const staticScales = scalesNode ? readVectorFromTensorNode(scalesNode) : undefined;
    if (staticScales && staticScales.length === rank) {
        const resizedMask = staticScales.map((s) => s !== 1);
        const staticOut = inDims.includes(-1)
            ? undefined
            : inDims.map((d, i) => Math.floor(d * staticScales[i]));
        return { canResolve: true, resizedMask, staticOut };
    }
    if (scalesNode && Array.isArray(scalesNode.shape) && scalesNode.shape[0] === rank) {
        return { canResolve: true, resizedMask: new Array<boolean>(rank).fill(true) };
    }

    const outShape = toStaticShape(op.getOutputs()[0]?.shape);
    if (outShape.length === rank && !outShape.includes(-1)) {
        const resizedMask = inDims.map((d, i) => d === -1 || d !== outShape[i]);
        return { canResolve: true, resizedMask, staticOut: outShape };
    }
    return { canResolve: false, resizedMask: [] };
}

export function computeResizeLoopBounds(
    op: OperationNode.Class,
    tagPrefix: string,
): {
    totalIters: number | ConcreteValueNode;
    carryShape: number[] | ConcreteValueNode;
    targetShape?: ConcreteValueNode;
} {
    const ins = op.getInputs()!;
    const X = ins[0];
    const inDims = toStaticShape(X.shape);
    const info = analyzeResizeAxes(op, inDims);
    if (info.staticOut && !info.staticOut.includes(-1)) {
        const total = info.staticOut.reduce((a, b) => a * b, 1);
        return { totalIters: total, carryShape: info.staticOut };
    }

    const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `${tagPrefix}_bounds_${op.id}`);
    const ax0 = builder.createConstant(`${tagPrefix}b_ax0_${op.id}`, int64Vec([0]));
    const sizesNode = ins.length >= 4 && ins[3]?.id ? ins[3] : undefined;
    const scalesNode = ins.length >= 3 && ins[2]?.id ? ins[2] : undefined;

    let targetShape: ConcreteValueNode;
    if (sizesNode) {
        targetShape =
            sizesNode.literalType === DataType.INT64
                ? (sizesNode as ConcreteValueNode)
                : builder.createOp("Cast", [sizesNode], { to: DataType.INT64 })[0];
    } else {
        const [inShapeVec] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [inDims.length] },
        ]);
        const [inShapeF] = builder.createOp("Cast", [inShapeVec], { to: DataType.FLOAT });
        const scalesF =
            scalesNode!.literalType === DataType.FLOAT
                ? scalesNode!
                : builder.createOp("Cast", [scalesNode!], { to: DataType.FLOAT })[0];
        const [scaledF] = builder.createOp("Mul", [inShapeF, scalesF]);
        const [flF] = builder.createOp("Floor", [scaledF]);
        [targetShape] = builder.createOp("Cast", [flF], { to: DataType.INT64 }, [
            { type: DataType.INT64, shape: [inDims.length] },
        ]);
    }

    const [totalIters] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
    const [carryShape] = builder.createOp("Unsqueeze", [totalIters, ax0]);
    return { totalIters, carryShape, targetShape };
}

export function resolveResizeOutDims(
    builder: GraphBuilder,
    info: ResizeAxisAnalysis,
    inDims: (number | ConcreteValueNode)[],
    targetShapeNode: ValueNode | undefined,
    tagPrefix: string,
    opId: string,
): (number | ConcreteValueNode)[] {
    if (info.staticOut && !info.staticOut.includes(-1)) {
        return [...info.staticOut];
    }
    const innerTgt = ensureInnerValueNode(builder, targetShapeNode!);
    return Array.from({ length: inDims.length }, (_, d) => {
        if (!info.resizedMask[d]) return inDims[d];
        return gatherScalar(
            builder,
            innerTgt,
            builder.createConstant(`${tagPrefix}_od_${opId}_${d}`, scalarInt64(d)),
        );
    });
}

export function mapResizeCoordToFloat(
    builder: GraphBuilder,
    coord: ConcreteValueNode,
    inD: number | ConcreteValueNode,
    outD: number | ConcreteValueNode,
    coordMode: string,
    halfF: ConcreteValueNode,
    zeroF: ConcreteValueNode,
    oneI64: ConcreteValueNode,
    twoI64: ConcreteValueNode,
    tagPrefix: string,
    d: number,
    opId: string,
): ConcreteValueNode {
    const cF = (tag: string, v: number) =>
        builder.createConstant(
            `${tagPrefix}_${tag}_${opId}`,
            makeTensorProto(DataType.FLOAT, [], [v]),
        );
    const [cFloat] = builder.createOp("Cast", [coord], { to: DataType.FLOAT });

    if (typeof inD === "number" && typeof outD === "number") {
        const invScale = inD / outD;
        if (coordMode === "asymmetric") {
            return builder.createOp("Mul", [cFloat, cF(`is_${d}`, invScale)])[0];
        }
        if (coordMode === "align_corners") {
            if (outD <= 1) return zeroF;
            return builder.createOp("Mul", [cFloat, cF(`ac_${d}`, (inD - 1) / (outD - 1))])[0];
        }
        if (coordMode === "pytorch_half_pixel" && outD <= 1) {
            return zeroF;
        }
        const [plusHalf] = builder.createOp("Add", [cFloat, halfF]);
        const [scaled] = builder.createOp("Mul", [plusHalf, cF(`hp_${d}`, invScale)]);
        return builder.createOp("Sub", [scaled, halfF])[0];
    }

    const inNode =
        typeof inD === "number"
            ? builder.createConstant(`${tagPrefix}_in_${d}_${opId}`, scalarInt64(inD))
            : inD;
    const outNode =
        typeof outD === "number"
            ? builder.createConstant(`${tagPrefix}_out_${d}_${opId}`, scalarInt64(outD))
            : outD;
    const [inF] = builder.createOp("Cast", [inNode], { to: DataType.FLOAT });
    const [outF] = builder.createOp("Cast", [outNode], { to: DataType.FLOAT });

    if (coordMode === "asymmetric") {
        const [invS] = builder.createOp("Div", [inF, outF]);
        return builder.createOp("Mul", [cFloat, invS])[0];
    }
    if (coordMode === "align_corners") {
        const [isSingle] = builder.createOp("Less", [outNode, twoI64]);
        const [inM1] = builder.createOp("Sub", [inNode, oneI64]);
        const [outM1] = builder.createOp("Sub", [outNode, oneI64]);
        const [safeOutM1] = builder.createOp("Where", [isSingle, oneI64, outM1]);
        const [inM1F] = builder.createOp("Cast", [inM1], { to: DataType.FLOAT });
        const [outM1F] = builder.createOp("Cast", [safeOutM1], { to: DataType.FLOAT });
        const [acScale] = builder.createOp("Div", [inM1F, outM1F]);
        const [scaled] = builder.createOp("Mul", [cFloat, acScale]);
        return builder.createOp("Where", [isSingle, zeroF, scaled])[0];
    }

    const [invS] = builder.createOp("Div", [inF, outF]);
    const [plusHalf] = builder.createOp("Add", [cFloat, halfF]);
    const [scaled] = builder.createOp("Mul", [plusHalf, invS]);
    const [hp] = builder.createOp("Sub", [scaled, halfF]);
    if (coordMode === "pytorch_half_pixel") {
        const [isSingle] = builder.createOp("Less", [outNode, twoI64]);
        return builder.createOp("Where", [isSingle, zeroF, hp])[0];
    }
    return hp;
}

export class LowerResizeRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerResize";
    public readonly targetOp = "Resize";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["imaging", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "ReduceProd",
        "Reshape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Cast",
        "Add",
        "Sub",
        "Mul",
        "Floor",
        "Clip",
        "Less",
        "Equal",
        "And",
        "Where",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Resize") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        const mode = getStringAttr(op, "mode", "nearest");
        if (mode !== "nearest" && mode !== "linear") return false;

        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        if (!SUPPORTED_COORD_MODES.has(coordMode)) return false;
        if (getIntAttr(op, "antialias", 0) !== 0) return false;

        const inDims = toStaticShape(ins[0].shape);
        if (inDims.length === 0) return false;

        const info = analyzeResizeAxes(op, inDims);
        if (!info.canResolve) return false;

        if (mode === "linear") {
            const resizedCount = info.resizedMask.filter(Boolean).length;
            return resizedCount <= 2; // Supports 1D and 2D (bilinear) linear interpolation
        }

        return true;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        return computeResizeLoopBounds(op, "rz");
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        _carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `resize_${op.id}`);
        const ins = op.getInputs()!;
        const X = ins[0];
        const rawInDims = toStaticShape(X.shape);
        const info = analyzeResizeAxes(op, rawInDims);
        const dtype = X.literalType as DataType;
        const rank = rawInDims.length;

        const mode = getStringAttr(op, "mode", "nearest");
        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        const nearestMode = getStringAttr(op, "nearest_mode", "round_prefer_floor");

        const rawX = resolveRecipeInput(
            builder,
            X,
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const inDims = extractMixedDims(builder, rawX, rawInDims, `rz_id_${op.id}`);
        const inStrides = buildMixedStrides(builder, inDims);
        const flatX = ensureFlatInput(builder, rawX);
        const outDims = resolveResizeOutDims(builder, info, inDims, targetShapeNode, "rz", op.id);

        // Separate unchanged axes (handled by AGU pointer) from resized axes
        const resizedAxes: number[] = [];
        const baseStrides: (number | ConcreteValueNode)[] = new Array(rank).fill(0);
        for (let d = 0; d < rank; d++) {
            if (!info.resizedMask[d]) {
                baseStrides[d] = inStrides[d];
            } else {
                resizedAxes.push(d);
            }
        }

        let coords: ConcreteValueNode[];
        let basePtr: ConcreteValueNode;
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                outDims,
                `rz_cnt_${op.id}`,
            );
            coords = counters;
            basePtr = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                outDims,
                baseStrides,
                `rz_base_${op.id}`,
            );
        } else {
            coords = decodeMixedRadix(builder, iter, outDims, `rz_dec_${op.id}`);
            basePtr = buildLinearIndex(builder, coords, baseStrides, `rz_lin_${op.id}`);
        }

        const cF = (tag: string, v: number) =>
            builder.createConstant(`rz_${tag}_${op.id}`, makeTensorProto(DataType.FLOAT, [], [v]));
        const zeroF = cF("zf", 0.0);
        const halfF = cF("hf", 0.5);
        const oneF = cF("of", 1.0);
        const oneI64 = builder.createConstant(`rz_o64_${op.id}`, scalarInt64(1));
        const twoI64 = builder.createConstant(`rz_two64_${op.id}`, scalarInt64(2));

        // Maps an integer output coordinate along axis `d` to floating-point input coordinate
        const mapCoordToFloat = (d: number): ConcreteValueNode =>
            mapResizeCoordToFloat(
                builder,
                coords[d],
                inDims[d],
                outDims[d],
                coordMode,
                halfF,
                zeroF,
                oneI64,
                twoI64,
                "rz",
                d,
                op.id,
            );

        if (mode === "nearest") {
            let flatInIdx: ConcreteValueNode = basePtr;
            for (const d of resizedAxes) {
                const fCoord = mapCoordToFloat(d);
                let snapped: ConcreteValueNode;
                if (nearestMode === "floor") {
                    [snapped] = builder.createOp("Floor", [fCoord]);
                } else if (nearestMode === "ceil") {
                    const [negF] = builder.createOp("Sub", [zeroF, fCoord]);
                    const [flNeg] = builder.createOp("Floor", [negF]);
                    [snapped] = builder.createOp("Sub", [zeroF, flNeg]);
                } else if (nearestMode === "round_prefer_ceil") {
                    const [plusH] = builder.createOp("Add", [fCoord, halfF]);
                    [snapped] = builder.createOp("Floor", [plusH]);
                } else {
                    // round_prefer_floor: ceil(x - 0.5)
                    const [minusH] = builder.createOp("Sub", [fCoord, halfF]);
                    const [negM] = builder.createOp("Sub", [zeroF, minusH]);
                    const [flNeg] = builder.createOp("Floor", [negM]);
                    [snapped] = builder.createOp("Sub", [zeroF, flNeg]);
                }

                const inD = inDims[d];
                const maxF =
                    typeof inD === "number"
                        ? cF(`max_${d}`, inD - 1)
                        : builder.createOp("Cast", [builder.createOp("Sub", [inD, oneI64])[0]], {
                              to: DataType.FLOAT,
                          })[0];
                const [clampedF] = builder.createOp("Clip", [snapped, zeroF, maxF]);
                const [iCoord] = builder.createOp("Cast", [clampedF], { to: DataType.INT64 });

                const str = inStrides[d];
                const term =
                    typeof str !== "number"
                        ? builder.createOp("Mul", [iCoord, str])[0]
                        : str === 1
                          ? iCoord
                          : builder.createOp("Mul", [
                                iCoord,
                                builder.createConstant(`rz_str_${d}_${op.id}`, scalarInt64(str)),
                            ])[0];
                [flatInIdx] = builder.createOp("Add", [flatInIdx, term]);
            }

            return gatherScalar(builder, flatX, flatInIdx);
        }

        // Linear / Bilinear interpolation across 0, 1, or 2 resized axes
        if (resizedAxes.length === 0) {
            return gatherScalar(builder, flatX, basePtr);
        }

        const axisInfo = resizedAxes.map((d) => {
            const rawF = mapCoordToFloat(d);
            const inD = inDims[d];
            const maxI =
                typeof inD === "number"
                    ? builder.createConstant(`rz_maxi_${d}_${op.id}`, scalarInt64(inD - 1))
                    : builder.createOp("Sub", [inD, oneI64])[0];
            const maxF =
                typeof inD === "number"
                    ? cF(`lmax_${d}`, inD - 1)
                    : builder.createOp("Cast", [maxI], { to: DataType.FLOAT })[0];
            const [clampedF] = builder.createOp("Clip", [rawF, zeroF, maxF]);
            const [loF] = builder.createOp("Floor", [clampedF]);
            const [frac] = builder.createOp("Sub", [clampedF, loF]);
            const [oneMinusFrac] = builder.createOp("Sub", [oneF, frac]);

            const [loI] = builder.createOp("Cast", [loF], { to: DataType.INT64 });
            const [loPlus1] = builder.createOp("Add", [loI, oneI64]);
            const [hiI] = builder.createOp("Clip", [
                loPlus1,
                builder.createConstant(`rz_zi_${d}_${op.id}`, scalarInt64(0)),
                maxI,
            ]);

            const str = inStrides[d];
            const scaleStride = (idx: ConcreteValueNode, tag: string) =>
                typeof str !== "number"
                    ? builder.createOp("Mul", [idx, str])[0]
                    : str === 1
                      ? idx
                      : builder.createOp("Mul", [
                            idx,
                            builder.createConstant(`rz_ls_${tag}_${d}_${op.id}`, scalarInt64(str)),
                        ])[0];

            return {
                loOff: scaleStride(loI, "lo"),
                hiOff: scaleStride(hiI, "hi"),
                w0: oneMinusFrac,
                w1: frac,
            };
        });

        const gatherAt = (idx: ConcreteValueNode, _tag: string): ConcreteValueNode => {
            const sq = gatherScalar(builder, flatX, idx);
            return dtype === DataType.FLOAT
                ? sq
                : builder.createOp("Cast", [sq], { to: DataType.FLOAT })[0];
        };

        let interpF: ConcreteValueNode;
        if (axisInfo.length === 1) {
            const a0 = axisInfo[0];
            const v0 = gatherAt(builder.createOp("Add", [basePtr, a0.loOff])[0], "0");
            const v1 = gatherAt(builder.createOp("Add", [basePtr, a0.hiOff])[0], "1");
            const [t0] = builder.createOp("Mul", [v0, a0.w0]);
            const [t1] = builder.createOp("Mul", [v1, a0.w1]);
            [interpF] = builder.createOp("Add", [t0, t1]);
        } else {
            const [a0, a1] = axisInfo;
            const [r0] = builder.createOp("Add", [basePtr, a0.loOff]);
            const [r1] = builder.createOp("Add", [basePtr, a0.hiOff]);
            const v00 = gatherAt(builder.createOp("Add", [r0, a1.loOff])[0], "00");
            const v01 = gatherAt(builder.createOp("Add", [r0, a1.hiOff])[0], "01");
            const v10 = gatherAt(builder.createOp("Add", [r1, a1.loOff])[0], "10");
            const v11 = gatherAt(builder.createOp("Add", [r1, a1.hiOff])[0], "11");

            const [top0] = builder.createOp("Mul", [v00, a1.w0]);
            const [top1] = builder.createOp("Mul", [v01, a1.w1]);
            const [top] = builder.createOp("Add", [top0, top1]);

            const [bot0] = builder.createOp("Mul", [v10, a1.w0]);
            const [bot1] = builder.createOp("Mul", [v11, a1.w1]);
            const [bot] = builder.createOp("Add", [bot0, bot1]);

            const [f0] = builder.createOp("Mul", [top, a0.w0]);
            const [f1] = builder.createOp("Mul", [bot, a0.w1]);
            [interpF] = builder.createOp("Add", [f0, f1]);
        }

        return dtype === DataType.FLOAT
            ? interpF
            : builder.createOp("Cast", [interpF], { to: dtype })[0];
    }
}
