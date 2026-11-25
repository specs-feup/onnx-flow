import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    computeStrides,
    getIntAttr,
    getStringAttr,
    makeTensorProto,
    readConstIntegerVectorFromTensorNode,
    readVectorFromTensorNode,
    scalarInt64,
    toStaticShape,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

const SUPPORTED_COORD_MODES = new Set([
    "half_pixel",
    "pytorch_half_pixel",
    "align_corners",
    "asymmetric",
]);

export class LowerResizeRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerResize";
    public readonly targetOp = "Resize";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["imaging", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
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

        const inDims = toStaticShape(ins[0].shape);
        if (inDims.length === 0 || inDims.includes(-1)) return false;

        const outDims = this.resolveOutDims(op, inDims);
        if (!outDims || outDims.length !== inDims.length || outDims.includes(-1)) return false;

        if (mode === "linear") {
            const resizedAxes = inDims.filter((d, i) => d !== outDims[i]).length;
            return resizedAxes <= 2; // Supports 1D and 2D (bilinear) linear interpolation
        }

        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        if (!SUPPORTED_COORD_MODES.has(coordMode)) return false;
        if (getIntAttr(op, "antialias", 0) !== 0) return false;

        return true;
    }

    private resolveOutDims(op: OperationNode.Class, inDims: number[]): number[] | undefined {
        const ins = op.getInputs()!;
        const sizesNode = ins.length >= 4 && ins[3]?.id ? ins[3] : undefined;
        const scalesNode = ins.length >= 3 && ins[2]?.id ? ins[2] : undefined;

        const sizes = sizesNode ? readConstIntegerVectorFromTensorNode(sizesNode) : undefined;
        if (sizes && sizes.length === inDims.length) return sizes;

        const scales = scalesNode ? readVectorFromTensorNode(scalesNode) : undefined;
        if (scales && scales.length === inDims.length) {
            return inDims.map((d, i) => Math.floor(d * scales[i]));
        }

        const outShape = toStaticShape(op.getOutputs()[0]?.shape);
        if (outShape.length === inDims.length && !outShape.includes(-1)) return outShape;
        return undefined;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const inDims = toStaticShape(op.getInputs()![0].shape);
        const outDims = this.resolveOutDims(op, inDims)!;
        const total = outDims.reduce((a, b) => a * b, 1);
        return { totalIters: total, carryShape: outDims };
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
        const inDims = toStaticShape(X.shape);
        const outDims = this.resolveOutDims(op, inDims)!;
        const dtype = X.literalType as DataType;
        const rank = inDims.length;

        const mode = getStringAttr(op, "mode", "nearest");
        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        const nearestMode = getStringAttr(op, "nearest_mode", "round_prefer_floor");
        const inStrides = computeStrides(inDims);

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
        const flatX = ensureFlatInput(builder, rawX);

        // Separate unchanged axes (handled by AGU pointer) from resized axes
        const resizedAxes: number[] = [];
        const baseStrides = new Array<number>(rank).fill(0);
        for (let d = 0; d < rank; d++) {
            if (inDims[d] === outDims[d]) {
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

        // Maps an integer output coordinate along axis `d` to floating-point input coordinate
        const mapCoordToFloat = (d: number): ConcreteValueNode => {
            const [cFloat] = builder.createOp("Cast", [coords[d]], { to: DataType.FLOAT });
            const inD = inDims[d];
            const outD = outDims[d];
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
            // half_pixel (default) & pytorch_half_pixel (outD > 1): (x + 0.5) * (inD/outD) - 0.5
            const [plusHalf] = builder.createOp("Add", [cFloat, halfF]);
            const [scaled] = builder.createOp("Mul", [plusHalf, cF(`hp_${d}`, invScale)]);
            return builder.createOp("Sub", [scaled, halfF])[0];
        };

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

                const maxF = cF(`max_${d}`, inDims[d] - 1);
                const [clampedF] = builder.createOp("Clip", [snapped, zeroF, maxF]);
                const [iCoord] = builder.createOp("Cast", [clampedF], { to: DataType.INT64 });

                const str = inStrides[d];
                const term =
                    str === 1
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
            const maxF = cF(`lmax_${d}`, inDims[d] - 1);
            const [clampedF] = builder.createOp("Clip", [rawF, zeroF, maxF]);
            const [loF] = builder.createOp("Floor", [clampedF]);
            const [frac] = builder.createOp("Sub", [clampedF, loF]);
            const [oneMinusFrac] = builder.createOp("Sub", [oneF, frac]);

            const [loI] = builder.createOp("Cast", [loF], { to: DataType.INT64 });
            const maxI = builder.createConstant(
                `rz_maxi_${d}_${op.id}`,
                scalarInt64(inDims[d] - 1),
            );
            const [loPlus1] = builder.createOp("Add", [loI, oneI64]);
            const [hiI] = builder.createOp("Clip", [
                loPlus1,
                builder.createConstant(`rz_zi_${d}_${op.id}`, scalarInt64(0)),
                maxI,
            ]);

            const str = inStrides[d];
            const scaleStride = (idx: ConcreteValueNode, tag: string) =>
                str === 1
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
