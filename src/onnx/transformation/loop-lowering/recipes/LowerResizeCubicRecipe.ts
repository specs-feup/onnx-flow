import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    getFloatAttr,
    getIntAttr,
    getStringAttr,
    makeTensorProto,
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
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import {
    analyzeResizeAxes,
    computeResizeLoopBounds,
    mapResizeCoordToFloat,
    resolveResizeOutDims,
} from "./LowerResizeRecipe.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

const SUPPORTED_COORD_MODES = new Set([
    "half_pixel",
    "pytorch_half_pixel",
    "align_corners",
    "asymmetric",
]);

export class LowerResizeCubicRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerResizeCubic";
    public readonly targetOp = "Resize";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["imaging", "cubic", "fuseable"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
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
        "Div",
        "Mod",
        "Floor",
        "Clip",
        "Less",
        "Not",
        "Equal",
        "And",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Resize") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 3) return false;
        if (getStringAttr(op, "mode", "nearest") !== "cubic") return false;

        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        if (!SUPPORTED_COORD_MODES.has(coordMode)) return false;
        if (getIntAttr(op, "antialias", 0) !== 0) return false;

        const inDims = toStaticShape(ins[0].shape);
        if (inDims.length === 0) return false;

        const info = analyzeResizeAxes(op, inDims);
        if (!info.canResolve) return false;

        return info.resizedMask.filter(Boolean).length <= 2;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        return computeResizeLoopBounds(op, "rzc");
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
        const builder = new GraphBuilder(body, `rzcub_${op.id}`);
        const ins = op.getInputs()!;
        const X = ins[0];
        const rawInDims = toStaticShape(X.shape);
        const info = analyzeResizeAxes(op, rawInDims);
        const dtype = X.literalType as DataType;
        const rank = rawInDims.length;

        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        const coeffA = getFloatAttr(op, "cubic_coeff_a", -0.75);
        const excludeOutside = getIntAttr(op, "exclude_outside", 0) !== 0;

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
        const inDims = extractMixedDims(builder, rawX, rawInDims, `rzc_id_${op.id}`);
        const inStrides = buildMixedStrides(builder, inDims);
        const flatX = ensureFlatInput(builder, rawX);
        const outDims = resolveResizeOutDims(builder, info, inDims, targetShapeNode, "rzc", op.id);

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
                `rzc_cnt_${op.id}`,
            );
            coords = counters;
            basePtr = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                outDims,
                baseStrides,
                `rzc_base_${op.id}`,
            );
        } else {
            coords = decodeMixedRadix(builder, iter, outDims, `rzc_dec_${op.id}`);
            basePtr = buildLinearIndex(builder, coords, baseStrides, `rzc_lin_${op.id}`);
        }

        if (resizedAxes.length === 0) {
            return gatherScalar(builder, flatX, basePtr);
        }

        const cF = (tag: string, v: number) =>
            builder.createConstant(`rzc_${tag}_${op.id}`, makeTensorProto(DataType.FLOAT, [], [v]));
        const zeroF = cF("zf", 0.0);
        const halfF = cF("hf", 0.5);
        const oneF = cF("of", 1.0);
        const zeroI64 = builder.createConstant(`rzc_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`rzc_o64_${op.id}`, scalarInt64(1));
        const twoI64 = builder.createConstant(`rzc_t64_${op.id}`, scalarInt64(2));

        // Cubic convolution kernel coefficients for parameter `a`
        const aC = cF("a", coeffA);
        const negAC = cF("na", -coeffA);
        const neg2AC = cF("n2a", -2.0 * coeffA);
        const aPlus2C = cF("ap2", coeffA + 2.0);
        const negAPlus2C = cF("nap2", -(coeffA + 2.0));
        const aPlus3C = cF("ap3", coeffA + 3.0);
        const twoAPlus3C = cF("2ap3", 2.0 * coeffA + 3.0);

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
                "rzc",
                d,
                op.id,
            );

        // For each resized axis, compute 4 strided tap offsets and 4 cubic weights
        const axisInfo = resizedAxes.map((d) => {
            const rawF = mapCoordToFloat(d);
            const [flF] = builder.createOp("Floor", [rawF]);
            const [s] = builder.createOp("Sub", [rawF, flF]);
            const [s2] = builder.createOp("Mul", [s, s]);
            const [s3] = builder.createOp("Mul", [s2, s]);

            // w0 = a*s^3 - 2a*s^2 + a*s
            const [w0a] = builder.createOp("Mul", [aC, s3]);
            const [w0b] = builder.createOp("Mul", [neg2AC, s2]);
            const [w0c] = builder.createOp("Mul", [aC, s]);
            const [w0] = builder.createOp("Add", [builder.createOp("Add", [w0a, w0b])[0], w0c]);

            // w1 = (a + 2)*s^3 - (a + 3)*s^2 + 1
            const [w1a] = builder.createOp("Mul", [aPlus2C, s3]);
            const [w1b] = builder.createOp("Mul", [aPlus3C, s2]);
            const [w1] = builder.createOp("Add", [builder.createOp("Sub", [w1a, w1b])[0], oneF]);

            // w2 = -(a + 2)*s^3 + (2a + 3)*s^2 - a*s
            const [w2a] = builder.createOp("Mul", [negAPlus2C, s3]);
            const [w2b] = builder.createOp("Mul", [twoAPlus3C, s2]);
            const [w2c] = builder.createOp("Mul", [negAC, s]);
            const [w2] = builder.createOp("Add", [builder.createOp("Add", [w2a, w2b])[0], w2c]);

            // w3 = -a*s^3 + a*s^2
            const [w3a] = builder.createOp("Mul", [negAC, s3]);
            const [w3b] = builder.createOp("Mul", [aC, s2]);
            const [w3] = builder.createOp("Add", [w3a, w3b]);

            let weights: ConcreteValueNode[] = [w0, w1, w2, w3];

            const [baseI] = builder.createOp("Cast", [flF], { to: DataType.INT64 });
            const inD = inDims[d];
            const dimI =
                typeof inD === "number"
                    ? builder.createConstant(`rzc_dim_${d}_${op.id}`, scalarInt64(inD))
                    : inD;
            const maxI =
                typeof inD === "number"
                    ? builder.createConstant(`rzc_max_${d}_${op.id}`, scalarInt64(inD - 1))
                    : builder.createOp("Sub", [inD, oneI64])[0];
            const str = inStrides[d];

            const offsets: ConcreteValueNode[] = [];
            const tapOffsets = [-1, 0, 1, 2];

            for (let m = 0; m < 4; m++) {
                const offC = builder.createConstant(
                    `rzc_off_${d}_${m}_${op.id}`,
                    scalarInt64(tapOffsets[m]),
                );
                const [tapI] = builder.createOp("Add", [baseI, offC]);

                if (excludeOutside) {
                    const [lt0] = builder.createOp("Less", [tapI, zeroI64]);
                    const [ge0] = builder.createOp("Not", [lt0]);
                    const [ltD] = builder.createOp("Less", [tapI, dimI]);
                    const [inB] = builder.createOp("And", [ge0, ltD]);
                    weights[m] = builder.createOp("Where", [inB, weights[m], zeroF])[0];
                }

                const [clampedI] = builder.createOp("Clip", [tapI, zeroI64, maxI]);
                const strided =
                    typeof str !== "number"
                        ? builder.createOp("Mul", [clampedI, str])[0]
                        : str === 1
                          ? clampedI
                          : builder.createOp("Mul", [
                                clampedI,
                                builder.createConstant(
                                    `rzc_str_${d}_${m}_${op.id}`,
                                    scalarInt64(str),
                                ),
                            ])[0];
                offsets.push(strided);
            }

            if (excludeOutside) {
                const [s01] = builder.createOp("Add", [weights[0], weights[1]]);
                const [s23] = builder.createOp("Add", [weights[2], weights[3]]);
                const [wSum] = builder.createOp("Add", [s01, s23]);
                const [isZero] = builder.createOp("Equal", [wSum, zeroF]);
                const [safeSum] = builder.createOp("Where", [isZero, oneF, wSum]);
                weights = weights.map((w) => builder.createOp("Div", [w, safeSum])[0]);
            }

            return { offsets, weights };
        });

        const gatherAt = (idx: ConcreteValueNode, _tag: string): ConcreteValueNode => {
            const sq = gatherScalar(builder, flatX, idx);
            return dtype === DataType.FLOAT
                ? sq
                : builder.createOp("Cast", [sq], { to: DataType.FLOAT })[0];
        };

        const weightedSum4 = (
            vals: ConcreteValueNode[],
            ws: ConcreteValueNode[],
        ): ConcreteValueNode => {
            const t0 = builder.createOp("Mul", [vals[0], ws[0]])[0];
            const t1 = builder.createOp("Mul", [vals[1], ws[1]])[0];
            const t2 = builder.createOp("Mul", [vals[2], ws[2]])[0];
            const t3 = builder.createOp("Mul", [vals[3], ws[3]])[0];
            const a01 = builder.createOp("Add", [t0, t1])[0];
            const a23 = builder.createOp("Add", [t2, t3])[0];
            return builder.createOp("Add", [a01, a23])[0];
        };

        let interpF: ConcreteValueNode;
        if (axisInfo.length === 1) {
            const a0 = axisInfo[0];
            const vals = a0.offsets.map((off, m) =>
                gatherAt(builder.createOp("Add", [basePtr, off])[0], `${m}`),
            );
            interpF = weightedSum4(vals, a0.weights);
        } else {
            const [a0, a1] = axisInfo;
            const rowVals: ConcreteValueNode[] = [];
            for (let r = 0; r < 4; r++) {
                const [rowPtr] = builder.createOp("Add", [basePtr, a0.offsets[r]]);
                const colVals = a1.offsets.map((off, c) =>
                    gatherAt(builder.createOp("Add", [rowPtr, off])[0], `${r}_${c}`),
                );
                rowVals.push(weightedSum4(colVals, a1.weights));
            }
            interpF = weightedSum4(rowVals, a0.weights);
        }

        return dtype === DataType.FLOAT
            ? interpF
            : builder.createOp("Cast", [interpF], { to: dtype })[0];
    }
}
