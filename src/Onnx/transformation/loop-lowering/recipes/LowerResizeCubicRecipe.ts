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
    getFloatAttr,
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
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
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
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["imaging", "cubic", "fuseable"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
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
        if (inDims.length === 0 || inDims.includes(-1)) return false;

        const outDims = this.resolveOutDims(op, inDims);
        if (!outDims || outDims.length !== inDims.length || outDims.includes(-1)) return false;

        const resizedAxes = inDims.filter((d, i) => d !== outDims[i]).length;
        return resizedAxes <= 2;
    }

    private resolveOutDims(op: OperationNode.Class, inDims: number[]): number[] | undefined {
        const ins = op.getInputs()!;
        const sizesNode = ins.length >= 4 && ins[3].id !== "" ? ins[3] : undefined;
        const scalesNode = ins.length >= 3 && ins[2].id !== "" ? ins[2] : undefined;

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
        const builder = new GraphBuilder(body, `rzcub_${op.id}`);
        const ins = op.getInputs()!;
        const X = ins[0];
        const inDims = toStaticShape(X.shape);
        const outDims = this.resolveOutDims(op, inDims)!;
        const dtype = X.literalType as DataType;
        const rank = inDims.length;

        const coordMode = getStringAttr(op, "coordinate_transformation_mode", "half_pixel");
        const coeffA = getFloatAttr(op, "cubic_coeff_a", -0.75);
        const excludeOutside = getIntAttr(op, "exclude_outside", 0) !== 0;
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
            const [idxUnsq] = builder.createOp("Unsqueeze", [basePtr, axes]);
            const [raw] = builder.createOp("Gather", [flatX, idxUnsq], { axis: 0 });
            return squeezeIfLen1(builder, raw, axes, `rzc_sq0_${op.id}`);
        }

        const cF = (tag: string, v: number) =>
            builder.createConstant(`rzc_${tag}_${op.id}`, makeTensorProto(DataType.FLOAT, [], [v]));
        const zeroF = cF("zf", 0.0);
        const halfF = cF("hf", 0.5);
        const oneF = cF("of", 1.0);
        const zeroI64 = builder.createConstant(`rzc_z64_${op.id}`, scalarInt64(0));

        // Cubic convolution kernel coefficients for parameter `a`
        const aC = cF("a", coeffA);
        const negAC = cF("na", -coeffA);
        const neg2AC = cF("n2a", -2.0 * coeffA);
        const aPlus2C = cF("ap2", coeffA + 2.0);
        const negAPlus2C = cF("nap2", -(coeffA + 2.0));
        const aPlus3C = cF("ap3", coeffA + 3.0);
        const twoAPlus3C = cF("2ap3", 2.0 * coeffA + 3.0);

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
            const [plusHalf] = builder.createOp("Add", [cFloat, halfF]);
            const [scaled] = builder.createOp("Mul", [plusHalf, cF(`hp_${d}`, invScale)]);
            return builder.createOp("Sub", [scaled, halfF])[0];
        };

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
            const dimI = builder.createConstant(`rzc_dim_${d}_${op.id}`, scalarInt64(inDims[d]));
            const maxI = builder.createConstant(
                `rzc_max_${d}_${op.id}`,
                scalarInt64(inDims[d] - 1),
            );
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
                    str === 1
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

        const gatherAt = (idx: ConcreteValueNode, tag: string): ConcreteValueNode => {
            const [u] = builder.createOp("Unsqueeze", [idx, axes]);
            const [g] = builder.createOp("Gather", [flatX, u], { axis: 0 });
            const sq = squeezeIfLen1(
                builder,
                g,
                axes,
                `rzc_sq_${tag}_${op.id}`,
            ) as ConcreteValueNode;
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
