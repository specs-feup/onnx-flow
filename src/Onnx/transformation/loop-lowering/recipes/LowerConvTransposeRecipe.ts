import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { DataType } from "../../../OnnxTypes.js";
import { type ValueNode, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import {
    computeStrides,
    getIntAttr,
    getIntsAttr,
    int64Vec,
    makeTensorProto,
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

export class LowerConvTransposeRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerConvTranspose";
    public readonly targetOp = "ConvTranspose";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "Less",
        "And",
        "Not",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "ConvTranspose") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        return (
            xDims.length >= 3 &&
            xDims.length === wDims.length &&
            !xDims.includes(-1) &&
            !wDims.includes(-1)
        );
    }

    private computeOutDims(op: OperationNode.Class, xDims: number[], wDims: number[]): number[] {
        const spatialRank = xDims.length - 2;
        const group = getIntAttr(op, "group", 1);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const pads = getIntsAttr(op, "pads", new Array<number>(spatialRank * 2).fill(0));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const outPadding = getIntsAttr(
            op,
            "output_padding",
            new Array<number>(spatialRank).fill(0),
        );
        const explicitOutShape = getIntsAttr(op, "output_shape", []);

        const outSpatial: number[] = [];
        for (let d = 0; d < spatialRank; d++) {
            if (explicitOutShape.length === spatialRank) {
                outSpatial.push(explicitOutShape[d]);
            } else {
                const k = wDims[2 + d];
                const s = strides[d] ?? 1;
                const pHead = pads[d] ?? 0;
                const pTail = pads[spatialRank + d] ?? 0;
                const dil = dilations[d] ?? 1;
                const opad = outPadding[d] ?? 0;
                outSpatial.push(s * (xDims[2 + d] - 1) + opad + (k - 1) * dil + 1 - pHead - pTail);
            }
        }
        return [xDims[0], wDims[1] * group, ...outSpatial];
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const ins = op.getInputs()!;
        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        const group = getIntAttr(op, "group", 1);
        const outDims = this.computeOutDims(op, xDims, wDims);

        const outTotal = outDims.reduce((a, b) => a * b, 1);
        const cInPerGroup = xDims[1] / group;
        const kTotal = wDims.slice(2).reduce((a, b) => a * b, 1);
        return { totalIters: outTotal * cInPerGroup * kTotal, carryShape: outDims };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        const ins = op.getInputs()!;
        if (ins.length < 3 || ins[2].id === "") return loopOut;

        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        const outDims = this.computeOutDims(op, xDims, wDims);
        const dtype = ins[0].literalType as DataType;

        const outShapeConst = builder.createConstant(`ct_outshape_${op.id}`, int64Vec(outDims));
        const [reshaped] = builder.createOp("Reshape", [loopOut, outShapeConst], {}, [
            { type: dtype, shape: outDims as KnownShape },
        ]);

        const biasShape = [1, outDims[1], ...new Array<number>(outDims.length - 2).fill(1)];
        const biasShapeConst = builder.createConstant(`ct_bshape_${op.id}`, int64Vec(biasShape));
        const [biasBcast] = builder.createOp("Reshape", [ins[2], biasShapeConst], {}, [
            { type: dtype, shape: biasShape as KnownShape },
        ]);

        const [added] = builder.createOp("Add", [reshaped, biasBcast], {}, [
            { type: dtype, shape: outDims as KnownShape },
        ]);
        const flatShape = builder.createConstant(
            `ct_flat_${op.id}`,
            int64Vec([outDims.reduce((a, b) => a * b, 1)]),
        );
        return builder.createOp("Reshape", [added, flatShape])[0];
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        axes: ConcreteValueNode,
        outShape: KnownShape,
        carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `ct_${op.id}`);
        const ins = op.getInputs()!;
        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        const dtype = ins[0].literalType as DataType;

        const spatialRank = xDims.length - 2;
        const group = getIntAttr(op, "group", 1);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const pads = getIntsAttr(op, "pads", new Array<number>(spatialRank * 2).fill(0));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));

        const outDims = this.computeOutDims(op, xDims, wDims);
        const cInPerGroup = xDims[1] / group;
        const cOutPerGroup = wDims[1];
        const kernelDims = wDims.slice(2);

        // Iteration space: [N, G, cOutG, ...outSpatial, cInG, ...kernelDims]
        const iterDims = [
            outDims[0],
            group,
            cOutPerGroup,
            ...outDims.slice(2),
            cInPerGroup,
            ...kernelDims,
        ];

        const xStrides = computeStrides(xDims);
        const wStrides = computeStrides(wDims);
        const outStrides = computeStrides(outDims);

        const rawX = resolveRecipeInput(
            builder,
            ins[0],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const rawW = resolveRecipeInput(
            builder,
            ins[1],
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
        const flatW = ensureFlatInput(builder, rawW);

        // Stride vectors over iterDims
        const outPtrStrides = [
            outStrides[0],
            cOutPerGroup * outStrides[1],
            outStrides[1],
            ...outStrides.slice(2),
            0,
            ...new Array<number>(spatialRank).fill(0),
        ];
        const xBaseStrides = [
            xStrides[0],
            cInPerGroup * xStrides[1],
            0,
            ...new Array<number>(spatialRank).fill(0),
            xStrides[1],
            ...new Array<number>(spatialRank).fill(0),
        ];
        const wPtrStrides = [
            0,
            cInPerGroup * wStrides[0],
            wStrides[1],
            ...new Array<number>(spatialRank).fill(0),
            wStrides[0],
            ...wStrides.slice(2),
        ];

        let coords: ConcreteValueNode[];
        let flatOutIdx: ConcreteValueNode;
        let xIdx: ConcreteValueNode;
        let wIdx: ConcreteValueNode;

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                `ct_cnt_${op.id}`,
            );
            coords = counters;
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                outPtrStrides,
                `ct_o_${op.id}`,
            );
            xIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                xBaseStrides,
                `ct_x_${op.id}`,
            );
            wIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                wPtrStrides,
                `ct_w_${op.id}`,
            );
        } else {
            coords = decodeMixedRadix(builder, iter, iterDims, `ct_dec_${op.id}`);
            flatOutIdx = buildLinearIndex(builder, coords, outPtrStrides, `ct_olin_${op.id}`);
            xIdx = buildLinearIndex(builder, coords, xBaseStrides, `ct_xlin_${op.id}`);
            wIdx = buildLinearIndex(builder, coords, wPtrStrides, `ct_wlin_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`ct_z64_${op.id}`, scalarInt64(0));
        const zeroVal = builder.createConstant(`ct_zval_${op.id}`, makeTensorProto(dtype, [], [0]));

        let isValid: ConcreteValueNode | null = null;
        for (let d = 0; d < spatialRank; d++) {
            const oCoord = coords[3 + d];
            const kCoord = coords[3 + spatialRank + 1 + d];
            const sVal = strides[d] ?? 1;
            const pVal = pads[d] ?? 0;
            const dilVal = dilations[d] ?? 1;

            // num = oCoord + pVal - kCoord * dilVal
            let num: ConcreteValueNode = oCoord;
            if (pVal !== 0) {
                [num] = builder.createOp("Add", [
                    num,
                    builder.createConstant(`ct_p_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }
            const kOff =
                dilVal === 1
                    ? kCoord
                    : builder.createOp("Mul", [
                          kCoord,
                          builder.createConstant(`ct_dil_${d}_${op.id}`, scalarInt64(dilVal)),
                      ])[0];
            [num] = builder.createOp("Sub", [num, kOff]);

            const [ltZero] = builder.createOp("Less", [num, zeroI64]);
            const [geZero] = builder.createOp("Not", [ltZero]);

            let inCoord: ConcreteValueNode = num;
            let divValid: ConcreteValueNode = geZero;
            if (sVal !== 1) {
                const sConst = builder.createConstant(`ct_st_${d}_${op.id}`, scalarInt64(sVal));
                const [safeNum] = builder.createOp("Where", [geZero, num, zeroI64]);
                const [rem] = builder.createOp("Mod", [safeNum, sConst]);
                const [remZero] = builder.createOp("Equal", [rem, zeroI64]);
                [divValid] = builder.createOp("And", [geZero, remZero]);
                [inCoord] = builder.createOp("Div", [safeNum, sConst]);
            }

            const dimConst = builder.createConstant(
                `ct_dim_${d}_${op.id}`,
                scalarInt64(xDims[2 + d]),
            );
            const [ltDim] = builder.createOp("Less", [inCoord, dimConst]);
            const [dimValid] = builder.createOp("And", [divValid, ltDim]);
            isValid = isValid ? builder.createOp("And", [isValid, dimValid])[0] : dimValid;

            const [safeCoord] = builder.createOp("Where", [dimValid, inCoord, zeroI64]);
            const str = xStrides[2 + d];
            const term =
                str === 1
                    ? safeCoord
                    : builder.createOp("Mul", [
                          safeCoord,
                          builder.createConstant(`ct_xstr_${d}_${op.id}`, scalarInt64(str)),
                      ])[0];
            [xIdx] = builder.createOp("Add", [xIdx, term]);
        }

        const [xUnsq] = builder.createOp("Unsqueeze", [xIdx, axes]);
        const [wUnsq] = builder.createOp("Unsqueeze", [wIdx, axes]);
        const [xRaw] = builder.createOp("Gather", [flatX, xUnsq], { axis: 0 });
        const [wRaw] = builder.createOp("Gather", [flatW, wUnsq], { axis: 0 });
        const xVal = squeezeIfLen1(builder, xRaw, axes, `ct_sqx_${op.id}`) as ConcreteValueNode;
        const wVal = squeezeIfLen1(builder, wRaw, axes, `ct_sqw_${op.id}`) as ConcreteValueNode;

        const [rawProd] = builder.createOp("Mul", [xVal, wVal]);
        const [prod] = builder.createOp("Where", [isValid!, rawProd, zeroVal]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const [accRaw] = builder.createOp("Gather", [carryNode, outUnsq], { axis: 0 });
        const curAcc = squeezeIfLen1(builder, accRaw, axes, `ct_sqa_${op.id}`) as ConcreteValueNode;

        const [newSum] = builder.createOp("Add", [curAcc, prod]);
        const [newSumUnsq] = builder.createOp("Unsqueeze", [newSum, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, newSumUnsq], {
            axis: 0,
        });

        return { resultNode: newSum, nextCarry };
    }
}
