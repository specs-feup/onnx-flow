import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { DataType } from "../../../OnnxTypes.js";
import { type ValueNode, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import {
    computeStrides,
    getIntAttr,
    getIntsAttr,
    getStringAttr,
    inferConvDim,
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

export class LowerConvNDRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerConvND";
    public readonly targetOp = "Conv";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "static" as const;
    public readonly tags = ["conv", "nd", "self-carry"] as const;
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
        if (op.type !== "Conv") return false;
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

    private resolvePadsAndOutDims(
        op: OperationNode.Class,
        xDims: number[],
        wDims: number[],
    ): { pads: number[]; outDims: number[] } {
        const spatialRank = xDims.length - 2;
        const kernelDims = wDims.slice(2);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");

        let pads = getIntsAttr(op, "pads", new Array<number>(spatialRank * 2).fill(0));
        const outSpatial: number[] = [];

        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            const pHead: number[] = [];
            const pTail: number[] = [];
            for (let d = 0; d < spatialRank; d++) {
                const inD = xDims[2 + d];
                const s = strides[d] ?? 1;
                const dil = dilations[d] ?? 1;
                const effK = (kernelDims[d] - 1) * dil + 1;
                const outD = Math.ceil(inD / s);
                const padTotal = Math.max((outD - 1) * s + effK - inD, 0);
                const head =
                    autoPad === "SAME_LOWER" ? Math.ceil(padTotal / 2) : Math.floor(padTotal / 2);
                pHead.push(head);
                pTail.push(padTotal - head);
                outSpatial.push(outD);
            }
            pads = [...pHead, ...pTail];
        } else {
            if (autoPad === "VALID") {
                pads = new Array<number>(spatialRank * 2).fill(0);
            }
            for (let d = 0; d < spatialRank; d++) {
                outSpatial.push(
                    inferConvDim(
                        xDims[2 + d],
                        kernelDims[d],
                        strides[d] ?? 1,
                        pads[d] ?? 0,
                        pads[spatialRank + d] ?? 0,
                        dilations[d] ?? 1,
                    ),
                );
            }
        }

        return { pads, outDims: [xDims[0], wDims[0], ...outSpatial] };
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): { totalIters: number; carryShape: number[] } {
        const ins = op.getInputs()!;
        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        const { outDims } = this.resolvePadsAndOutDims(op, xDims, wDims);

        const outTotal = outDims.reduce((a, b) => a * b, 1);
        const cInPerGroup = wDims[1];
        const kTotal = wDims.slice(2).reduce((a, b) => a * b, 1);
        return { totalIters: outTotal * cInPerGroup * kTotal, carryShape: outDims };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        const ins = op.getInputs()!;
        if (ins.length < 3 || ins[2].id === "") return loopOut;

        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        const { outDims } = this.resolvePadsAndOutDims(op, xDims, wDims);
        const dtype = ins[0].literalType as DataType;

        const outShapeConst = builder.createConstant(`cnd_outshape_${op.id}`, int64Vec(outDims));
        const [reshaped] = builder.createOp("Reshape", [loopOut, outShapeConst], {}, [
            { type: dtype, shape: outDims as KnownShape },
        ]);

        const biasShape = [1, outDims[1], ...new Array<number>(outDims.length - 2).fill(1)];
        const biasShapeConst = builder.createConstant(`cnd_bshape_${op.id}`, int64Vec(biasShape));
        const [biasBcast] = builder.createOp("Reshape", [ins[2], biasShapeConst], {}, [
            { type: dtype, shape: biasShape as KnownShape },
        ]);

        const [added] = builder.createOp("Add", [reshaped, biasBcast], {}, [
            { type: dtype, shape: outDims as KnownShape },
        ]);
        const flatShape = builder.createConstant(
            `cnd_flat_${op.id}`,
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
        const builder = new GraphBuilder(body, `cnd_${op.id}`);
        const ins = op.getInputs()!;
        const xDims = toStaticShape(ins[0].shape);
        const wDims = toStaticShape(ins[1].shape);
        const dtype = ins[0].literalType as DataType;

        const spatialRank = xDims.length - 2;
        const group = getIntAttr(op, "group", 1);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const { pads, outDims } = this.resolvePadsAndOutDims(op, xDims, wDims);

        const cInPerGroup = wDims[1];
        const cOutPerGroup = Math.floor(wDims[0] / group);
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
            cOutPerGroup * wStrides[0],
            wStrides[0],
            ...new Array<number>(spatialRank).fill(0),
            wStrides[1],
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
                `cnd_cnt_${op.id}`,
            );
            coords = counters;
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                outPtrStrides,
                `cnd_o_${op.id}`,
            );
            xIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                xBaseStrides,
                `cnd_x_${op.id}`,
            );
            wIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                wPtrStrides,
                `cnd_w_${op.id}`,
            );
        } else {
            coords = decodeMixedRadix(builder, iter, iterDims, `cnd_dec_${op.id}`);
            flatOutIdx = buildLinearIndex(builder, coords, outPtrStrides, `cnd_olin_${op.id}`);
            xIdx = buildLinearIndex(builder, coords, xBaseStrides, `cnd_xlin_${op.id}`);
            wIdx = buildLinearIndex(builder, coords, wPtrStrides, `cnd_wlin_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`cnd_z64_${op.id}`, scalarInt64(0));
        const zeroVal = builder.createConstant(
            `cnd_zval_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );

        let isValid: ConcreteValueNode | null = null;
        for (let d = 0; d < spatialRank; d++) {
            const oCoord = coords[3 + d];
            const kCoord = coords[3 + spatialRank + 1 + d];
            const sVal = strides[d] ?? 1;
            const pVal = pads[d] ?? 0;
            const dilVal = dilations[d] ?? 1;

            let inCoord: ConcreteValueNode =
                sVal === 1
                    ? oCoord
                    : builder.createOp("Mul", [
                          oCoord,
                          builder.createConstant(`cnd_st_${d}_${op.id}`, scalarInt64(sVal)),
                      ])[0];

            const kOff: ConcreteValueNode =
                dilVal === 1
                    ? kCoord
                    : builder.createOp("Mul", [
                          kCoord,
                          builder.createConstant(`cnd_dil_${d}_${op.id}`, scalarInt64(dilVal)),
                      ])[0];

            [inCoord] = builder.createOp("Add", [inCoord, kOff]);
            if (pVal !== 0) {
                [inCoord] = builder.createOp("Sub", [
                    inCoord,
                    builder.createConstant(`cnd_p_${d}_${op.id}`, scalarInt64(pVal)),
                ]);
            }

            const dimConst = builder.createConstant(
                `cnd_dim_${d}_${op.id}`,
                scalarInt64(xDims[2 + d]),
            );
            const [ltZero] = builder.createOp("Less", [inCoord, zeroI64]);
            const [geZero] = builder.createOp("Not", [ltZero]);
            const [ltDim] = builder.createOp("Less", [inCoord, dimConst]);
            const [dimValid] = builder.createOp("And", [geZero, ltDim]);
            isValid = isValid ? builder.createOp("And", [isValid, dimValid])[0] : dimValid;

            const [safeCoord] = builder.createOp("Where", [dimValid, inCoord, zeroI64]);
            const str = xStrides[2 + d];
            const term =
                str === 1
                    ? safeCoord
                    : builder.createOp("Mul", [
                          safeCoord,
                          builder.createConstant(`cnd_xstr_${d}_${op.id}`, scalarInt64(str)),
                      ])[0];
            [xIdx] = builder.createOp("Add", [xIdx, term]);
        }

        const [xUnsq] = builder.createOp("Unsqueeze", [xIdx, axes]);
        const [wUnsq] = builder.createOp("Unsqueeze", [wIdx, axes]);
        const [xRaw] = builder.createOp("Gather", [flatX, xUnsq], { axis: 0 });
        const [wRaw] = builder.createOp("Gather", [flatW, wUnsq], { axis: 0 });
        const xVal = squeezeIfLen1(builder, xRaw, axes, `cnd_sqx_${op.id}`) as ConcreteValueNode;
        const wVal = squeezeIfLen1(builder, wRaw, axes, `cnd_sqw_${op.id}`) as ConcreteValueNode;

        const [rawProd] = builder.createOp("Mul", [xVal, wVal]);
        const [prod] = builder.createOp("Where", [isValid!, rawProd, zeroVal]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const [accRaw] = builder.createOp("Gather", [carryNode, outUnsq], { axis: 0 });
        const curAcc = squeezeIfLen1(
            builder,
            accRaw,
            axes,
            `cnd_sqa_${op.id}`,
        ) as ConcreteValueNode;

        const [newSum] = builder.createOp("Add", [curAcc, prod]);
        const [newSumUnsq] = builder.createOp("Unsqueeze", [newSum, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, newSumUnsq], {
            axis: 0,
        });

        return { resultNode: newSum, nextCarry };
    }
}
