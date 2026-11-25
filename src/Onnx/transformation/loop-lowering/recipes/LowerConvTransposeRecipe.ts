import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { DataType } from "../../../OnnxTypes.js";
import { type ValueNode, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    getIntsAttr,
    getStringAttr,
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
    buildMixedStrides,
    decodeMixedRadix,
    ensureFlatInput,
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    mulMixed,
    packShapeVector,
    resolveRecipeInput,
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
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["conv", "self-carry"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Concat",
        "ReduceProd",
        "Max",
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
        return xDims.length >= 3 && xDims.length === wDims.length;
    }

    private resolvePadsAndOutDimsMixed(
        op: OperationNode.Class,
        builder: GraphBuilder,
        xDims: (number | ConcreteValueNode)[],
        wDims: (number | ConcreteValueNode)[],
        tag: string,
    ): { pHeads: (number | ConcreteValueNode)[]; outDims: (number | ConcreteValueNode)[] } {
        const spatialRank = xDims.length - 2;
        const group = getIntAttr(op, "group", 1);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));
        const outPadding = getIntsAttr(
            op,
            "output_padding",
            new Array<number>(spatialRank).fill(0),
        );
        const explicitOutShape = getIntsAttr(op, "output_shape", []);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const explicitPads = getIntsAttr(op, "pads", []);

        const outC = mulMixed(builder, wDims[1], group, `${tag}_oc`);
        const pHeads: (number | ConcreteValueNode)[] = [];
        const outSpatial: (number | ConcreteValueNode)[] = [];

        const zeroI64 = builder.createConstant(`${tag}_z`, scalarInt64(0));
        const oneI64 = builder.createConstant(`${tag}_o`, scalarInt64(1));
        const twoI64 = builder.createConstant(`${tag}_two`, scalarInt64(2));

        for (let d = 0; d < spatialRank; d++) {
            const inD = xDims[2 + d];
            const k = wDims[2 + d];
            const s = strides[d] ?? 1;
            const dil = dilations[d] ?? 1;
            const opad = outPadding[d] ?? 0;

            if (typeof inD === "number" && typeof k === "number") {
                const unpadded = s * (inD - 1) + opad + (k - 1) * dil + 1;
                if (autoPad === "VALID") {
                    pHeads.push(0);
                    outSpatial.push(
                        explicitOutShape.length === spatialRank ? explicitOutShape[d] : unpadded,
                    );
                } else if (
                    autoPad === "SAME_UPPER" ||
                    autoPad === "SAME_LOWER" ||
                    explicitOutShape.length === spatialRank
                ) {
                    const outD =
                        explicitOutShape.length === spatialRank ? explicitOutShape[d] : inD * s;
                    const totalPad = Math.max(0, unpadded - outD);
                    const head =
                        autoPad === "SAME_UPPER"
                            ? Math.floor(totalPad / 2)
                            : Math.ceil(totalPad / 2);
                    pHeads.push(head);
                    outSpatial.push(outD);
                } else {
                    const pH =
                        explicitPads.length === spatialRank * 2 ? Number(explicitPads[d] ?? 0) : 0;
                    const pT =
                        explicitPads.length === spatialRank * 2
                            ? Number(explicitPads[spatialRank + d] ?? 0)
                            : 0;
                    pHeads.push(pH);
                    outSpatial.push(unpadded - pH - pT);
                }
                continue;
            }

            // Compute dynamic unpadded = s * (inD - 1) + opad + (k - 1) * dil + 1
            const inDM1 =
                typeof inD === "number" ? inD - 1 : builder.createOp("Sub", [inD, oneI64])[0];
            const sPart = mulMixed(builder, inDM1, s, `${tag}_sp_${d}`);
            const kM1 = typeof k === "number" ? k - 1 : builder.createOp("Sub", [k, oneI64])[0];
            const kPart = mulMixed(builder, kM1, dil, `${tag}_kp_${d}`);
            const biasVal = opad + 1;
            const sPartNode =
                typeof sPart === "number"
                    ? builder.createConstant(`${tag}_spn_${d}`, scalarInt64(sPart))
                    : sPart;
            let unpaddedNode: ConcreteValueNode;
            if (typeof kPart === "number") {
                unpaddedNode = builder.createOp("Add", [
                    sPartNode,
                    builder.createConstant(`${tag}_kb_${d}`, scalarInt64(kPart + biasVal)),
                ])[0];
            } else {
                const [tmp] = builder.createOp("Add", [sPartNode, kPart]);
                unpaddedNode = builder.createOp("Add", [
                    tmp,
                    builder.createConstant(`${tag}_bv_${d}`, scalarInt64(biasVal)),
                ])[0];
            }

            if (autoPad === "VALID") {
                pHeads.push(0);
                outSpatial.push(
                    explicitOutShape.length === spatialRank ? explicitOutShape[d] : unpaddedNode,
                );
            } else if (
                autoPad === "SAME_UPPER" ||
                autoPad === "SAME_LOWER" ||
                explicitOutShape.length === spatialRank
            ) {
                const outD =
                    explicitOutShape.length === spatialRank
                        ? explicitOutShape[d]
                        : (mulMixed(builder, inD, s, `${tag}_od_${d}`) as ConcreteValueNode);
                const outDNode =
                    typeof outD === "number"
                        ? builder.createConstant(`${tag}_odn_${d}`, scalarInt64(outD))
                        : outD;
                const [rawPad] = builder.createOp("Sub", [unpaddedNode, outDNode]);
                const [totalPad] = builder.createOp("Max", [rawPad, zeroI64]);
                const padNum =
                    autoPad === "SAME_UPPER"
                        ? totalPad
                        : builder.createOp("Add", [totalPad, oneI64])[0];
                const [head] = builder.createOp("Div", [padNum, twoI64]);
                pHeads.push(head);
                outSpatial.push(outD);
            } else {
                const pH =
                    explicitPads.length === spatialRank * 2 ? Number(explicitPads[d] ?? 0) : 0;
                const pT =
                    explicitPads.length === spatialRank * 2
                        ? Number(explicitPads[spatialRank + d] ?? 0)
                        : 0;
                pHeads.push(pH);
                const padSum = pH + pT;
                const outD =
                    padSum === 0
                        ? unpaddedNode
                        : builder.createOp("Sub", [
                              unpaddedNode,
                              builder.createConstant(`${tag}_ps_${d}`, scalarInt64(padSum)),
                          ])[0];
                outSpatial.push(outD);
            }
        }

        return { pHeads, outDims: [xDims[0], outC, ...outSpatial] };
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        const ins = op.getInputs()!;
        const rawXDims = toStaticShape(ins[0].shape);
        const rawWDims = toStaticShape(ins[1].shape);
        const group = getIntAttr(op, "group", 1);
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `ct_bounds_${op.id}`);

        if (!rawXDims.includes(-1) && !rawWDims.includes(-1)) {
            const { outDims } = this.resolvePadsAndOutDimsMixed(
                op,
                builder,
                rawXDims,
                rawWDims,
                "cts",
            );
            const staticOut = outDims as number[];
            const outTotal = staticOut.reduce((a, b) => a * b, 1);
            const cInPerGroup = rawXDims[1] / group;
            const kTotal = rawWDims.slice(2).reduce((a, b) => a * b, 1);
            return { totalIters: outTotal * cInPerGroup * kTotal, carryShape: staticOut };
        }

        const xDims = extractMixedDims(builder, ins[0], rawXDims, `ctbx_${op.id}`);
        const wDims = extractMixedDims(builder, ins[1], rawWDims, `ctbw_${op.id}`);
        const { outDims } = this.resolvePadsAndOutDimsMixed(
            op,
            builder,
            xDims,
            wDims,
            `ctb_${op.id}`,
        );
        const targetShape = packShapeVector(builder, outDims, `ctb_tsh_${op.id}`);
        const ax0 = builder.createConstant(`ctb_ax0_${op.id}`, int64Vec([0]));
        const [outTotal] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [outTotal, ax0]);

        const cInPerGroup =
            typeof xDims[1] === "number"
                ? xDims[1] / group
                : builder.createOp("Div", [
                      xDims[1],
                      builder.createConstant(`ctb_grp_${op.id}`, scalarInt64(group)),
                  ])[0];
        let kTotal: number | ConcreteValueNode = 1;
        for (let i = 2; i < wDims.length; i++) {
            kTotal = mulMixed(builder, kTotal, wDims[i], `ctb_k_${i}_${op.id}`);
        }
        const ckTotal = mulMixed(builder, cInPerGroup, kTotal, `ctb_ck_${op.id}`);
        const totalIters = mulMixed(
            builder,
            outTotal,
            ckTotal,
            `ctb_it_${op.id}`,
        ) as ConcreteValueNode;
        return { totalIters, carryShape, targetShape };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        const ins = op.getInputs()!;
        if (ins.length < 3 || ins[2].id === "") return loopOut;

        const rawXDims = toStaticShape(ins[0].shape);
        const rawWDims = toStaticShape(ins[1].shape);
        const dtype = ins[0].literalType as DataType;

        if (!rawXDims.includes(-1) && !rawWDims.includes(-1)) {
            const { outDims } = this.resolvePadsAndOutDimsMixed(
                op,
                builder,
                rawXDims,
                rawWDims,
                "ctp",
            );
            const staticOut = outDims as number[];
            const outShapeConst = builder.createConstant(
                `ct_outshape_${op.id}`,
                int64Vec(staticOut),
            );
            const [reshaped] = builder.createOp("Reshape", [loopOut, outShapeConst], {}, [
                { type: dtype, shape: staticOut as KnownShape },
            ]);

            const biasShape = [1, staticOut[1], ...new Array<number>(staticOut.length - 2).fill(1)];
            const biasShapeConst = builder.createConstant(
                `ct_bshape_${op.id}`,
                int64Vec(biasShape),
            );
            const [biasBcast] = builder.createOp("Reshape", [ins[2], biasShapeConst], {}, [
                { type: dtype, shape: biasShape as KnownShape },
            ]);

            const [added] = builder.createOp("Add", [reshaped, biasBcast], {}, [
                { type: dtype, shape: staticOut as KnownShape },
            ]);
            const flatShape = builder.createConstant(
                `ct_flat_${op.id}`,
                int64Vec([staticOut.reduce((a, b) => a * b, 1)]),
            );
            return builder.createOp("Reshape", [added, flatShape])[0];
        }

        const xDims = extractMixedDims(builder, ins[0], rawXDims, `ctpx_${op.id}`);
        const wDims = extractMixedDims(builder, ins[1], rawWDims, `ctpw_${op.id}`);
        const { outDims } = this.resolvePadsAndOutDimsMixed(
            op,
            builder,
            xDims,
            wDims,
            `ctp_${op.id}`,
        );
        const knownOutShape = outDims.map((d) => (typeof d === "number" ? d : -1)) as KnownShape;
        const outShapeVec = packShapeVector(builder, outDims, `ct_oshv_${op.id}`);
        const [reshaped] = builder.createOp("Reshape", [loopOut, outShapeVec], {}, [
            { type: dtype, shape: knownOutShape },
        ]);

        const biasDims: (number | ConcreteValueNode)[] = [
            1,
            outDims[1],
            ...new Array<number>(outDims.length - 2).fill(1),
        ];
        const biasShapeVec = packShapeVector(builder, biasDims, `ct_bshv_${op.id}`);
        const [biasBcast] = builder.createOp("Reshape", [ins[2], biasShapeVec]);
        const [added] = builder.createOp("Add", [reshaped, biasBcast], {}, [
            { type: dtype, shape: knownOutShape },
        ]);
        const flatShape = builder.createConstant(`ct_flat_${op.id}`, int64Vec([-1]));
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
        const rawXDims = toStaticShape(ins[0].shape);
        const rawWDims = toStaticShape(ins[1].shape);
        const dtype = ins[0].literalType as DataType;

        const spatialRank = rawXDims.length - 2;
        const group = getIntAttr(op, "group", 1);
        const strides = getIntsAttr(op, "strides", new Array<number>(spatialRank).fill(1));
        const dilations = getIntsAttr(op, "dilations", new Array<number>(spatialRank).fill(1));

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

        const xDims = extractMixedDims(builder, rawX, rawXDims, `ctx_${op.id}`);
        const wDims = extractMixedDims(builder, rawW, rawWDims, `ctw_${op.id}`);
        const { pHeads, outDims } = this.resolvePadsAndOutDimsMixed(
            op,
            builder,
            xDims,
            wDims,
            `cta_${op.id}`,
        );
        const cInPerGroup =
            typeof xDims[1] === "number"
                ? xDims[1] / group
                : builder.createOp("Div", [
                      xDims[1],
                      builder.createConstant(`ct_grp_${op.id}`, scalarInt64(group)),
                  ])[0];
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

        const xStrides = buildMixedStrides(builder, xDims);
        const wStrides = buildMixedStrides(builder, wDims);
        const outStrides = buildMixedStrides(builder, outDims);
        const flatX = ensureFlatInput(builder, rawX);
        const flatW = ensureFlatInput(builder, rawW);

        // Stride vectors over iterDims
        const outPtrStrides = [
            outStrides[0],
            mulMixed(builder, cOutPerGroup, outStrides[1], `ct_osg_${op.id}`),
            outStrides[1],
            ...outStrides.slice(2),
            0,
            ...new Array<number>(spatialRank).fill(0),
        ];
        const xBaseStrides = [
            xStrides[0],
            mulMixed(builder, cInPerGroup, xStrides[1], `ct_xsg_${op.id}`),
            0,
            ...new Array<number>(spatialRank).fill(0),
            xStrides[1],
            ...new Array<number>(spatialRank).fill(0),
        ];
        const wPtrStrides = [
            0,
            mulMixed(builder, cInPerGroup, wStrides[0], `ct_wsg_${op.id}`),
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
            const pVal = pHeads[d] ?? 0;
            const dilVal = dilations[d] ?? 1;

            // num = oCoord + pVal - kCoord * dilVal
            let num: ConcreteValueNode = oCoord;
            if (typeof pVal !== "number") {
                [num] = builder.createOp("Add", [num, pVal]);
            } else if (pVal !== 0) {
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

            const dimNode =
                typeof xDims[2 + d] === "number"
                    ? builder.createConstant(
                          `ct_dim_${d}_${op.id}`,
                          scalarInt64(xDims[2 + d] as number),
                      )
                    : (xDims[2 + d] as ConcreteValueNode);
            const [ltDim] = builder.createOp("Less", [inCoord, dimNode]);
            const [dimValid] = builder.createOp("And", [divValid, ltDim]);
            isValid = isValid ? builder.createOp("And", [isValid, dimValid])[0] : dimValid;

            const [safeCoord] = builder.createOp("Where", [dimValid, inCoord, zeroI64]);
            const str = xStrides[2 + d];
            const term =
                typeof str !== "number"
                    ? builder.createOp("Mul", [safeCoord, str])[0]
                    : str === 1
                      ? safeCoord
                      : builder.createOp("Mul", [
                            safeCoord,
                            builder.createConstant(`ct_xstr_${d}_${op.id}`, scalarInt64(str)),
                        ])[0];
            [xIdx] = builder.createOp("Add", [xIdx, term]);
        }

        const xVal = gatherScalar(builder, flatX, xIdx);
        const wVal = gatherScalar(builder, flatW, wIdx);

        const [rawProd] = builder.createOp("Mul", [xVal, wVal]);
        const [prod] = builder.createOp("Where", [isValid!, rawProd, zeroVal]);

        const [outUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);
        const curAcc = gatherScalar(builder, carryNode, flatOutIdx);

        const [newSum] = builder.createOp("Add", [curAcc, prod]);
        const [newSumUnsq] = builder.createOp("Unsqueeze", [newSum, axes]);
        const [nextCarry] = builder.createOp("ScatterElements", [carryNode, outUnsq, newSumUnsq], {
            axis: 0,
        });

        return { resultNode: newSum, nextCarry };
    }
}
