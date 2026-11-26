import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asStaticDims,
    getIntsAttr,
    getStringAttr,
    getIntAttr,
    makeTensorProto,
    resolveAutoPads,
    scalarInt64,
    int64Vec,
    toStaticShape,
} from "../../../utils/index.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import { buildForLoopRegion } from "../LoopBuilder.js";
import {
    addMixed,
    buildAguPointer,
    buildLinearIndex,
    buildMixedStrides,
    captureMixedList,
    decodeMixedRadix,
    ensureFlatInput,
    ensureInnerValueNode,
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    mulMixed,
    packShapeVector,
    productMixed,
    resolvePoolShapeAndPadsMixed,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerConvRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerConv";
    public readonly targetOp = "Conv";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["conv", "2d", "fuseable"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Pad",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Mul",
        "Div",
        "Mod",
        "Add",
        "Sub",
        "Equal",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Conv") return false;
        const inputs = op.getInputs();
        if (!inputs || inputs.length < 2) return false;
        if (inputs[0].shape.length !== 4 || inputs[1].shape.length !== 4) return false;

        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
            const xDims = toStaticShape(inputs[0].shape);
            if (xDims[2] <= 0 || xDims[3] <= 0) return false;
        }
        return true;
    }

    getLoopBounds(
        op: OperationNode.Class,
        outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: number[] | ConcreteValueNode;
    } {
        const inputs = op.getInputs()!;
        const staticOut = asStaticDims(outShape);

        // 1. Static Case
        if (staticOut.length > 0 && !outShape.includes(-1) && !inputs[0].shape.includes(-1)) {
            const totalIters = staticOut.reduce((a, b) => a * b, 1);
            return { totalIters, carryShape: [totalIters] };
        }

        // 2. Dynamic Case: O(rank) INT64 pooling/conv output shape math (no dummy tensors)
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `conv_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        const xShape = toStaticShape(inputs[0].shape);
        const wShape = toStaticShape(inputs[1].shape);
        const dimsX = extractMixedDims(builder, inputs[0], xShape, `cb_x_${op.id}`);
        const dimsW = extractMixedDims(builder, inputs[1], wShape, `cb_w_${op.id}`);

        const strides = getIntsAttr(op, "strides", [1, 1]);
        const dilations = getIntsAttr(op, "dilations", [1, 1]);
        const kernelShape = getIntsAttr(op, "kernel_shape", [wShape[2], wShape[3]]);
        const explicitPads = getIntsAttr(op, "pads", []);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");

        const { outSpatial } = resolvePoolShapeAndPadsMixed(
            autoPad,
            [dimsX[2], dimsX[3]],
            kernelShape,
            strides,
            dilations,
            explicitPads,
            0,
            builder,
            `cb_sp_${op.id}`,
        );
        const targetShapeNode = packShapeVector(
            builder,
            [dimsX[0], dimsW[0], outSpatial[0], outSpatial[1]],
            `cb_tsh_${op.id}`,
        );
        const totalIters = productMixed(
            builder,
            [dimsX[0], dimsW[0], outSpatial[0], outSpatial[1]],
            `cb_tot_${op.id}`,
        );
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0], {}, [
            { type: DataType.INT64, shape: [1] },
        ]);

        return { totalIters, carryShape, targetShape: targetShapeNode };
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
        const builder = new GraphBuilder(body, `conv_${op.id}`);
        const inputs = op.getInputs()!;
        const canHoistX =
            loopCtx !== undefined &&
            !valueMap.has(inputs[0].id) &&
            loopCtx.outerBuilder.graph.hasNode(inputs[0].id);

        // 1. Resolve inputs (Captured Tensors inside body)
        const X = canHoistX
            ? inputs[0]
            : resolveRecipeInput(
                  builder,
                  inputs[0],
                  valueMap,
                  iter,
                  axes,
                  outShape,
                  false,
                  false,
                  targetShapeNode,
                  loopCtx,
              );
        const W = resolveRecipeInput(
            builder,
            inputs[1],
            valueMap,
            iter,
            axes,
            outShape,
            false,
            false,
            targetShapeNode,
            loopCtx,
        );
        const B_bias =
            inputs.length > 2
                ? resolveRecipeInput(
                      builder,
                      inputs[2],
                      valueMap,
                      iter,
                      axes,
                      outShape,
                      false,
                      false,
                      targetShapeNode,
                      loopCtx,
                  )
                : null;

        const xShape = toStaticShape(inputs[0].shape);
        const wShape = toStaticShape(inputs[1].shape);

        const C = xShape[1] !== -1 ? xShape[1] : 1;
        const M = wShape[0];

        // 2. Extract N_batch, H_out, W_out and pad X outside the loop when loopCtx is available
        const dimBuilder = canHoistX ? loopCtx!.outerBuilder : builder;
        const outerXRef = inputs[0];

        const strides = getIntsAttr(op, "strides", [1, 1]);
        const dilations = getIntsAttr(op, "dilations", [1, 1]);
        const kernelShape = getIntsAttr(op, "kernel_shape", [wShape[2], wShape[3]]);
        const explicitPads = getIntsAttr(op, "pads", []);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const group = getIntAttr(op, "group", 1);

        const kH = kernelShape[0],
            kW = kernelShape[1];
        const sH = strides[0] ?? 1,
            sW = strides[1] ?? 1;
        const dH = dilations[0] ?? 1,
            dW = dilations[1] ?? 1;
        const kEffH = dH * (kH - 1) + 1;
        const kEffW = dW * (kW - 1) + 1;

        const dimsX = extractMixedDims(dimBuilder, outerXRef, xShape, `conv_dx_${op.id}`);
        const dimsW = extractMixedDims(dimBuilder, inputs[1], wShape, `conv_dw_${op.id}`);
        const H = dimsX[2];
        const Win = dimsX[3];

        // 3. Handle Padding logic
        if (
            (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") &&
            (typeof H !== "number" || typeof Win !== "number")
        ) {
            throw new Error(
                "Dynamic spatial dimensions with auto_pad is not supported. Run CanonicalizePadPass first.",
            );
        }
        const pads = resolveAutoPads(
            autoPad,
            typeof H === "number" && typeof Win === "number" ? [H, Win] : [1, 1],
            [kH, kW],
            [sH, sW],
            [dH, dW],
            explicitPads,
        );

        const pT = pads[0] ?? 0,
            pL = pads[1] ?? 0,
            pB = pads[2] ?? 0,
            pR = pads[3] ?? 0;

        let paddedXSource = canHoistX ? outerXRef : X;
        if (pads.some((p) => Number(p) !== 0)) {
            const padBuilder = canHoistX ? loopCtx!.outerBuilder : builder;
            const padsConst = padBuilder.createConstant(`pads_${op.id}`, {
                dataType: DataType.INT64,
                dims: [8],
                int64Data: [0, 0, Number(pT), Number(pL), 0, 0, Number(pB), Number(pR)],
            });
            paddedXSource = padBuilder.createOp("Pad", [paddedXSource, padsConst])[0];
        }
        const paddedX = canHoistX ? ensureInnerValueNode(builder, paddedXSource) : paddedXSource;

        const paddedDimsX = [
            dimsX[0],
            dimsX[1],
            addMixed(dimBuilder, H, Number(pT) + Number(pB), `conv_ph_${op.id}`),
            addMixed(dimBuilder, Win, Number(pL) + Number(pR), `conv_pw_${op.id}`),
        ];
        const xStridesRaw = buildMixedStrides(dimBuilder, paddedDimsX);
        const wStridesRaw = buildMixedStrides(dimBuilder, dimsW);

        const buildOutDim = (
            dimIn: number | ConcreteValueNode,
            pA: number,
            pB_pad: number,
            kEff: number,
            stride: number,
            tag: string,
        ): number | ConcreteValueNode => {
            if (typeof dimIn === "number") {
                return Math.floor((dimIn + pA + pB_pad - kEff) / stride + 1);
            }
            const numOffset = dimBuilder.createConstant(
                `${tag}_num_off`,
                scalarInt64(pA + pB_pad - kEff),
            );
            const [numAdd] = dimBuilder.createOp("Add", [dimIn, numOffset]);
            const strideConst = dimBuilder.createConstant(`${tag}_stride`, scalarInt64(stride));
            const [divOut] = dimBuilder.createOp("Div", [numAdd, strideConst]);
            const oneConst = dimBuilder.createConstant(`${tag}_one`, scalarInt64(1));
            return dimBuilder.createOp("Add", [divOut, oneConst])[0];
        };

        const rawH_out = buildOutDim(H, pT, pB, kEffH, sH, `H_out_${op.id}`);
        const rawW_out = buildOutDim(Win, pL, pR, kEffW, sW, `W_out_${op.id}`);
        const [N_batch, H_out, W_out, CperG_cap, xStrN, xStrC, xStrH, wStrM, wStrC] =
            captureMixedList(builder, [
                dimsX[0],
                rawH_out,
                rawW_out,
                dimsW[1],
                xStridesRaw[0],
                xStridesRaw[1],
                xStridesRaw[2],
                wStridesRaw[0],
                wStridesRaw[1],
            ]);

        const canHoistW =
            loopCtx !== undefined &&
            !valueMap.has(inputs[1].id) &&
            loopCtx.outerBuilder.graph.hasNode(inputs[1].id);
        const flatX = canHoistX
            ? ensureInnerValueNode(builder, ensureFlatInput(loopCtx!.outerBuilder, paddedXSource))
            : ensureFlatInput(builder, paddedX);
        const flatW = canHoistW
            ? ensureInnerValueNode(builder, ensureFlatInput(loopCtx!.outerBuilder, inputs[1]))
            : ensureFlatInput(builder, W);

        // 4. Coordinate & Window Bounds Generation: Pointer Mode (AGU) vs Radix Mode (Div/Mod)
        let nIdx: ConcreteValueNode;
        let mIdx: ConcreteValueNode;
        let yStart: ConcreteValueNode;
        let xStart: ConcreteValueNode;
        let cStart: ConcreteValueNode | undefined;

        const MperG = Math.floor(M / group);
        const CperG = wShape[1] !== -1 ? wShape[1] : Math.floor(C / group);

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const iterDims: (number | ValueNode)[] = [N_batch, M, H_out, W_out];
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                `conv_space_${op.id}`,
            );

            nIdx = counters[0];
            mIdx = counters[1];

            // Directly step yStart by sH and xStart by sW using AGU pointers
            yStart = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                [0, 0, sH, 0],
                `conv_yStart_${op.id}`,
            );
            xStart = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                iterDims,
                [0, 0, 0, sW],
                `conv_xStart_${op.id}`,
            );

            if (group > 1) {
                cStart = buildAguPointer(
                    builder,
                    loopCtx,
                    valueMap,
                    [N_batch, group, MperG, H_out, W_out],
                    [0, CperG, 0, 0, 0],
                    `conv_cStart_${op.id}`,
                );
            }
        } else {
            const mConst = builder.createConstant(`M_const`, scalarInt64(M));
            const hOutNode =
                typeof H_out === "number"
                    ? builder.createConstant(`H_out_const`, scalarInt64(H_out))
                    : H_out;
            const wOutNode =
                typeof W_out === "number"
                    ? builder.createConstant(`W_out_const`, scalarInt64(W_out))
                    : W_out;

            const [hwOutNode] = builder.createOp("Mul", [hOutNode, wOutNode]);
            const [mHwOutNode] = builder.createOp("Mul", [mConst, hwOutNode]);

            [nIdx] = builder.createOp("Div", [iter, mHwOutNode]);
            const [remN] = builder.createOp("Mod", [iter, mHwOutNode]);

            [mIdx] = builder.createOp("Div", [remN, hwOutNode]);
            const [rem] = builder.createOp("Mod", [remN, hwOutNode]);

            const [yIdx] = builder.createOp("Div", [rem, wOutNode]);
            const [xIdx] = builder.createOp("Mod", [rem, wOutNode]);

            [yStart] = builder.createOp("Mul", [
                yIdx,
                builder.createConstant(`sH`, scalarInt64(sH)),
            ]);
            [xStart] = builder.createOp("Mul", [
                xIdx,
                builder.createConstant(`sW`, scalarInt64(sW)),
            ]);

            if (group > 1) {
                const [gIdx] = builder.createOp("Div", [
                    mIdx,
                    builder.createConstant(`MperG`, scalarInt64(MperG)),
                ]);
                [cStart] = builder.createOp("Mul", [
                    gIdx,
                    builder.createConstant(`CperG`, scalarInt64(CperG)),
                ]);
            }
        }

        // 5. Hoist per-output base offsets & execute scalar reduction inner Loop over [CperG, kH, kW]
        const dtype = inputs[0].literalType || DataType.FLOAT;
        const nOffset = mulMixed(builder, nIdx, xStrN, `conv_noff_${op.id}`);
        const cOffset = cStart ? mulMixed(builder, cStart, xStrC, `conv_coff_${op.id}`) : 0;
        const ncBaseOuter = addMixed(builder, nOffset, cOffset, `conv_ncb_${op.id}`);
        const wMBaseOuter = mulMixed(builder, mIdx, wStrM, `conv_wmb_${op.id}`);

        const kInnerTotal = mulMixed(builder, CperG_cap, kH * kW, `conv_ktot_${op.id}`);
        const innerMode = loopCtx?.indexingMode ?? IndexingMode.Radix;
        const kLoop = buildForLoopRegion(
            builder,
            kInnerTotal,
            dtype,
            [1],
            `conv_k_${op.id}`,
            `conv_k_${op.id}`,
            innerMode,
            false,
        );
        const kb = kLoop.innerBuilder;
        const zeroOuterScalar = builder.createConstant(
            `conv_zacc_${op.id}`,
            makeTensorProto(dtype, [], [0]),
        );
        const accIn = kLoop.loopCtx.addCarry(`conv_acc_${op.id}`, zeroOuterScalar, dtype, []);
        const kbFlatX = ensureInnerValueNode(kb, flatX);
        const kbFlatW = ensureInnerValueNode(kb, flatW);

        const [kbNCBase, kbWMBase, kbYWinStart, kbXWinStart, kbCperG, kbXStrC, kbXStrH, kbWStrC] =
            captureMixedList(kb, [
                ncBaseOuter,
                wMBaseOuter,
                yStart,
                xStart,
                CperG_cap,
                xStrC,
                xStrH,
                wStrC,
            ]);

        const innerIterDims: (number | ConcreteValueNode)[] = [kbCperG, kH, kW];
        const innerValMap = new Map<string, ValueNode>();
        let xChanOff: number | ConcreteValueNode;
        let wOff: number | ConcreteValueNode;
        let yDilOff: number | ConcreteValueNode;
        let xDilOff: number | ConcreteValueNode;

        if (innerMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                kb,
                kLoop.loopCtx,
                innerValMap,
                innerIterDims,
                `conv_kc_${op.id}`,
            );
            xChanOff = buildAguPointer(
                kb,
                kLoop.loopCtx,
                innerValMap,
                innerIterDims,
                [kbXStrC, 0, 0],
                `conv_kxc_${op.id}`,
            );
            wOff = buildAguPointer(
                kb,
                kLoop.loopCtx,
                innerValMap,
                innerIterDims,
                [kbWStrC, kW, 1],
                `conv_kwo_${op.id}`,
            );
            yDilOff =
                dH === 1
                    ? counters[1]
                    : buildAguPointer(
                          kb,
                          kLoop.loopCtx,
                          innerValMap,
                          innerIterDims,
                          [0, dH, 0],
                          `conv_kyd_${op.id}`,
                      );
            xDilOff =
                dW === 1
                    ? counters[2]
                    : buildAguPointer(
                          kb,
                          kLoop.loopCtx,
                          innerValMap,
                          innerIterDims,
                          [0, 0, dW],
                          `conv_kxd_${op.id}`,
                      );
        } else {
            const kCoords = decodeMixedRadix(kb, kLoop.trip, innerIterDims, `conv_kdec_${op.id}`);
            xChanOff = mulMixed(kb, kCoords[0], kbXStrC, `conv_kxc_${op.id}`);
            wOff = buildLinearIndex(kb, kCoords, [kbWStrC, kW, 1], `conv_kwo_${op.id}`);
            yDilOff = mulMixed(kb, kCoords[1], dH, `conv_kyd_${op.id}`);
            xDilOff = mulMixed(kb, kCoords[2], dW, `conv_kxd_${op.id}`);
        }

        const toI64Node = (v: number | ConcreteValueNode, t: string): ConcreteValueNode =>
            typeof v === "number" ? kb.createConstant(t, scalarInt64(v)) : v;

        const yIn = addMixed(kb, kbYWinStart, yDilOff, `conv_yin_${op.id}`);
        const xIn = addMixed(kb, kbXWinStart, xDilOff, `conv_xin_${op.id}`);

        const yXOff = mulMixed(kb, yIn, kbXStrH, `conv_yxo_${op.id}`);
        const basePlusC = addMixed(kb, kbNCBase, xChanOff, `conv_bpc_${op.id}`);
        const basePlusCY = addMixed(kb, basePlusC, yXOff, `conv_bpcy_${op.id}`);
        const xIdx = toI64Node(
            addMixed(kb, basePlusCY, xIn, `conv_xidx_${op.id}`),
            `conv_xidxn_${op.id}`,
        );
        const wIdx = toI64Node(
            addMixed(kb, kbWMBase, wOff, `conv_widx_${op.id}`),
            `conv_widxn_${op.id}`,
        );

        const xVal = gatherScalar(kb, kbFlatX, xIdx);
        const wVal = gatherScalar(kb, kbFlatW, wIdx);
        const [prod] = kb.createOp("Mul", [xVal, wVal]);
        const [accNext] = kb.createOp("Add", [accIn, prod]);
        kLoop.loopCtx.setNextState(accIn, accNext);
        kLoop.finalize([]);
        const sumOut = kLoop.loopOutputs[0];

        // 7. Final bias addition
        let finalScalar: ValueNode = sumOut;
        if (B_bias) {
            const biasSq = gatherScalar(builder, B_bias, mIdx);
            [finalScalar] = builder.createOp("Add", [sumOut, biasSq]);
        }

        return finalScalar;
    }
}
