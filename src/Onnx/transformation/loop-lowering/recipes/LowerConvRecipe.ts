import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    asStaticDims,
    getIntsAttr,
    getStringAttr,
    getIntAttr,
    scalarInt64,
    int64Vec,
    UNKNOWN_SHAPE,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    buildAguPointer,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerConvRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerConv";
    public readonly targetOp = "Conv";
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Size",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Mul",
        "Div",
        "Mod",
        "Add",
        "Sub",
    ];

    match(op: OperationNode.Class): boolean {
        return op.type === "Conv";
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

        // 2. Dynamic Case
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `conv_bounds_${op.id}`);
        const axes0 = builder.createConstant(`axes0_${op.id}`, int64Vec([0]));

        const [shapeX] = builder.createOp("Shape", [inputs[0]]);
        const [shapeW] = builder.createOp("Shape", [inputs[1]]);

        const expectedCoS = [{ type: DataType.FLOAT, shape: UNKNOWN_SHAPE }];
        const [dummyX] = builder.createOp("ConstantOfShape", [shapeX], {}, expectedCoS);
        const [dummyW] = builder.createOp("ConstantOfShape", [shapeW], {}, expectedCoS);

        const dummyInputs = [dummyX, dummyW];
        if (inputs.length > 2) {
            const [shapeB] = builder.createOp("Shape", [inputs[2]]);
            const [dummyB] = builder.createOp("ConstantOfShape", [shapeB], {}, expectedCoS);
            dummyInputs.push(dummyB);
        }

        const expectedConv = [{ type: DataType.FLOAT, shape: UNKNOWN_SHAPE }];
        const [dummyOut] = builder.createOp("Conv", dummyInputs, op.attributes, expectedConv);

        const [targetShapeNode] = builder.createOp("Shape", [dummyOut]);
        const [totalIters] = builder.createOp("ReduceProd", [targetShapeNode, axes0], {
            keepdims: 0,
        });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0]);

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

        // 1. Resolve inputs (Captured Tensors inside body)
        const X = resolveRecipeInput(
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

        const xShape = asStaticDims(inputs[0].shape);
        const wShape = asStaticDims(inputs[1].shape);

        const C = xShape[1] !== -1 ? xShape[1] : 1;
        const M = wShape[0];

        // 2. Extract N_batch, H, and Win (statically or dynamically from captured X)
        let N_batch: number | ValueNode = xShape[0];
        let H: number | ValueNode = xShape[2];
        let Win: number | ValueNode = xShape[3];

        if (inputs[0].shape.includes(-1) || inputs[0].shape.length === 0) {
            const [shapeX] = builder.createOp("Shape", [X]);
            const [rankX] = builder.createOp("Size", [shapeX]);

            const getSpatialDim = (offsetFromEnd: number, tag: string) => {
                const offsetNode = builder.createConstant(
                    `${tag}_offset`,
                    scalarInt64(offsetFromEnd),
                );
                const [targetAxis] = builder.createOp("Add", [rankX, offsetNode]);
                const [dimRaw] = builder.createOp(
                    "Gather",
                    [shapeX, builder.createOp("Unsqueeze", [targetAxis, axes])[0]],
                    { axis: 0 },
                );
                return squeezeIfLen1(builder, dimRaw, axes, `${tag}_sq`);
            };

            if (inputs[0].shape[0] === -1 || typeof inputs[0].shape[0] === "string") {
                const zeroIdx = builder.createConstant(`batch_idx_0`, int64Vec([0]));
                const [bRaw] = builder.createOp("Gather", [shapeX, zeroIdx], { axis: 0 });
                N_batch = squeezeIfLen1(builder, bRaw, axes, `N_batch_sq`);
            }
            if (inputs[0].shape[2] === -1 || typeof inputs[0].shape[2] === "string") {
                H = getSpatialDim(-2, "H");
            }
            if (inputs[0].shape[3] === -1 || typeof inputs[0].shape[3] === "string") {
                Win = getSpatialDim(-1, "Win");
            }
        }

        const strides = getIntsAttr(op, "strides", [1, 1]);
        const dilations = getIntsAttr(op, "dilations", [1, 1]);
        const kernelShape = getIntsAttr(op, "kernel_shape", [wShape[2], wShape[3]]);
        let pads = getIntsAttr(op, "pads", []);
        const autoPad = getStringAttr(op, "auto_pad", "NOTSET");
        const group = getIntAttr(op, "group", 1);

        const kH = kernelShape[0],
            kW = kernelShape[1];
        const sH = strides[0],
            sW = strides[1];
        const dH = dilations[0],
            dW = dilations[1];
        const kEffH = dH * (kH - 1) + 1;
        const kEffW = dW * (kW - 1) + 1;

        // 3. Handle Padding logic
        if (pads.length === 0) {
            if (autoPad === "SAME_UPPER" || autoPad === "SAME_LOWER") {
                if (typeof H !== "number" || typeof Win !== "number") {
                    throw new Error(
                        "Dynamic spatial dimensions with auto_pad is not supported. Run CanonicalizePadPass first.",
                    );
                }
                const isLower = autoPad === "SAME_LOWER";
                const padH = Math.max((Math.ceil(H / sH) - 1) * sH + kEffH - H, 0);
                const padW = Math.max((Math.ceil(Win / sW) - 1) * sW + kEffW - Win, 0);
                const pT = isLower ? Math.ceil(padH / 2) : Math.floor(padH / 2);
                const pL = isLower ? Math.ceil(padW / 2) : Math.floor(padW / 2);
                pads = [pT, pL, padH - pT, padW - pL];
            } else {
                pads = [0, 0, 0, 0];
            }
        }

        const pT = pads[0] ?? 0,
            pL = pads[1] ?? 0,
            pB = pads[2] ?? 0,
            pR = pads[3] ?? 0;

        const buildOutDim = (
            dimIn: number | ValueNode,
            pA: number,
            pB_pad: number,
            kEff: number,
            stride: number,
            tag: string,
        ) => {
            if (typeof dimIn === "number") {
                return Math.floor((dimIn + pA + pB_pad - kEff) / stride + 1);
            }
            const numOffset = builder.createConstant(
                `${tag}_num_off`,
                scalarInt64(pA + pB_pad - kEff),
            );
            const [numAdd] = builder.createOp("Add", [dimIn, numOffset]);
            const strideConst = builder.createConstant(`${tag}_stride`, scalarInt64(stride));
            const [divOut] = builder.createOp("Div", [numAdd, strideConst]);
            const oneConst = builder.createConstant(`${tag}_one`, scalarInt64(1));
            return builder.createOp("Add", [divOut, oneConst])[0];
        };

        const H_out = buildOutDim(H, pT, pB, kEffH, sH, "H_out");
        const W_out = buildOutDim(Win, pL, pR, kEffW, sW, "W_out");

        let paddedX = X;
        if (pads.some((p) => Number(p) !== 0)) {
            const padsConst = builder.createConstant(`pads`, {
                dataType: DataType.INT64,
                dims: [8],
                int64Data: [0, 0, Number(pT), Number(pL), 0, 0, Number(pB), Number(pR)],
            });
            [paddedX] = builder.createOp("Pad", [X, padsConst]);
        }

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

        // 5. Compute Slice Ends
        const [yEnd] = builder.createOp("Add", [
            yStart,
            builder.createConstant(`keH`, scalarInt64(kEffH)),
        ]);
        const [xEnd] = builder.createOp("Add", [
            xStart,
            builder.createConstant(`keW`, scalarInt64(kEffW)),
        ]);
        const [nEnd] = builder.createOp("Add", [
            nIdx,
            builder.createConstant(`one`, scalarInt64(1)),
        ]);

        let starts: ValueNode, ends: ValueNode, sliceAxes: ValueNode, sliceSteps: ValueNode;

        if (group > 1) {
            const [cEnd] = builder.createOp("Add", [
                cStart!,
                builder.createConstant(`CperG_val`, scalarInt64(CperG)),
            ]);

            const coordsS = [nIdx, cStart!, yStart, xStart].map(
                (c) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [starts] = builder.createOp("Concat", coordsS, { axis: 0 });

            const coordsE = [nEnd, cEnd, yEnd, xEnd].map(
                (c) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [ends] = builder.createOp("Concat", coordsE, { axis: 0 });

            sliceAxes = builder.createConstant(`sliceAxes`, int64Vec([0, 1, 2, 3]));
            sliceSteps = builder.createConstant(`sliceSteps`, int64Vec([1, 1, dH, dW]));
        } else {
            const coordsS = [nIdx, yStart, xStart].map(
                (c) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [starts] = builder.createOp("Concat", coordsS, { axis: 0 });

            const coordsE = [nEnd, yEnd, xEnd].map(
                (c) => builder.createOp("Unsqueeze", [c, axes])[0],
            );
            [ends] = builder.createOp("Concat", coordsE, { axis: 0 });

            sliceAxes = builder.createConstant(`sliceAxes`, int64Vec([0, 2, 3]));
            sliceSteps = builder.createConstant(`sliceSteps`, int64Vec([1, dH, dW]));
        }

        // 6. Extract patch, gather kernel, and perform dot product
        const [patch] = builder.createOp("Slice", [paddedX, starts, ends, sliceAxes, sliceSteps]);

        const [mIdxUnsq] = builder.createOp("Unsqueeze", [mIdx, axes]);
        const [kernel] = builder.createOp("Gather", [W, mIdxUnsq], { axis: 0 });

        const [mulOut] = builder.createOp("Mul", [patch, kernel]);
        const [sumOut] = builder.createOp("ReduceSum", [mulOut], { keepdims: 0 });

        // 7. Final bias addition
        let finalScalar = sumOut;
        if (B_bias) {
            const [gBiasOut] = builder.createOp("Gather", [B_bias, mIdxUnsq], { axis: 0 });
            const biasSq = squeezeIfLen1(builder, gBiasOut, axes, `sqBias`);
            [finalScalar] = builder.createOp("Add", [sumOut, biasSq]);
        }

        return finalScalar;
    }
}
