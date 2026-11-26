import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
    getIntAttr,
    int64Vec,
    makeTensorProto,
    normalizeAxis,
    readScalarFromTensorNode,
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
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerCumSumRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerCumSum";
    public readonly targetOp = "CumSum";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly priority = 50;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["scan", "self-carry"] as const;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "ReduceProd",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Sub",
        "Equal",
        "And",
        "Where",
        "Mul",
        "Div",
        "Mod",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "CumSum") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        const dims = toStaticShape(ins[0].shape);
        const axisVal = readScalarFromTensorNode(ins[1]);
        return dims.length > 0 && axisVal !== undefined;
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: number | ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        const X = op.getInputs()![0];
        const dims = toStaticShape(X.shape);
        if (!dims.includes(-1)) {
            const total = dims.reduce((a, b) => a * b, 1);
            return { totalIters: total, carryShape: dims };
        }
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `cs_bounds_${op.id}`);
        const ax0 = builder.createConstant(`csb_ax0_${op.id}`, int64Vec([0]));
        const [targetShape] = builder.createOp("Shape", [X], {}, [
            { type: DataType.INT64, shape: [dims.length] },
        ]);
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, ax0]);
        return { totalIters, carryShape, targetShape };
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
        const builder = new GraphBuilder(body, `cumsum_${op.id}`);
        const ins = op.getInputs()!;
        const X = ins[0];
        const rawDims = toStaticShape(X.shape);
        const dtype = X.literalType as DataType;

        const axis = normalizeAxis(Math.trunc(readScalarFromTensorNode(ins[1])!), rawDims.length);
        const exclusive = getIntAttr(op, "exclusive", 0) !== 0;
        const reverse = getIntAttr(op, "reverse", 0) !== 0;

        const rawInnerX = resolveRecipeInput(
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
        const dims = extractMixedDims(builder, rawInnerX, rawDims, `cs_${op.id}`);
        const strides = buildMixedStrides(builder, dims);
        const axisStride = strides[axis];
        const axisDim = dims[axis];
        const flatX = ensureFlatInput(builder, rawInnerX);

        const zeroI64 = builder.createConstant(`cs_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`cs_one_${op.id}`, scalarInt64(1));

        let flatOutIdx: ConcreteValueNode;
        let axisCoord: ConcreteValueNode;

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                dims,
                `cs_cnt_${op.id}`,
            );
            axisCoord = counters[axis];

            const ptrStrides: (number | ConcreteValueNode)[] = [...strides];
            let initialOffset: number | ConcreteValueNode = 0;
            if (reverse) {
                ptrStrides[axis] =
                    typeof axisStride === "number"
                        ? -axisStride
                        : builder.createOp("Sub", [zeroI64, axisStride])[0];
                if (typeof axisDim === "number" && typeof axisStride === "number") {
                    initialOffset = (axisDim - 1) * axisStride;
                } else {
                    const outerB = loopCtx.outerBuilder;
                    const oDims = extractMixedDims(outerB, X, rawDims, `cso_${op.id}`);
                    const oStrides = buildMixedStrides(outerB, oDims);
                    const oAxisDim =
                        typeof oDims[axis] === "number"
                            ? outerB.createConstant(
                                  `cso_adim_${op.id}`,
                                  scalarInt64(oDims[axis] as number),
                              )
                            : (oDims[axis] as ConcreteValueNode);
                    const oAxisStride =
                        typeof oStrides[axis] === "number"
                            ? outerB.createConstant(
                                  `cso_astr_${op.id}`,
                                  scalarInt64(oStrides[axis] as number),
                              )
                            : (oStrides[axis] as ConcreteValueNode);
                    const oOne = outerB.createConstant(`cso_one_${op.id}`, scalarInt64(1));
                    const [dimMinus1] = outerB.createOp("Sub", [oAxisDim, oOne]);
                    [initialOffset] = outerB.createOp("Mul", [dimMinus1, oAxisStride]);
                }
            }
            flatOutIdx = buildAguPointer(
                builder,
                loopCtx,
                valueMap,
                dims,
                ptrStrides,
                `cs_ptr_${op.id}`,
                initialOffset,
            );
        } else {
            const coords = decodeMixedRadix(builder, iter, dims, `cs_dec_${op.id}`);
            axisCoord = coords[axis];
            const mappedCoords = [...coords];
            if (reverse) {
                const lastIdx =
                    typeof axisDim === "number"
                        ? builder.createConstant(`cs_last_${op.id}`, scalarInt64(axisDim - 1))
                        : builder.createOp("Sub", [axisDim, oneI64])[0];
                [mappedCoords[axis]] = builder.createOp("Sub", [lastIdx, axisCoord]);
            }
            flatOutIdx = buildLinearIndex(builder, mappedCoords, strides, `cs_lin_${op.id}`);
        }

        const strideNode =
            typeof axisStride === "number"
                ? builder.createConstant(`cs_str_${op.id}`, scalarInt64(axisStride))
                : axisStride;
        const zeroVal = builder.createConstant(`cs_zval_${op.id}`, makeTensorProto(dtype, [], [0]));

        const [isFirst] = builder.createOp("Equal", [axisCoord, zeroI64]);

        // Previous element index along the cumsum axis (clamped to 0 when isFirst)
        const [rawPrevIdx] = reverse
            ? builder.createOp("Add", [flatOutIdx, strideNode])
            : builder.createOp("Sub", [flatOutIdx, strideNode]);
        const [safePrevIdx] = builder.createOp("Where", [isFirst, zeroI64, rawPrevIdx]);

        const prevCarry = gatherScalar(builder, carryNode, safePrevIdx);

        const [outIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        let outVal: ConcreteValueNode;
        if (exclusive) {
            // exclusive: out[0] = 0; out[i] = out[i-1] + x[i-1]
            const xPrev = gatherScalar(builder, flatX, safePrevIdx);
            const [sum] = builder.createOp("Add", [prevCarry, xPrev]);
            [outVal] = builder.createOp("Where", [isFirst, zeroVal, sum]);
        } else {
            // inclusive: out[0] = x[0]; out[i] = out[i-1] + x[i]
            const xCur = gatherScalar(builder, flatX, flatOutIdx);
            const [prevEff] = builder.createOp("Where", [isFirst, zeroVal, prevCarry]);
            [outVal] = builder.createOp("Add", [prevEff, xCur]);
        }

        const [outValUnsq] = builder.createOp("Unsqueeze", [outVal, axes]);
        const [nextCarry] = builder.createOp(
            "ScatterElements",
            [carryNode, outIdxUnsq, outValUnsq],
            { axis: 0 },
        );

        return { resultNode: outVal, nextCarry };
    }
}
