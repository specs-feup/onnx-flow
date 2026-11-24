import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import type { ValueNode, ConcreteValueNode, KnownShape } from "../../../OnnxTypes.js";
import { DataType } from "../../../OnnxTypes.js";
import {
    getIntAttr,
    getIntsAttr,
    int64Vec,
    normalizeAxis,
    readConstIntegerVectorFromTensorNode,
    scalarInt64,
} from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import { OpRegistry } from "../../../Schema/OpRegistry.js";
import { OpCategory } from "../../../Schema/OpSchema.js";
import {
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import ConstantNode from "../../../ConstantNode.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerDynamicReductionRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerDynamicReduction";
    public readonly targetOp = OpCategory.Reduction;
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["reduction", "dynamic", "self-carry"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Gather",
        "ScatterElements",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Mul",
        "Div",
        "Mod",
        "Max",
        "Min",
        "Abs",
        "Exp",
        "Sqrt",
        "Log",
        "Equal",
        "And",
        "Where",
        "Cast",
        "ReduceProd",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type === "ArgMax" || op.type === "ArgMin" || op.type === "TopK") return false;
        const schema = OpRegistry.getInstance().get(op.type, 19);
        if (schema?.category !== OpCategory.Reduction) return false;
        const ins = op.getInputs();
        if (!ins || ins.length === 0 || !Array.isArray(ins[0].shape) || ins[0].shape.length === 0) {
            return false;
        }
        if (ins.length > 1 && ins[1].id !== "" && !ins[1].is(ConstantNode)) {
            return false;
        }
        return true;
    }

    private resolveAxes(op: OperationNode.Class, rank: number): number[] {
        const ins = op.getInputs()!;
        let axes: number[] = [];
        if (ins.length > 1 && ins[1].is(ConstantNode)) {
            axes = readConstIntegerVectorFromTensorNode(ins[1].as(ConstantNode)) ?? [];
        } else {
            axes = getIntsAttr(op, "axes", []);
        }
        if (axes.length === 0) {
            return Array.from({ length: rank }, (_, i) => i);
        }
        return axes.map((a) => normalizeAxis(a, rank));
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: ConcreteValueNode;
        carryShape: number[] | ConcreteValueNode;
        targetShape?: ConcreteValueNode;
    } {
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;
        const axesList = this.resolveAxes(op, rank);
        const keepdims = getIntAttr(op, "keepdims", 1);

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `dred_bounds_${op.id}`);
        const ax0 = builder.createConstant(`dredb_ax0_${op.id}`, int64Vec([0]));
        const [shapeNode] = builder.createOp("Shape", [ins[0]], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const [totalIters] = builder.createOp("ReduceProd", [shapeNode, ax0], { keepdims: 0 });

        if (keepdims === 1) {
            const axesC = builder.createConstant(`dredb_axes_${op.id}`, int64Vec(axesList));
            const onesC = builder.createConstant(
                `dredb_ones_${op.id}`,
                int64Vec(new Array(axesList.length).fill(1)),
            );
            const [targetShape] = builder.createOp(
                "ScatterElements",
                [shapeNode, axesC, onesC],
                { axis: 0 },
                [{ type: DataType.INT64, shape: [rank] }],
            );
            const [carryLen] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
            const [carryShape] = builder.createOp("Unsqueeze", [carryLen, ax0]);
            return { totalIters, carryShape, targetShape };
        }

        const keptAxes = Array.from({ length: rank }, (_, i) => i).filter(
            (i) => !axesList.includes(i),
        );
        if (keptAxes.length === 0) {
            // Full reduction to 0D scalar -> 1-element carry
            return { totalIters, carryShape: [1] };
        }

        const keptC = builder.createConstant(`dredb_kept_${op.id}`, int64Vec(keptAxes));
        const [targetShape] = builder.createOp("Gather", [shapeNode, keptC], { axis: 0 }, [
            { type: DataType.INT64, shape: [keptAxes.length] },
        ]);
        const [carryLen] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [carryLen, ax0]);
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
        const builder = new GraphBuilder(body, `dred_${op.id}`);
        const ins = op.getInputs()! as ConcreteValueNode[];
        const rank = ins[0].shape.length;
        const axesList = this.resolveAxes(op, rank);

        const rawIn = resolveRecipeInput(
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
        const [shapeIn] = builder.createOp("Shape", [rawIn], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const flatInput = ensureFlatInput(builder, rawIn);

        const inDims: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            const dIdx = builder.createConstant(`dredi_d_${op.id}_${d}`, int64Vec([d]));
            const [rawD] = builder.createOp("Gather", [shapeIn, dIdx], { axis: 0 });
            inDims.push(
                squeezeIfLen1(builder, rawD, axes, `dredi_sqd_${op.id}_${d}`) as ConcreteValueNode,
            );
        }

        let readIdx: ConcreteValueNode = iter;
        let inCoords: ConcreteValueNode[];

        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            inCoords = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                inDims,
                `dredi_cnt_${op.id}`,
            ).counters;

            const initRead = loopCtx.outerBuilder.createConstant(
                `dredi_init_r_${op.id}`,
                scalarInt64(0),
            );
            const rIn = loopCtx.addCarry(`dredi_rptr_${op.id}`, initRead, DataType.INT64, []);
            const oneI64 = builder.createConstant(`dredi_one_${op.id}`, scalarInt64(1));
            const [nextR] = builder.createOp("Add", [rIn, oneI64]);
            loopCtx.setNextState(rIn, nextR);
            readIdx = rIn;
        } else {
            inCoords = decodeMixedRadix(builder, iter, inDims, `dredi_dec_${op.id}`);
        }

        const [readIdxUnsq] = builder.createOp("Unsqueeze", [readIdx, axes]);
        const [gatheredOut] = builder.createOp("Gather", [flatInput, readIdxUnsq], { axis: 0 });
        let val = squeezeIfLen1(
            builder,
            gatheredOut,
            axes,
            `dredi_sq_in_${op.id}`,
        ) as ConcreteValueNode;

        // Pre-processing for specialized reductions
        if (op.type === "ReduceLogSumExp") [val] = builder.createOp("Exp", [val]);
        else if (op.type === "ReduceSumSquare" || op.type === "ReduceL2")
            [val] = builder.createOp("Mul", [val, val]);
        else if (op.type === "ReduceL1") [val] = builder.createOp("Abs", [val]);

        const zeroConst = builder.createConstant(`dredi_z_${op.id}`, scalarInt64(0));
        const accDims: (number | ValueNode)[] = inDims.map((dim, i) =>
            axesList.includes(i) ? 1 : dim,
        );
        const accStrides: (number | ValueNode)[] = new Array(rank).fill(1);
        let curStride: number | ValueNode = 1;
        for (let d = rank - 1; d >= 0; d--) {
            accStrides[d] = curStride;
            if (d > 0) {
                const dimD = accDims[d];
                if (typeof curStride === "number" && typeof dimD === "number") {
                    curStride = curStride * dimD;
                } else if (typeof curStride === "number" && curStride === 1) {
                    curStride = dimD;
                } else if (typeof dimD === "number" && dimD === 1) {
                    // keep curStride unchanged
                } else {
                    const cNode =
                        typeof curStride === "number"
                            ? builder.createConstant(
                                  `dredi_cs_${op.id}_${d}`,
                                  scalarInt64(curStride),
                              )
                            : curStride;
                    const dNode =
                        typeof dimD === "number"
                            ? builder.createConstant(`dredi_cd_${op.id}_${d}`, scalarInt64(dimD))
                            : dimD;
                    [curStride] = builder.createOp("Mul", [cNode, dNode]);
                }
            }
        }

        const outCoords = inCoords.map((c, i) => (axesList.includes(i) ? zeroConst : c));
        const flatOutIdx = buildLinearIndex(builder, outCoords, accStrides, `dredi_out_${op.id}`);
        const [flatOutIdxUnsq] = builder.createOp("Unsqueeze", [flatOutIdx, axes]);

        const [gAccOut] = builder.createOp("Gather", [carryNode, flatOutIdxUnsq], { axis: 0 });
        const currentAcc = squeezeIfLen1(
            builder,
            gAccOut,
            axes,
            `dredi_sq_acc_${op.id}`,
        ) as ConcreteValueNode;

        let mathOpType = "Add";
        if (op.type === "ReduceMax") mathOpType = "Max";
        else if (op.type === "ReduceMin") mathOpType = "Min";
        else if (op.type === "ReduceProd") mathOpType = "Mul";
        const [mathOut] = builder.createOp(mathOpType, [currentAcc, val]);

        let isFirstHit: ConcreteValueNode | undefined;
        for (const ax of axesList) {
            const [isZero] = builder.createOp("Equal", [inCoords[ax], zeroConst]);
            isFirstHit = isFirstHit ? builder.createOp("And", [isFirstHit, isZero])[0] : isZero;
        }

        const [finalValue] = builder.createOp("Where", [isFirstHit!, val, mathOut]);
        const [updateVal] = builder.createOp("Unsqueeze", [finalValue, axes]);
        const [nextCarry] = builder.createOp(
            "ScatterElements",
            [carryNode, flatOutIdxUnsq, updateVal],
            { axis: 0 },
        );

        return { resultNode: finalValue, nextCarry };
    }

    postProcess(op: OperationNode.Class, builder: GraphBuilder, loopOut: ValueNode): ValueNode {
        if (op.type === "ReduceL2") return builder.createOp("Sqrt", [loopOut])[0];
        if (op.type === "ReduceLogSum" || op.type === "ReduceLogSumExp") {
            return builder.createOp("Log", [loopOut])[0];
        }
        if (op.type === "ReduceMean") {
            const ins = op.getInputs()!;
            const rank = ins[0].shape.length;
            const axesList = this.resolveAxes(op, rank);

            const [shapeX] = builder.createOp("Shape", [ins[0]], {}, [
                { type: DataType.INT64, shape: [rank] },
            ]);
            const axesC = builder.createConstant(`dredp_axes_${op.id}`, int64Vec(axesList));
            const [redDims] = builder.createOp("Gather", [shapeX, axesC], { axis: 0 }, [
                { type: DataType.INT64, shape: [axesList.length] },
            ]);
            const ax0 = builder.createConstant(`dredp_ax0_${op.id}`, int64Vec([0]));
            const [countRaw] = builder.createOp("ReduceProd", [redDims, ax0], { keepdims: 0 });
            const [countCast] = builder.createOp("Cast", [countRaw], {
                to: loopOut.literalType || DataType.FLOAT,
            });
            return builder.createOp("Div", [loopOut, countCast])[0];
        }
        return loopOut;
    }
}
