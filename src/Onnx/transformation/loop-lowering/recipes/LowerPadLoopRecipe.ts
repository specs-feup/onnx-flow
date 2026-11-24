import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import {
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
    buildLinearIndex,
    decodeMixedRadix,
    ensureFlatInput,
    ensureInnerValueNode,
    getOrCreateIterationSpaceCounters,
    resolveRecipeInput,
    squeezeIfLen1,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

const isPresent = (n?: ValueNode): n is ValueNode => n !== undefined && n.id !== "";

export class LowerPadLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerPadLoop";
    public readonly targetOp = "Pad";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["padding", "wrap-reflect", "dynamic-pads", "fuseable"] as const;
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
        "Sub",
        "Mul",
        "Div",
        "Mod",
        "Min",
        "Max",
        "Less",
        "Not",
        "Equal",
        "And",
        "Where",
        "Cast",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Pad") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        if (!Array.isArray(ins[0].shape) || ins[0].shape.length === 0) return false;
        const mode = getStringAttr(op, "mode", "constant").toLowerCase();
        if (mode !== "constant" && mode !== "edge" && mode !== "reflect" && mode !== "wrap") {
            return false;
        }
        const k = isPresent(ins[3]) ? toStaticShape(ins[3].shape)[0] : ins[0].shape.length;
        return k > 0;
    }

    private resolveFullPads(
        builder: GraphBuilder,
        padsIn: ValueNode,
        axesIn: ValueNode | undefined,
        rank: number,
        k: number,
        tag: string,
    ): { fullBeg: ConcreteValueNode; fullEnd: ConcreteValueNode } {
        const toI64 = (v: ValueNode): ConcreteValueNode =>
            v.literalType === DataType.INT64
                ? (v as ConcreteValueNode)
                : builder.createOp("Cast", [v], { to: DataType.INT64 })[0];

        const pads64 = toI64(padsIn);
        const begIdx = builder.createConstant(
            `${tag}_bidx`,
            int64Vec(Array.from({ length: k }, (_, i) => i)),
        );
        const endIdx = builder.createConstant(
            `${tag}_eidx`,
            int64Vec(Array.from({ length: k }, (_, i) => k + i)),
        );
        const [begK] = builder.createOp("Gather", [pads64, begIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [k] },
        ]);
        const [endK] = builder.createOp("Gather", [pads64, endIdx], { axis: 0 }, [
            { type: DataType.INT64, shape: [k] },
        ]);

        if (axesIn === undefined) {
            return { fullBeg: begK, fullEnd: endK };
        }

        const axes64 = toI64(axesIn);
        const zeroSc = builder.createConstant(`${tag}_z`, scalarInt64(0));
        const rankSc = builder.createConstant(`${tag}_r`, scalarInt64(rank));
        const [isNegAx] = builder.createOp("Less", [axes64, zeroSc]);
        const [normAxes] = builder.createOp("Where", [
            isNegAx,
            builder.createOp("Add", [axes64, rankSc])[0],
            axes64,
        ]);

        const zeroVec = builder.createConstant(`${tag}_zvec`, int64Vec(new Array(rank).fill(0)));
        const [fullBeg] = builder.createOp("ScatterElements", [zeroVec, normAxes, begK], {
            axis: 0,
        });
        const [fullEnd] = builder.createOp("ScatterElements", [zeroVec, normAxes, endK], {
            axis: 0,
        });
        return { fullBeg, fullEnd };
    }

    getLoopBounds(
        op: OperationNode.Class,
        _outShape: KnownShape,
    ): {
        totalIters: ConcreteValueNode;
        carryShape: ConcreteValueNode;
        targetShape: ConcreteValueNode;
    } {
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;
        const k = isPresent(ins[3]) ? toStaticShape(ins[3].shape)[0] : rank;
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `pad_bounds_${op.id}`);

        const [shapeData] = builder.createOp("Shape", [ins[0]], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const { fullBeg, fullEnd } = this.resolveFullPads(
            builder,
            ins[1],
            isPresent(ins[3]) ? ins[3] : undefined,
            rank,
            k,
            `padb_${op.id}`,
        );

        const [sumBeg] = builder.createOp("Add", [shapeData, fullBeg]);
        const [targetShape] = builder.createOp("Add", [sumBeg, fullEnd], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);

        const axes0 = builder.createConstant(`padb_ax0_${op.id}`, int64Vec([0]));
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, axes0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, axes0]);

        return { totalIters, carryShape, targetShape };
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
        const builder = new GraphBuilder(body, `padloop_${op.id}`);
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;
        const k = isPresent(ins[3]) ? toStaticShape(ins[3].shape)[0] : rank;
        const dtype = ins[0].literalType as DataType;
        const mode = getStringAttr(op, "mode", "constant").toLowerCase();

        const rawData = resolveRecipeInput(
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
        const rawPads = resolveRecipeInput(
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
        const rawConstVal = isPresent(ins[2])
            ? resolveRecipeInput(
                  builder,
                  ins[2],
                  valueMap,
                  iter,
                  axes,
                  outShape,
                  false,
                  false,
                  targetShapeNode,
                  loopCtx,
              )
            : builder.createConstant(`pad_defval_${op.id}`, makeTensorProto(dtype, [], [0]));
        const rawAxes = isPresent(ins[3])
            ? resolveRecipeInput(
                  builder,
                  ins[3],
                  valueMap,
                  iter,
                  axes,
                  outShape,
                  false,
                  false,
                  targetShapeNode,
                  loopCtx,
              )
            : undefined;

        const [shapeData] = builder.createOp("Shape", [rawData], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const { fullBeg } = this.resolveFullPads(
            builder,
            rawPads,
            rawAxes,
            rank,
            k,
            `padi_${op.id}`,
        );
        const innerTargetShape = ensureInnerValueNode(builder, targetShapeNode!);
        const flatData = ensureFlatInput(builder, rawData);

        const gatherElem = (vec: ValueNode, idx: number, tag: string): ConcreteValueNode => {
            const idxC = builder.createConstant(`${tag}_${idx}`, int64Vec([idx]));
            const [raw] = builder.createOp("Gather", [vec, idxC], { axis: 0 });
            return squeezeIfLen1(builder, raw, axes, `${tag}_sq_${idx}`) as ConcreteValueNode;
        };

        const inDims: ConcreteValueNode[] = [];
        const outDims: ConcreteValueNode[] = [];
        const begDims: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            inDims.push(gatherElem(shapeData, d, `padi_id_${op.id}`));
            outDims.push(gatherElem(innerTargetShape, d, `padi_od_${op.id}`));
            begDims.push(gatherElem(fullBeg, d, `padi_bg_${op.id}`));
        }

        const outCoords: ConcreteValueNode[] =
            loopCtx && loopCtx.indexingMode === IndexingMode.Pointer
                ? getOrCreateIterationSpaceCounters(
                      builder,
                      loopCtx,
                      valueMap,
                      outDims,
                      `padi_cnt_${op.id}`,
                  ).counters
                : decodeMixedRadix(builder, iter, outDims, `padi_dec_${op.id}`);

        const zeroI64 = builder.createConstant(`padi_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`padi_o64_${op.id}`, scalarInt64(1));
        const twoI64 = builder.createConstant(`padi_t64_${op.id}`, scalarInt64(2));

        const inCoords: ConcreteValueNode[] = [];
        let isInBounds: ConcreteValueNode | null = null;

        for (let d = 0; d < rank; d++) {
            const [rawCoord] = builder.createOp("Sub", [outCoords[d], begDims[d]]);
            const dimD = inDims[d];
            const [maxIdx] = builder.createOp("Sub", [dimD, oneI64]);

            if (mode === "constant") {
                const [lt0] = builder.createOp("Less", [rawCoord, zeroI64]);
                const [ge0] = builder.createOp("Not", [lt0]);
                const [ltD] = builder.createOp("Less", [rawCoord, dimD]);
                const [validD] = builder.createOp("And", [ge0, ltD]);
                isInBounds = isInBounds ? builder.createOp("And", [isInBounds, validD])[0] : validD;
                const [safeC] = builder.createOp("Where", [validD, rawCoord, zeroI64]);
                inCoords.push(safeC);
            } else if (mode === "edge") {
                const [clampedLo] = builder.createOp("Max", [rawCoord, zeroI64]);
                const [safeC] = builder.createOp("Min", [clampedLo, maxIdx]);
                inCoords.push(safeC);
            } else if (mode === "wrap") {
                const [rem] = builder.createOp("Mod", [rawCoord, dimD]);
                const [remNeg] = builder.createOp("Less", [rem, zeroI64]);
                const [safeC] = builder.createOp("Where", [
                    remNeg,
                    builder.createOp("Add", [rem, dimD])[0],
                    rem,
                ]);
                inCoords.push(safeC);
            } else {
                // reflect: period p = max(2 * (dim - 1), 1)
                const [pRaw] = builder.createOp("Mul", [maxIdx, twoI64]);
                const [period] = builder.createOp("Max", [pRaw, oneI64]);
                const [r0] = builder.createOp("Mod", [rawCoord, period]);
                const [rNeg] = builder.createOp("Less", [r0, zeroI64]);
                const [r] = builder.createOp("Where", [
                    rNeg,
                    builder.createOp("Add", [r0, period])[0],
                    r0,
                ]);
                const [inFirstHalf] = builder.createOp("Less", [r, dimD]);
                const [mirrored] = builder.createOp("Sub", [period, r]);
                const [safeC] = builder.createOp("Where", [inFirstHalf, r, mirrored]);
                inCoords.push(safeC);
            }
        }

        const inStrides: (number | ValueNode)[] = new Array(rank).fill(1);
        let curStride: number | ValueNode = 1;
        for (let d = rank - 1; d >= 0; d--) {
            inStrides[d] = curStride;
            if (d > 0) {
                curStride =
                    typeof curStride === "number"
                        ? inDims[d]
                        : builder.createOp("Mul", [curStride, inDims[d]])[0];
            }
        }

        const flatInIdx = buildLinearIndex(builder, inCoords, inStrides, `padi_lin_${op.id}`);
        const [idxUnsq] = builder.createOp("Unsqueeze", [flatInIdx, axes]);
        const [gathered] = builder.createOp("Gather", [flatData, idxUnsq], { axis: 0 });
        const val = squeezeIfLen1(builder, gathered, axes, `padi_sq_${op.id}`) as ConcreteValueNode;

        if (mode === "constant" && isInBounds !== null) {
            const constScalar = squeezeIfLen1(builder, rawConstVal, axes, `padi_cval_${op.id}`);
            return builder.createOp("Where", [isInBounds, val, constScalar])[0];
        }
        return val;
    }
}
