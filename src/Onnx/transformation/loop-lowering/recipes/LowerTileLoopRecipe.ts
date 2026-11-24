import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import {
    type ValueNode,
    type ConcreteValueNode,
    type KnownShape,
    DataType,
} from "../../../OnnxTypes.js";
import { int64Vec, scalarInt64 } from "../../../Utils.js";
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

export class LowerTileLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerTileLoop";
    public readonly targetOp = "Tile";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 10;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["tiling", "dynamic-repeats", "fuseable"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Reshape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Add",
        "Mul",
        "Div",
        "Mod",
        "Equal",
        "And",
        "Where",
        "Cast",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Tile") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2) return false;
        return Array.isArray(ins[0].shape) && ins[0].shape.length > 0;
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
        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `tile_bounds_${op.id}`);

        const [shapeIn] = builder.createOp("Shape", [ins[0]], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);
        const repeats64: ConcreteValueNode =
            ins[1].literalType === DataType.INT64
                ? (ins[1] as ConcreteValueNode)
                : builder.createOp("Cast", [ins[1]], { to: DataType.INT64 })[0];

        const [targetShape] = builder.createOp("Mul", [shapeIn, repeats64], {}, [
            { type: DataType.INT64, shape: [rank] },
        ]);

        const axes0 = builder.createConstant(`tileb_ax0_${op.id}`, int64Vec([0]));
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
        const builder = new GraphBuilder(body, `tileloop_${op.id}`);
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;

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
        const innerTargetShape = ensureInnerValueNode(builder, targetShapeNode!);
        const flatIn = ensureFlatInput(builder, rawIn);

        const gatherElem = (vec: ValueNode, idx: number, tag: string): ConcreteValueNode => {
            const idxC = builder.createConstant(`${tag}_${idx}`, int64Vec([idx]));
            const [raw] = builder.createOp("Gather", [vec, idxC], { axis: 0 });
            return squeezeIfLen1(builder, raw, axes, `${tag}_sq_${idx}`) as ConcreteValueNode;
        };

        const inDims: ConcreteValueNode[] = [];
        const outDims: ConcreteValueNode[] = [];
        for (let d = 0; d < rank; d++) {
            inDims.push(gatherElem(shapeIn, d, `tl_id_${op.id}`));
            outDims.push(gatherElem(innerTargetShape, d, `tl_od_${op.id}`));
        }

        const inCoords: ConcreteValueNode[] = [];
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { wraps } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                outDims,
                `tl_cnt_${op.id}`,
            );
            const zeroI64 = builder.createConstant(`tl_z64_${op.id}`, scalarInt64(0));
            const oneI64 = builder.createConstant(`tl_o64_${op.id}`, scalarInt64(1));

            for (let d = 0; d < rank; d++) {
                const initC = loopCtx.outerBuilder.createConstant(
                    `tl_init_${op.id}_${d}`,
                    scalarInt64(0),
                );
                const cIn = loopCtx.addCarry(`tl_ic_${op.id}_${d}`, initC, DataType.INT64, []);
                const [cInc] = builder.createOp("Add", [cIn, oneI64]);
                const [atEnd] = builder.createOp("Equal", [cInc, inDims[d]]);
                const [wrappedC] = builder.createOp("Where", [atEnd, zeroI64, cInc]);

                const nextC =
                    d === rank - 1
                        ? wrappedC
                        : builder.createOp("Where", [wraps[d + 1], wrappedC, cIn])[0];

                loopCtx.setNextState(cIn, nextC);
                inCoords.push(cIn);
            }
        } else {
            const outCoords = decodeMixedRadix(builder, iter, outDims, `tl_dec_${op.id}`);
            for (let d = 0; d < rank; d++) {
                const [ic] = builder.createOp("Mod", [outCoords[d], inDims[d]]);
                inCoords.push(ic);
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

        const flatInIdx = buildLinearIndex(builder, inCoords, inStrides, `tl_lin_${op.id}`);
        const [idxUnsq] = builder.createOp("Unsqueeze", [flatInIdx, axes]);
        const [gathered] = builder.createOp("Gather", [flatIn, idxUnsq], { axis: 0 });
        return squeezeIfLen1(builder, gathered, axes, `tl_sq_${op.id}`);
    }
}
