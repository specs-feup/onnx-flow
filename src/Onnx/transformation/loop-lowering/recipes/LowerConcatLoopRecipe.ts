import type OnnxGraph from "../../../OnnxGraph.js";
import type OperationNode from "../../../OperationNode.js";
import { type ValueNode, type ConcreteValueNode, type KnownShape } from "../../../OnnxTypes.js";
import { getIntAttr, int64Vec, scalarInt64, toStaticShape } from "../../../Utils.js";
import type {
    LoopCarryContext,
    LoopLoweringRecipe,
    RecipeApplyResult,
} from "../LoopLoweringRecipe.js";
import {
    addMixed,
    buildLinearIndex,
    buildMixedStrides,
    captureMixedList,
    decodeMixedRadix,
    ensureFlatInput,
    extractMixedDims,
    gatherScalar,
    getOrCreateIterationSpaceCounters,
    packShapeVector,
    resolveRecipeInput,
} from "../RecipeUtils.js";
import { GraphBuilder } from "../../../GraphBuilder.js";
import { IndexingMode } from "../../../../DecompositionOptions.js";

export class LowerConcatLoopRecipe implements LoopLoweringRecipe {
    public readonly name = "LowerConcatLoop";
    public readonly targetOp = "Concat";
    public readonly phase = "loop-lowering" as const;
    public readonly precision = "exact" as const;
    public readonly priority = 20;
    public readonly shapeSupport = "dynamic" as const;
    public readonly tags = ["concat", "dynamic-bounds", "fuseable"] as const;
    public readonly exposesControlFlow = true;
    public readonly exposesDataAccess = true;
    public readonly producedOps = [
        "Loop",
        "Shape",
        "Concat",
        "ReduceProd",
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
        "Equal",
        "And",
        "Where",
    ];

    match(op: OperationNode.Class): boolean {
        if (op.type !== "Concat") return false;
        const ins = op.getInputs();
        if (!ins || ins.length < 2 || ins.length > 8) return false;
        const rank = ins[0].shape.length;
        if (rank <= 0) return false;
        return ins.every((t) => Array.isArray(t.shape) && t.shape.length === rank);
    }

    private normalizeAxis(op: OperationNode.Class, rank: number): number {
        const ax = getIntAttr(op, "axis", 0);
        return ax < 0 ? rank + ax : ax;
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
        const rank = ins[0].shape.length;
        const axis = this.normalizeAxis(op, rank);

        const rawPerIn = ins.map((t) => toStaticShape(t.shape));
        const sharedDimsStatic: number[] = [];
        let allStatic = true;

        for (let d = 0; d < rank; d++) {
            if (d === axis) {
                if (rawPerIn.every((s) => s[axis] >= 0)) {
                    sharedDimsStatic.push(rawPerIn.reduce((sum, s) => sum + s[axis], 0));
                } else {
                    sharedDimsStatic.push(-1);
                    allStatic = false;
                }
            } else {
                const known = rawPerIn.find((s) => s[d] >= 0)?.[d] ?? -1;
                sharedDimsStatic.push(known);
                if (known < 0) allStatic = false;
            }
        }

        if (allStatic) {
            const totalIters = sharedDimsStatic.reduce((a, b) => a * b, 1);
            return { totalIters, carryShape: sharedDimsStatic };
        }

        const builder = new GraphBuilder(op.graph as OnnxGraph.Class, `cat_bounds_${op.id}`);
        const inMixedDims = ins.map((inp, j) =>
            extractMixedDims(builder, inp, rawPerIn[j], `catb_${j}_${op.id}`),
        );

        const outDims: (number | ConcreteValueNode)[] = [];
        for (let d = 0; d < rank; d++) {
            if (d === axis) {
                let acc: number | ConcreteValueNode = inMixedDims[0][axis];
                for (let j = 1; j < ins.length; j++) {
                    acc = addMixed(builder, acc, inMixedDims[j][axis], `catb_ax_${j}_${op.id}`);
                }
                outDims.push(acc);
            } else {
                const known = rawPerIn.findIndex((s) => s[d] >= 0);
                outDims.push(known !== -1 ? rawPerIn[known][d] : inMixedDims[0][d]);
            }
        }

        const targetShape = packShapeVector(builder, outDims, `catb_tsh_${op.id}`);
        const ax0 = builder.createConstant(`catb_ax0_${op.id}`, int64Vec([0]));
        const [totalIters] = builder.createOp("ReduceProd", [targetShape, ax0], { keepdims: 0 });
        const [carryShape] = builder.createOp("Unsqueeze", [totalIters, ax0]);
        return { totalIters, carryShape, targetShape };
    }

    apply(
        op: OperationNode.Class,
        body: OnnxGraph.Class,
        valueMap: Map<string, ValueNode>,
        iter: ConcreteValueNode,
        _axes: ConcreteValueNode,
        outShape: KnownShape,
        _carryNode: ConcreteValueNode,
        targetShapeNode?: ValueNode,
        loopCtx?: LoopCarryContext,
    ): RecipeApplyResult {
        const builder = new GraphBuilder(body, `concat_${op.id}`);
        const ins = op.getInputs()!;
        const rank = ins[0].shape.length;
        const axis = this.normalizeAxis(op, rank);
        const canHoist =
            loopCtx !== undefined &&
            ins.every((t) => !valueMap.has(t.id) && loopCtx.outerBuilder.graph.hasNode(t.id));

        const rawIns = ins.map((inp) =>
            resolveRecipeInput(
                builder,
                inp,
                valueMap,
                iter,
                _axes,
                outShape,
                false,
                false,
                targetShapeNode,
                loopCtx,
            ),
        );
        const flatIns = rawIns.map((r) => ensureFlatInput(builder, r));
        const rawStatic = ins.map((inp) => toStaticShape(inp.shape));
        const dimBuilder = canHoist ? loopCtx!.outerBuilder : builder;
        const rawMixedPerIn = (canHoist ? ins : rawIns).map((r, j) =>
            extractMixedDims(dimBuilder, r, rawStatic[j], `cat_id_${j}_${op.id}`),
        );
        const rawStridesPerIn = rawMixedPerIn.map((dims) => buildMixedStrides(dimBuilder, dims));

        const rawOffsets: (number | ConcreteValueNode)[] = [0];
        let accAxis: number | ConcreteValueNode = 0;
        for (let j = 0; j < ins.length; j++) {
            accAxis = addMixed(
                dimBuilder,
                accAxis,
                rawMixedPerIn[j][axis],
                `cat_off_${j}_${op.id}`,
            );
            rawOffsets.push(accAxis);
        }

        const inMixedDims = canHoist
            ? rawMixedPerIn.map((d) => captureMixedList(builder, d))
            : rawMixedPerIn;
        const inStrides = canHoist
            ? rawStridesPerIn.map((s) => captureMixedList(builder, s))
            : rawStridesPerIn;
        const offsets = canHoist ? captureMixedList(builder, rawOffsets) : rawOffsets;

        const outDims: (number | ConcreteValueNode)[] = [];
        for (let d = 0; d < rank; d++) {
            if (d === axis) {
                outDims.push(offsets[ins.length]);
            } else {
                const known = rawStatic.findIndex((s) => s[d] >= 0);
                outDims.push(known !== -1 ? rawStatic[known][d] : inMixedDims[0][d]);
            }
        }

        let coords: ConcreteValueNode[];
        if (loopCtx && loopCtx.indexingMode === IndexingMode.Pointer) {
            const { counters } = getOrCreateIterationSpaceCounters(
                builder,
                loopCtx,
                valueMap,
                outDims,
                `cat_cnt_${op.id}`,
            );
            coords = counters;
        } else {
            coords = decodeMixedRadix(builder, iter, outDims, `cat_dec_${op.id}`);
        }

        const zeroI64 = builder.createConstant(`cat_z64_${op.id}`, scalarInt64(0));
        const oneI64 = builder.createConstant(`cat_o64_${op.id}`, scalarInt64(1));
        const axCoord = coords[axis];

        const gathered: ConcreteValueNode[] = [];
        for (let j = 0; j < ins.length; j++) {
            const offVal = offsets[j];
            let localAx: ConcreteValueNode;
            if (typeof offVal === "number" && offVal === 0) {
                localAx = axCoord;
            } else {
                const offNode =
                    typeof offVal === "number"
                        ? builder.createConstant(`cat_offc_${j}_${op.id}`, scalarInt64(offVal))
                        : offVal;
                [localAx] = builder.createOp("Sub", [axCoord, offNode]);
            }

            const dimJ = inMixedDims[j][axis];
            const maxIdxJ: ConcreteValueNode =
                typeof dimJ === "number"
                    ? builder.createConstant(
                          `cat_mx_${j}_${op.id}`,
                          scalarInt64(Math.max(0, dimJ - 1)),
                      )
                    : builder.createOp("Max", [
                          zeroI64,
                          builder.createOp("Sub", [dimJ, oneI64])[0],
                      ])[0];

            const [nonNegAx] = builder.createOp("Max", [localAx, zeroI64]);
            const [safeAx] = builder.createOp("Min", [nonNegAx, maxIdxJ]);

            const coordsJ = coords.slice();
            coordsJ[axis] = safeAx;
            const linIdxJ = buildLinearIndex(
                builder,
                coordsJ,
                inStrides[j],
                `cat_lin_${j}_${op.id}`,
            );
            gathered.push(gatherScalar(builder, flatIns[j], linIdxJ));
        }

        let selected: ConcreteValueNode = gathered[gathered.length - 1];
        for (let j = gathered.length - 2; j >= 0; j--) {
            const bound = offsets[j + 1];
            const boundNode =
                typeof bound === "number"
                    ? builder.createConstant(`cat_bnd_${j}_${op.id}`, scalarInt64(bound))
                    : bound;
            const [isLower] = builder.createOp("Less", [axCoord, boundNode]);
            [selected] = builder.createOp("Where", [isLower, gathered[j], selected]);
        }

        return selected;
    }
}
