import type OnnxGraph from "../../OnnxGraph.js";
import type OperationNode from "../../OperationNode.js";
import TensorNode from "../../TensorNode.js";
import type { GraphPass } from "../../PassManager.js";
import { DataType, type ValueNode } from "../../OnnxTypes.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { TransformationOpportunity } from "../TransformationOpportunity.js";
import type { DecompositionRecipe } from "../Recipe.js";
import {
    assignRegionScopeTag,
    getIntsAttr,
    getRegionScopeTag,
    topologicalSortOperationNodes,
    toStaticShape,
} from "../../utils/index.js";
import { LOOP_LOWERING_RECIPE_NAMES, lookupScheduleMap, type ScheduleConfig } from "../Schedule.js";
import { createDefaultCanonicalizationRecipes } from "../recipes.js";

/**
 * Normalization pass that rewrites high-level ONNX operations into simpler
 * primitive subgraphs using priority-ordered `DecompositionRecipe` instances.
 *
 * Supports optional recursion into control-flow regions (`recurse`), deferral of
 * ops with dedicated loop-lowering recipes (`loopLowering`), and 0D scalar-only
 * post-loop lowering (`scalar0DOnly`).
 */
export class CanonicalizationPass implements GraphPass {
    public readonly name = "Canonicalization";

    /** Maps ONNX opTypes to priority-sorted candidate recipes (first match wins). */
    private registry = new Map<string, DecompositionRecipe[]>();

    constructor(
        private customRecipes?: DecompositionRecipe[],
        private options: {
            loopLowering?: boolean;
            recurse?: boolean;
            regionOnly?: boolean;
            scalar0DOnly?: boolean;
            schedule?: ScheduleConfig;
        } = {},
    ) {
        const recipes = customRecipes ?? createDefaultCanonicalizationRecipes();
        for (const recipe of recipes) {
            this.register(recipe);
        }
    }

    /** Registers a recipe under its `targetOp`, maintaining descending priority order. */
    public register(recipe: DecompositionRecipe): void {
        const key = String(recipe.targetOp);
        const existing = this.registry.get(key);
        if (existing !== undefined) {
            existing.push(recipe);
            existing.sort((a, b) => b.priority - a.priority);
        } else {
            this.registry.set(key, [recipe]);
        }
    }

    /** Returns all registered canonicalization recipes for `opType` in priority order. */
    public getRecipesFor(opType: string): readonly DecompositionRecipe[] {
        return this.registry.get(opType) ?? [];
    }

    /**
     * Checks whether `op` should bypass algebraic canonicalization so that Phase 2
     * loop-lowering can emit a more compact fused or scalar Loop directly.
     */
    private shouldDeferToLoopLowering(op: OperationNode.Class): boolean {
        if (this.options.loopLowering !== true) return false;
        const ins = op.getInputs();
        if (!ins || ins.length === 0) return false;

        if (op.type === "Slice") {
            const rank = ins[0].shape.length;
            const k = ins.length > 1 ? toStaticShape(ins[1].shape)[0] : -1;
            return rank > 0 && k > 1;
        }

        if (op.type === "Concat") {
            if (ins.length < 2 || ins.length > 8) return false;
            const rank = ins[0].shape.length;
            if (rank <= 0) return false;
            const isSmallIntShape =
                rank === 1 &&
                (ins[0].literalType === DataType.INT64 || ins[0].literalType === DataType.INT32);
            return (
                !isSmallIntShape &&
                ins.every((t: ValueNode) => Array.isArray(t.shape) && t.shape.length === rank)
            );
        }

        if (op.type === "AveragePool") {
            const rank = ins[0].shape.length;
            if (rank < 3) return false;
            const kernel = getIntsAttr(op, "kernel_shape", []);
            return kernel.length === rank - 2;
        }

        if (op.type === "Det") {
            const dims = toStaticShape(ins[0].shape);
            if (dims.length < 2) return false;
            const n = dims[dims.length - 1];
            const m = dims[dims.length - 2];
            if ((n !== -1 && n < 1) || (m !== -1 && m < 1)) return false;
            if (n !== -1 && m !== -1 && n !== m) return false;
            // Keep algebraic closed-form LowerDet for static 1x1..3x3; defer N >= 4 or dynamic dims to LowerDetLoop.
            return n === -1 || m === -1 || n >= 4 || dims.slice(0, -2).includes(-1);
        }

        if (op.type === "TopK") {
            return (
                ins.length >= 2 &&
                Array.isArray(ins[0].shape) &&
                ins[0].shape.length >= 1 &&
                op.getOutputs().length >= 2
            );
        }

        return false;
    }

    public run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        let globalChanged = false;

        if (this.options.recurse === true) {
            const innerLoopLowering =
                this.options.regionOnly === true ? false : this.options.loopLowering;
            const innerPass = new CanonicalizationPass(this.customRecipes, {
                ...(innerLoopLowering !== undefined ? { loopLowering: innerLoopLowering } : {}),
                recurse: true,
                regionOnly: false,
                ...(this.options.scalar0DOnly !== undefined
                    ? { scalar0DOnly: this.options.scalar0DOnly }
                    : {}),
                ...(this.options.schedule !== undefined ? { schedule: this.options.schedule } : {}),
            });
            for (const op of graph.getOperationNodes()) {
                for (const region of op.regions) {
                    assignRegionScopeTag(region, "canon");
                    if (innerPass.run(region, historyManager)) {
                        globalChanged = true;
                    }
                }
            }
        }

        if (this.options.regionOnly === true) {
            return globalChanged;
        }

        const isScalar0DOp = (op: OperationNode.Class): boolean => {
            const ins = op.getInputs() ?? [];
            const outs = op.getOutgoers.targets.filterIs(TensorNode).toArray();
            if (outs.length === 0) return false;
            return (
                ins.every((v) => Array.isArray(v.shape) && v.shape.length === 0) &&
                outs.every((v) => Array.isArray(v.shape) && v.shape.length === 0)
            );
        };

        let localChanged = true;

        while (localChanged) {
            localChanged = false;
            const ops = topologicalSortOperationNodes(graph);

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                if (this.options.scalar0DOnly === true) {
                    if (!isScalar0DOp(op)) continue;
                    // Preserve integer index Mod arithmetic inside Loop bodies
                    const dt = op.getInputs()?.[0]?.literalType as DataType | undefined;
                    if (
                        op.type === "Mod" &&
                        (dt === DataType.INT64 ||
                            dt === DataType.INT32 ||
                            dt === DataType.INT16 ||
                            dt === DataType.INT8 ||
                            dt === DataType.UINT64 ||
                            dt === DataType.UINT32 ||
                            dt === DataType.UINT16 ||
                            dt === DataType.UINT8)
                    ) {
                        continue;
                    }
                }

                const stopPhase = lookupScheduleMap(this.options.schedule?.stopAt, op);
                if (stopPhase === "none") continue;

                const preferredRecipe = lookupScheduleMap(
                    this.options.schedule?.recipeOverrides,
                    op,
                );
                if (
                    preferredRecipe !== undefined &&
                    this.options.loopLowering !== undefined &&
                    LOOP_LOWERING_RECIPE_NAMES.has(preferredRecipe)
                ) {
                    continue;
                }
                if (preferredRecipe === undefined && this.shouldDeferToLoopLowering(op)) continue;

                const candidates = this.registry.get(op.type);
                if (candidates === undefined) continue;

                const recipe =
                    preferredRecipe !== undefined
                        ? (candidates.find((c) => c.name === preferredRecipe && c.match(op)) ??
                          candidates.find((c) => c.match(op)))
                        : candidates.find((candidate) => candidate.match(op));
                if (recipe === undefined) continue;

                const opportunity = new TransformationOpportunity(
                    recipe.name,
                    op.id,
                    `Apply ${recipe.name} to ${op.type}`,
                    (builder) => recipe.apply(op, builder),
                );

                const prevOps = op.getMetadata("sourceOps");
                const prevTypes = op.getMetadata("sourceOpTypes");
                const inheritedSrcOps = Array.from(
                    new Set([...(Array.isArray(prevOps) ? prevOps.map(String) : []), op.id]),
                );
                const inheritedSrcTypes = Array.from(
                    new Set([...(Array.isArray(prevTypes) ? prevTypes.map(String) : []), op.type]),
                );
                const beforeIds = new Set(
                    graph
                        .getOperationNodes()
                        .toArray()
                        .map((o) => o.id),
                );

                const builder = new TrackedGraphBuilder(
                    graph,
                    opportunity.id,
                    opportunity.description,
                    getRegionScopeTag(graph, "canonicalization"),
                );

                opportunity.apply(builder);

                for (const newOp of graph.getOperationNodes().toArray()) {
                    if (!beforeIds.has(newOp.id)) {
                        newOp.setMetadata("sourceOps", inheritedSrcOps);
                        newOp.setMetadata("sourceOpTypes", inheritedSrcTypes);
                    }
                }

                localChanged = true;
                globalChanged = true;

                const patch = builder.commitPatch();
                historyManager.pushPatch(patch);
            }
        }

        return globalChanged;
    }
}
