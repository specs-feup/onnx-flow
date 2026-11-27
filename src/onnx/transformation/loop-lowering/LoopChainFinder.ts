import type OnnxGraph from "../../ir/OnnxGraph.js";
import OperationNode from "../../ir/OperationNode.js";
import TensorNode from "../../ir/TensorNode.js";
import { DataType, type ValueNode, type KnownShape } from "../../ir/OnnxTypes.js";
import {
    asStaticDims,
    toStaticShape,
    UNKNOWN_SHAPE,
    topologicalSortOperationNodes,
} from "../../utils/index.js";
import {
    lookupScheduleMap,
    matchesScheduleTarget,
    type ScheduleConfig,
} from "../schedule/Schedule.js";
import { OpCategory } from "../../schema/OpSchema.js";
import { OpRegistry } from "../../schema/OpRegistry.js";
import type { IndexingMode } from "../../../DecompositionOptions.js";
import type { LoopLoweringRecipe } from "./LoopLoweringRecipe.js";
import { createDefaultLoopLoweringRecipes } from "../RecipeCatalog.js";

const SHAPE_PLUMBING_CONSUMERS = new Set([
    "Reshape",
    "Expand",
    "ConstantOfShape",
    "Slice",
    "Gather",
    "ScatterElements",
    "Loop",
    "Concat",
    "Unsqueeze",
    "Squeeze",
    "Range",
]);

export interface LoopFusionOptions {
    coalesce: boolean;
    fuse: boolean;
    indexingMode: IndexingMode;
    useScanOutputs?: boolean;
    fuseMultiOutput?: boolean;
    /** When true, skips 0D scalar compute ops and 1D integer shape/index plumbing ops. */
    inRegion?: boolean;
    schedule?: ScheduleConfig;
}

export interface LoopBounds {
    totalIters: number | ValueNode;
    carryShape: KnownShape | ValueNode;
    targetShape?: KnownShape | ValueNode;
}

/**
 * Discovers single-operation and fused multi-operation loop-lowering chains in an `OnnxGraph`,
 * validating shape/bounds compatibility, fusion legality, and schedule directives.
 */
export class LoopChainFinder {
    private boundsCache = new Map<string, LoopBounds>();
    private recipes: LoopLoweringRecipe[];

    constructor(
        private readonly options: LoopFusionOptions,
        customRecipes?: LoopLoweringRecipe[],
    ) {
        this.recipes =
            customRecipes !== undefined
                ? [...customRecipes]
                : createDefaultLoopLoweringRecipes(this.options.coalesce);

        this.recipes.sort((a, b) => b.priority - a.priority);
    }

    /** Clears cached loop bounds between discovery passes. */
    public clearCache(): void {
        this.boundsCache.clear();
    }

    /** Discovers chains according to the configured fusion mode (`fuse` vs. single-op). */
    public findChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        this.clearCache();
        return this.options.fuse ? this.findFuseableChains(graph) : this.findSingleOpChains(graph);
    }

    /**
     * Prevents infinite re-lowering inside nested regions by skipping 0D scalar ops
     * and 1D integer shape/index plumbing ops (e.g., `packShapeVector` `Concat`,
     * `ReduceProd` carry-size calculations, or `INT64` index vectors).
     */
    private isScalarOrLoopPlumbing(op: OperationNode.Class): boolean {
        if (op.getMetadata("loopPlumbing") === true) return true;

        const ins = op.getInputs() ?? [];
        const outs = op.getOutgoers.targets.filterIs(TensorNode).toArray();
        if (outs.length === 0) return false;

        // Already scalarized (0D inputs & 0D outputs)
        const allInputs0D = ins.every((v) => Array.isArray(v.shape) && v.shape.length === 0);
        const allOutputs0D = outs.every((v) => Array.isArray(v.shape) && v.shape.length === 0);
        if (allInputs0D && allOutputs0D) return true;

        // 1D integer shape/index plumbing vectors inside loop regions
        // (never skip `Range`: it is a non-native generator that must lower to a Loop if not constant-folded)
        const is1DIntOut =
            op.type !== "Range" &&
            outs.every(
                (v) =>
                    Array.isArray(v.shape) &&
                    v.shape.length === 1 &&
                    (v.literalType === DataType.INT64 || v.literalType === DataType.INT32),
            );
        if (is1DIntOut) {
            if (op.type === "Concat" || op.type === "ReduceProd" || op.type === "Slice") {
                return true;
            }
            const consumers = outs.flatMap((o) =>
                o.outgoers.targets.filterIs(OperationNode).toArray(),
            );
            if (
                consumers.length > 0 &&
                consumers.every((c) => SHAPE_PLUMBING_CONSUMERS.has(c.type))
            ) {
                return true;
            }
        }

        return false;
    }

    /** Resolves the highest-priority `LoopLoweringRecipe` matching `op`, respecting schedule overrides. */
    public getRecipeFor(op: OperationNode.Class): LoopLoweringRecipe | undefined {
        if (this.options.inRegion === true && this.isScalarOrLoopPlumbing(op)) {
            return undefined;
        }
        const stopPhase = lookupScheduleMap(this.options.schedule?.stopAt, op);
        if (stopPhase === "none" || stopPhase === "canonicalization") {
            return undefined;
        }
        const preferred = lookupScheduleMap(this.options.schedule?.recipeOverrides, op);
        if (preferred !== undefined) {
            const match = this.recipes.find((r) => r.name === preferred && r.match(op));
            if (match !== undefined) return match;
        }
        return this.recipes.find((recipe) => recipe.match(op));
    }

    /** Returns true when `op` has a matching loop-lowering recipe under the active schedule. */
    public isSupported(op: OperationNode.Class): boolean {
        return this.getRecipeFor(op) !== undefined;
    }

    /** Computes and caches iteration bounds and carry buffer shapes for `op`. */
    public getBoundsFor(op: OperationNode.Class): LoopBounds {
        const cached = this.boundsCache.get(op.id);
        if (cached !== undefined) return cached;

        const recipe = this.getRecipeFor(op)!;
        const outTensors = op.getOutgoers.targets.filterIs(TensorNode).toArray();
        const originalOutShape: KnownShape =
            outTensors.length > 0 ? [...(outTensors[0].shape as KnownShape)] : UNKNOWN_SHAPE;
        const safeOut = asStaticDims(originalOutShape);

        if (recipe.getLoopBounds !== undefined) {
            const result = recipe.getLoopBounds(op, originalOutShape);
            this.boundsCache.set(op.id, result);
            return result;
        }

        if (
            (originalOutShape.includes(-1) || originalOutShape.length === 0) &&
            outTensors.length > 0
        ) {
            throw new Error(`Dynamic output shape missing bounds in recipe for ${op.type}`);
        }

        const totalIters = safeOut.length === 0 ? 1 : safeOut.reduce((a, b) => a * b, 1);
        const result: LoopBounds = { totalIters, carryShape: [totalIters] };
        this.boundsCache.set(op.id, result);
        return result;
    }

    public findSingleOpChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        return topologicalSortOperationNodes(graph)
            .filter((op) => this.isSupported(op))
            .map((op) => [op]);
    }

    public findFuseableChains(graph: OnnxGraph.Class): OperationNode.Class[][] {
        const chains: OperationNode.Class[][] = [];
        const ops = topologicalSortOperationNodes(graph);
        const visited = new Set<string>();
        const registry = OpRegistry.getInstance();
        const opset = graph.getTargetOpset();
        const originCache = new Map<string, string>();

        const isBarrierRecipe = (recipe: LoopLoweringRecipe): boolean =>
            recipe.tags?.includes("self-carry") === true ||
            recipe.postProcess !== undefined ||
            recipe.lowerStandalone !== undefined;

        const isStaticTensorShape = (t: TensorNode.Class | undefined): boolean => {
            if (!t || !Array.isArray(t.shape) || t.shape.length === 0) return false;
            return !(t.shape as (number | string)[]).includes(-1);
        };

        const boundsCompatible = (
            prod: OperationNode.Class,
            root: OperationNode.Class,
        ): boolean => {
            const prodOut = prod.getOutgoers.targets.filterIs(TensorNode).first();
            const rootOut = root.getOutgoers.targets.filterIs(TensorNode).first();
            if (!prodOut || !rootOut) return false;

            // Dynamic diamond check FIRST (pure read-only, zero graph mutations)
            if (
                (!isStaticTensorShape(prodOut) || !isStaticTensorShape(rootOut)) &&
                this.getUnaryShapeOrigin(prodOut, opset, registry, originCache) ===
                    this.getUnaryShapeOrigin(rootOut, opset, registry, originCache)
            ) {
                return true;
            }

            if (isStaticTensorShape(prodOut) && isStaticTensorShape(rootOut)) {
                const prodBounds = this.getBoundsFor(prod);
                const rootBounds = this.getBoundsFor(root);
                return (
                    this.compareIters(prodBounds.totalIters, rootBounds.totalIters) &&
                    this.compareCarryShapes(prodBounds.carryShape, rootBounds.carryShape)
                );
            }

            return false;
        };

        const fuseMulti = this.options.fuseMultiOutput === true;
        const computeAtPairs = this.options.schedule?.computeAtList ?? [];
        const hasComputeAt = (
            prod: OperationNode.Class,
            clusterOps: OperationNode.Class[],
        ): boolean =>
            computeAtPairs.some(
                (p) =>
                    matchesScheduleTarget(p.producer, prod) &&
                    clusterOps.some((c) => matchesScheduleTarget(p.consumer, c)),
            );

        // Build forward op adjacency for convexity checks (prevents multi-output fusion
        // from absorbing a sibling consumer that feeds back into the cluster through an
        // external non-elementwise op like ReduceMean in LayerNorm/GroupNorm/InstanceNorm).
        const forwardAdj = new Map<string, OperationNode.Class[]>();
        for (const op of ops) {
            const succs: OperationNode.Class[] = [];
            const seenSucc = new Set<string>();
            for (const outT of op.getOutgoers.targets.filterIs(TensorNode).toArray()) {
                for (const c of outT.outgoers.targets.filterIs(OperationNode).toArray()) {
                    if (!seenSucc.has(c.id)) {
                        seenSucc.add(c.id);
                        succs.push(c);
                    }
                }
            }
            forwardAdj.set(op.id, succs);
        }

        const isConvexCluster = (proposedSet: Set<string>): boolean => {
            const queue: OperationNode.Class[] = [];
            const seen = new Set<string>();
            for (const opId of proposedSet) {
                for (const succ of forwardAdj.get(opId) ?? []) {
                    if (!proposedSet.has(succ.id) && !seen.has(succ.id)) {
                        seen.add(succ.id);
                        queue.push(succ);
                    }
                }
            }
            while (queue.length > 0) {
                const curr = queue.shift()!;
                for (const next of forwardAdj.get(curr.id) ?? []) {
                    if (proposedSet.has(next.id)) return false;
                    if (!seen.has(next.id)) {
                        seen.add(next.id);
                        queue.push(next);
                    }
                }
            }
            return true;
        };

        // 0. Pre-form explicit Schedule.fuse(...) multi-op / multi-output clusters
        for (const groupTargets of this.options.schedule?.fuseGroups ?? []) {
            const members: OperationNode.Class[] = [];
            for (const op of ops) {
                if (visited.has(op.id)) continue;
                if (!groupTargets.some((t) => matchesScheduleTarget(t, op))) continue;
                const r = this.getRecipeFor(op);
                if (!r || isBarrierRecipe(r)) continue;
                if (registry.get(op.type, opset)?.category !== OpCategory.ElementWise) continue;
                members.push(op);
            }
            if (members.length >= 2) {
                const anchor = members[members.length - 1];
                const compatible = members.filter(
                    (m) => m.id === anchor.id || boundsCompatible(m, anchor),
                );
                if (
                    compatible.length >= 2 &&
                    isConvexCluster(new Set(compatible.map((m) => m.id)))
                ) {
                    compatible.forEach((m) => visited.add(m.id));
                    chains.push(this.localTopologicalSort(compatible));
                }
            }
        }

        // Grow convex single-output DAG clusters in reverse topological order (consumers before producers).
        // Supports diamond DAGs (e.g. Swish, Gelu, Reciprocal) while stopping cleanly at multi-output forks
        // unless fuseMultiOutput=true or explicitly scheduled via Schedule.fuse / Schedule.computeAt.
        for (let i = ops.length - 1; i >= 0; i--) {
            const root = ops[i];
            if (visited.has(root.id)) continue;
            const rootRecipe = this.getRecipeFor(root);
            if (rootRecipe === undefined) continue;

            if (rootRecipe.lowerStandalone !== undefined) {
                visited.add(root.id);
                chains.push([root]);
                continue;
            }

            const clusterSet = new Set<string>([root.id]);
            const cluster: OperationNode.Class[] = [root];
            visited.add(root.id);

            const rootIsElementWise =
                registry.get(root.type, opset)?.category === OpCategory.ElementWise &&
                !isBarrierRecipe(rootRecipe);

            // Only chains whose consumers are elementwise can absorb compatible producers
            let grew = rootIsElementWise;
            while (grew) {
                grew = false;
                for (const member of [...cluster]) {
                    const inTensors =
                        member
                            .getInputs()
                            ?.filter(
                                (n) => n.is(TensorNode) && n.as(TensorNode).type === "intermediate",
                            ) ?? [];

                    for (const t of inTensors) {
                        const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                        for (const prod of producers) {
                            if (visited.has(prod.id)) continue;
                            const prodRecipe = this.getRecipeFor(prod);
                            if (prodRecipe === undefined || isBarrierRecipe(prodRecipe)) continue;

                            const prodOuts = prod.getOutgoers.targets
                                .filterIs(TensorNode)
                                .toArray();
                            if (prodOuts.length !== 1) continue;
                            const isOutputAllowed = fuseMulti || hasComputeAt(prod, cluster);
                            if (prodOuts[0].type === "output" && !isOutputAllowed) continue;

                            const consumers = prodOuts[0].outgoers.targets
                                .filterIs(OperationNode)
                                .toArray();
                            if (consumers.length === 0) continue;

                            const outsideConsumers = consumers.filter((c) => !clusterSet.has(c.id));
                            if (outsideConsumers.length > 0) {
                                if (!fuseMulti) continue;
                                const canAbsorbSiblings = outsideConsumers.every((c) => {
                                    if (visited.has(c.id)) return false;
                                    const cr = this.getRecipeFor(c);
                                    const cSchema = registry.get(c.type, opset);
                                    return (
                                        cr !== undefined &&
                                        !isBarrierRecipe(cr) &&
                                        cSchema?.category === OpCategory.ElementWise &&
                                        boundsCompatible(c, root)
                                    );
                                });
                                if (!canAbsorbSiblings) continue;
                            }

                            const allClusterConsumersEW = consumers
                                .filter((c) => clusterSet.has(c.id))
                                .every(
                                    (c) =>
                                        registry.get(c.type, opset)?.category ===
                                        OpCategory.ElementWise,
                                );
                            if (!allClusterConsumersEW) continue;
                            if (!isStaticTensorShape(prodOuts[0])) {
                                const prodSig = toStaticShape(prodOuts[0].shape).join(",");
                                const hasSameShapeInput = (prod.getInputs() ?? []).some(
                                    (inp) => toStaticShape(inp.shape).join(",") === prodSig,
                                );
                                if (!hasSameShapeInput) continue;
                            }
                            if (!boundsCompatible(prod, root)) continue;

                            const proposedSet = new Set(clusterSet);
                            proposedSet.add(prod.id);
                            for (const sib of outsideConsumers) proposedSet.add(sib.id);
                            if (!isConvexCluster(proposedSet)) continue;

                            visited.add(prod.id);
                            clusterSet.add(prod.id);
                            cluster.push(prod);
                            for (const sib of outsideConsumers) {
                                visited.add(sib.id);
                                clusterSet.add(sib.id);
                                cluster.push(sib);
                            }
                            grew = true;
                        }
                    }
                }
            }

            chains.push(this.localTopologicalSort(cluster));
        }

        // Preserve topological execution order across clusters
        const pos = new Map<string, number>();
        ops.forEach((o, idx) => pos.set(o.id, idx));
        chains.sort(
            (a, b) => (pos.get(a[a.length - 1].id) ?? 0) - (pos.get(b[b.length - 1].id) ?? 0),
        );
        return chains;
    }

    private compareIters(a: number | ValueNode, b: number | ValueNode): boolean {
        if (typeof a === "number" && typeof b === "number") return a === b;
        if (typeof a !== "number" && typeof b !== "number") return a.id === b.id;
        return false;
    }

    private compareCarryShapes(a: KnownShape | ValueNode, b: KnownShape | ValueNode): boolean {
        const isArrayA = Array.isArray(a);
        const isArrayB = Array.isArray(b);
        if (isArrayA && isArrayB) {
            return (a as KnownShape).join(",") === (b as KnownShape).join(",");
        }
        if (!isArrayA && !isArrayB) return (a as ValueNode).id === (b as ValueNode).id;
        return false;
    }

    /**
     * Traces a chain of ElementWise ops whose non-scalar input determines the exact runtime shape,
     * enabling dynamic diamond DAG fusion without false broadcast merges.
     */
    private getUnaryShapeOrigin(
        tensor: ValueNode,
        opset: number,
        registry: OpRegistry,
        originCache: Map<string, string>,
        visiting = new Set<string>(),
    ): string {
        const cached = originCache.get(tensor.id);
        if (cached !== undefined) return cached;
        if (visiting.has(tensor.id)) return tensor.id;
        visiting.add(tensor.id);

        let result = tensor.id;
        const prod = tensor.is(TensorNode)
            ? tensor.incomers.sources.filterIs(OperationNode).first()
            : undefined;
        if (prod !== undefined) {
            const schema = registry.get(prod.type, opset);
            if (schema?.category === OpCategory.ElementWise) {
                const outRank = (tensor.shape as unknown[] | undefined)?.length ?? 0;
                const nonScalarInputs = (prod.getInputs() ?? []).filter((inp) => {
                    const sh = inp.shape as (number | string)[] | undefined;
                    if (!sh || sh.length === 0) return false;
                    return !(sh.length <= outRank && sh.every((d) => d === 1));
                });
                if (nonScalarInputs.length >= 1) {
                    const origins = nonScalarInputs.map((inp) =>
                        this.getUnaryShapeOrigin(inp, opset, registry, originCache, visiting),
                    );
                    if (origins.every((o) => o === origins[0])) {
                        result = origins[0];
                    }
                }
            }
        }

        visiting.delete(tensor.id);
        originCache.set(tensor.id, result);
        return result;
    }

    private localTopologicalSort(ops: OperationNode.Class[]): OperationNode.Class[] {
        const visited = new Set<string>();
        const sorted: OperationNode.Class[] = [];

        const visit = (node: OperationNode.Class): void => {
            if (visited.has(node.id)) return;
            visited.add(node.id);

            const inTensors = node.getInputs()?.filter((v) => v.is(TensorNode)) ?? [];
            for (const t of inTensors) {
                const producers = t.incomers.sources.filterIs(OperationNode).toArray();
                for (const prod of producers) {
                    if (ops.some((o) => o.id === prod.id)) visit(prod);
                }
            }
            sorted.push(node);
        };

        for (const op of ops) visit(op);
        return sorted;
    }
}
