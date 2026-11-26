import type OnnxGraph from "../OnnxGraph.js";
import OperationNode from "../OperationNode.js";
import TensorNode from "../TensorNode.js";
import ConstantNode from "../ConstantNode.js";
import RegionArgumentNode from "../RegionArgumentNode.js";
import type { ConcreteValueNode, KnownShape, ValueNode } from "../OnnxTypes.js";
import type { DecompositionOptions } from "../../DecompositionOptions.js";
import { GraphBuilder } from "../GraphBuilder.js";
import type { TrackedGraphBuilder as TrackedGraphBuilderType } from "./tracking/TrackedGraphBuilder.js";
import { TrackedGraphBuilder } from "./tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "./tracking/HistoryManager.js";
import OnnxGraphTransformer from "./index.js";
import { LoopTilingPass } from "./loop-lowering/LoopTilingPass.js";
import {
    int64Vec,
    readScalarFromTensorNode,
    scalarInt64,
    toStaticShape,
    topologicalSortOperationNodes,
} from "../Utils.js";

export type StopPhase = "none" | "canonicalization" | "loop-lowering";

export interface ScheduleConfig {
    recipeOverrides?: Record<string, string>;
    stopAt?: Record<string, StopPhase>;
    fuseGroups?: string[][];
    computeAtList?: Array<{ producer: string; consumer: string }>;
    axisPerms?: Record<string, number[]>;
    splits?: Array<{ target: string; factor: number }>;
    unrolls?: Array<{ target: string; maxTripCount?: number }>;
    vectorizes?: Array<{ target: string; width: number }>;
    cacheReads?: Array<{ tensor: string; consumer?: string }>;
}

export const LOOP_LOWERING_RECIPE_NAMES = new Set<string>([
    "LowerElementWise",
    "LowerReduction",
    "LowerDynamicReduction",
    "LowerTranspose",
    "LowerRange",
    "LowerMatMul",
    "LowerCoalescedMatMul",
    "LowerDynamicBatchedMatMul",
    "LowerConv",
    "LowerConvND",
    "LowerConvTranspose",
    "LowerMaxPool",
    "LowerAveragePoolLoop",
    "LowerCumSum",
    "LowerReverseSequence",
    "LowerGatherElements",
    "LowerGatherND",
    "LowerScatterND",
    "LowerResize",
    "LowerResizeCubic",
    "LowerSliceLoop",
    "LowerPadLoop",
    "LowerTileLoop",
    "LowerOneHotLoop",
    "LowerConcatLoop",
    "LowerSoftmaxLoop",
    "LowerLayerNormLoop",
]);

export function resolveScheduleConfig(
    schedule?: Schedule | ScheduleConfig,
): ScheduleConfig | undefined {
    if (!schedule) return undefined;
    return schedule instanceof Schedule ? schedule.toConfig() : schedule;
}

export function getScheduledSourceTargets(op: OperationNode.Class): string[] {
    const out = new Set<string>([op.id, op.type]);
    const srcOps = op.getMetadata("sourceOps");
    if (Array.isArray(srcOps)) {
        for (const s of srcOps) out.add(String(s));
    }
    const srcTypes = op.getMetadata("sourceOpTypes");
    if (Array.isArray(srcTypes)) {
        for (const s of srcTypes) out.add(String(s));
    }
    return Array.from(out);
}

export function lookupScheduleMap<T>(
    map: Record<string, T> | undefined,
    op: OperationNode.Class,
): T | undefined {
    if (!map) return undefined;
    if (op.id in map) return map[op.id];
    const srcOps = op.getMetadata("sourceOps");
    if (Array.isArray(srcOps)) {
        for (const s of srcOps) {
            const key = String(s);
            if (key in map) return map[key];
        }
    }
    if (op.type in map) return map[op.type];
    const srcTypes = op.getMetadata("sourceOpTypes");
    if (Array.isArray(srcTypes)) {
        for (const t of srcTypes) {
            const key = String(t);
            if (key in map) return map[key];
        }
    }
    if ("*" in map) return map["*"];
    return undefined;
}

export function matchesScheduleTarget(target: string, op: OperationNode.Class): boolean {
    if (target === "*") return true;
    return getScheduledSourceTargets(op).includes(target);
}

/**
 * Fluent transformation schedule controlling per-op recipe selection, fusion groups,
 * loop axis permutations, strip-mined splits/tiling, unrolling, vectorization, and cache reads.
 */
export class Schedule {
    private recipeOverrides: Record<string, string> = {};
    private stopAtMap: Record<string, StopPhase> = {};
    private fuseGroups: string[][] = [];
    private computeAtList: Array<{ producer: string; consumer: string }> = [];
    private axisPerms: Record<string, number[]> = {};
    private splits: Array<{ target: string; factor: number }> = [];
    private unrolls: Array<{ target: string; maxTripCount?: number }> = [];
    private vectorizes: Array<{ target: string; width: number }> = [];
    private cacheReads: Array<{ tensor: string; consumer?: string }> = [];

    /** Force a specific recipe (canonicalization or loop-lowering) for a node ID or opType. */
    public useRecipe(target: string, recipeName: string): this {
        this.recipeOverrides[target] = recipeName;
        return this;
    }

    /** Stop transforming matching node(s) or opType after the given phase. */
    public stopAt(target: string, phase: StopPhase): this {
        this.stopAtMap[target] = phase;
        return this;
    }

    /** Explicitly fuse a group of operation IDs into a single (possibly multi-output) Loop. */
    public fuse(targets: string[]): this {
        if (targets.length >= 2) {
            this.fuseGroups.push([...targets]);
        }
        return this;
    }

    /** Pull a producer operation into the consumer's fused Loop cluster when compatible. */
    public computeAt(producer: string, consumer: string): this {
        this.computeAtList.push({ producer, consumer });
        return this;
    }

    /** Permute iteration-space loop axes for a lowered operation (`perm` is output axis order). */
    public reorder(target: string, perm: number[]): this {
        this.axisPerms[target] = [...perm];
        return this;
    }

    /** Strip-mine / tile Loops associated with `target` using tile `factor`. */
    public split(target: string, factor: number): this {
        this.splits.push({ target, factor });
        return this;
    }

    /** Alias for `split(target, factor)`. */
    public tile(target: string, factor: number): this {
        return this.split(target, factor);
    }

    /** Statically unroll Loops matching `target` when static tripCount <= `maxTripCount`. */
    public unroll(target: string, maxTripCount: number = 16): this {
        this.unrolls.push({ target, maxTripCount });
        return this;
    }

    /** Tag/annotate innermost matching Loops with vector lane width metadata. */
    public vectorize(target: string, width: number): this {
        this.vectorizes.push({ target, width });
        return this;
    }

    /** Insert an explicit scratchpad staging Identity read for `tensor` (optionally scoped to `consumer`). */
    public cacheRead(tensor: string, consumer?: string): this {
        this.cacheReads.push(consumer !== undefined ? { tensor, consumer } : { tensor });
        return this;
    }

    public toConfig(): ScheduleConfig {
        return {
            recipeOverrides: { ...this.recipeOverrides },
            stopAt: { ...this.stopAtMap },
            fuseGroups: this.fuseGroups.map((g) => [...g]),
            computeAtList: this.computeAtList.map((c) => ({ ...c })),
            axisPerms: Object.fromEntries(
                Object.entries(this.axisPerms).map(([k, v]) => [k, [...v]]),
            ),
            splits: this.splits.map((s) => ({ ...s })),
            unrolls: this.unrolls.map((u) => ({ ...u })),
            vectorizes: this.vectorizes.map((v) => ({ ...v })),
            cacheReads: this.cacheReads.map((c) => ({ ...c })),
        };
    }

    /** Executes the full OnnxGraphTransformer pipeline with this Schedule attached. */
    public apply(
        graph: OnnxGraph.Class,
        baseOptions: Partial<DecompositionOptions> = {},
        printSummary: boolean = false,
    ): OnnxGraph.Class {
        return new OnnxGraphTransformer({
            ...baseOptions,
            schedule: this,
        }).apply(graph, printSummary);
    }

    public static applyCacheReads(
        graph: OnnxGraph.Class,
        config: ScheduleConfig,
        historyManager?: HistoryManager,
    ): boolean {
        const specs = config.cacheReads ?? [];
        if (specs.length === 0) return false;
        let changed = false;

        for (const spec of specs) {
            if (!graph.hasNode(spec.tensor)) continue;
            const node = graph.getNodeById(spec.tensor);
            if (!node || (!node.is(TensorNode) && !node.is(ConstantNode))) continue;
            const srcTensor = node as ConcreteValueNode;

            const consumers = srcTensor.outgoers.targets.filterIs(OperationNode).toArray();
            const matchingConsumers =
                spec.consumer !== undefined
                    ? consumers.filter((c) => matchesScheduleTarget(spec.consumer!, c))
                    : consumers;
            if (matchingConsumers.length === 0) continue;

            const tb = new TrackedGraphBuilder(
                graph,
                `cache_read_${spec.tensor}`,
                `Insert cacheRead staging Identity on '${spec.tensor}'`,
                "schedule_cache_read",
            );
            const srcShape = toStaticShape(srcTensor.shape);
            const staged = tb.createOp("Identity", [srcTensor], {}, [
                { type: srcTensor.literalType, shape: srcShape },
            ])[0];
            const idOp = staged.incomers.sources.filterIs(OperationNode).first();
            if (idOp) {
                idOp.setMetadata("cacheRead", true);
                idOp.setMetadata("targetTensor", spec.tensor);
            }

            for (const c of matchingConsumers) {
                if (idOp && c.id === idOp.id) continue;
                const oldEdge = graph.getEdge(srcTensor.id, c.id);
                if (oldEdge) tb.removeEdge(oldEdge);
                const updatedIns = (c.getInputs() ?? []).map((inp) =>
                    inp.id === srcTensor.id ? staged : inp,
                );
                c.setInputs(updatedIns);
                tb.addEdge(staged, c, staged.literalType, toStaticShape(staged.shape));
            }

            const patch = tb.commitPatch();
            if (historyManager) historyManager.pushPatch(patch);
            changed = true;
        }
        return changed;
    }

    public static applyLoopTransforms(
        graph: OnnxGraph.Class,
        config: ScheduleConfig,
        historyManager?: HistoryManager,
    ): boolean {
        let changed = false;

        // 1. Targeted Split / Tiling
        for (const sp of config.splits ?? []) {
            if (sp.factor <= 1) continue;
            const tiler = new LoopTilingPass({ tileSize: sp.factor });
            const ops = topologicalSortOperationNodes(graph);
            for (const op of ops) {
                if (!graph.hasNode(op.id) || op.type !== "Loop") continue;
                if (!matchesScheduleTarget(sp.target, op)) continue;
                if (!tiler.canTileOp(op)) continue;

                const tb = new TrackedGraphBuilder(
                    graph,
                    `ScheduleSplit_${op.id}_${sp.factor}`,
                    `Split Loop '${op.id}' with factor ${sp.factor}`,
                    "schedule_split",
                );
                tiler.tileTargetLoop(op, tb);
                const patch = tb.commitPatch();
                if (historyManager) historyManager.pushPatch(patch);
                changed = true;
            }
        }

        // 2. Vectorize annotations (outer & nested inner Loops matching target)
        for (const vec of config.vectorizes ?? []) {
            if (vec.width <= 1) continue;
            const markMatching = (g: OnnxGraph.Class) => {
                for (const op of g.getOperationNodes().toArray()) {
                    if (op.type === "Loop" && matchesScheduleTarget(vec.target, op)) {
                        op.setMetadata("vectorWidth", vec.width);
                        changed = true;
                    }
                    for (const sub of op.regions) markMatching(sub);
                }
            };
            markMatching(graph);
        }

        // 3. Static Loop Unrolling
        for (const ur of config.unrolls ?? []) {
            const maxTrips = ur.maxTripCount ?? 16;
            const unrollInGraph = (g: OnnxGraph.Class, hm?: HistoryManager): boolean => {
                let localChanged = false;
                for (const op of topologicalSortOperationNodes(g)) {
                    if (!g.hasNode(op.id)) continue;
                    for (const sub of op.regions) {
                        if (unrollInGraph(sub)) localChanged = true;
                    }
                    if (op.type !== "Loop") continue;
                    if (!matchesScheduleTarget(ur.target, op)) continue;

                    const inputs = op.getInputs() ?? [];
                    if (inputs.length < 2) continue;
                    const tripCount = readScalarFromTensorNode(inputs[0]);
                    if (
                        tripCount === undefined ||
                        !Number.isInteger(tripCount) ||
                        tripCount < 1 ||
                        tripCount > maxTrips
                    ) {
                        continue;
                    }
                    if (op.regions.length === 0) continue;
                    const loopBody = op.regions[0];
                    const hasInnerLoop = loopBody
                        .getOperationNodes()
                        .toArray()
                        .some((o) => o.type === "Loop");
                    if (hasInnerLoop) continue;

                    const tb: GraphBuilder = hm
                        ? new TrackedGraphBuilder(
                              g,
                              `ScheduleUnroll_${op.id}_${tripCount}`,
                              `Unroll Loop '${op.id}' (${tripCount} iterations)`,
                              "schedule_unroll",
                          )
                        : new GraphBuilder(g, `unroll_${op.id}`);
                    Schedule.unrollStaticLoop(op, tripCount, tb);
                    if (hm) {
                        hm.pushPatch((tb as TrackedGraphBuilderType).commitPatch());
                    }
                    localChanged = true;
                    changed = true;
                }
                return localChanged;
            };
            unrollInGraph(graph, historyManager);
        }

        return changed;
    }

    private static unrollStaticLoop(
        loopOp: OperationNode.Class,
        tripCount: number,
        builder: GraphBuilder,
    ): void {
        const graph = builder.graph;
        const loopBody = loopOp.regions[0];
        const loopInputs = loopOp.getInputs() ?? [];
        const condInit = loopInputs[1] as ConcreteValueNode;
        const carryInits = loopInputs.slice(2) as ConcreteValueNode[];
        const outerOutputs = loopOp.getOutputs();

        const bodyInputs = loopBody.getInputTensorNodes().toArray();
        const bodyIter = bodyInputs[0];
        const bodyCondIn = bodyInputs[1];
        const bodyCarryIns = bodyInputs.slice(2);
        const bodyOutputs = loopBody.getOutputTensorNodes().toArray();
        const numCarries = carryInits.length;
        const numScans = Math.max(0, bodyOutputs.length - 1 - numCarries);

        const resolveInnerOutputSource = (innerOutTensor: TensorNode.Class): ValueNode => {
            const prod = innerOutTensor.incomers.sources.filterIs(OperationNode).first();
            if (prod && prod.type === "Identity") {
                return (prod.getInputs()?.[0] ?? innerOutTensor) as ValueNode;
            }
            return innerOutTensor;
        };

        const condOutSource = resolveInnerOutputSource(bodyOutputs[0]);
        const carryOutSources = bodyOutputs.slice(1, 1 + numCarries).map(resolveInnerOutputSource);
        const scanOutSources = bodyOutputs.slice(1 + numCarries).map(resolveInnerOutputSource);

        let curCond: ValueNode = condInit;
        let curCarries: ValueNode[] = [...carryInits];
        const scanSteps: ConcreteValueNode[][] = Array.from({ length: numScans }, () => []);
        const ax0 = builder.createConstant(`unroll_ax0_${loopOp.id}`, int64Vec([0]));
        const bodyOps = topologicalSortOperationNodes(loopBody);
        const bodyConstants = loopBody.getConstantNodes().toArray();
        const captures = loopBody.getNodes().filterIs(RegionArgumentNode).toArray();

        for (let it = 0; it < tripCount; it++) {
            const valMap = new Map<string, ValueNode>();
            valMap.set(
                bodyIter.id,
                builder.createConstant(`unroll_${loopOp.id}_i${it}`, scalarInt64(it)),
            );
            valMap.set(bodyCondIn.id, curCond);
            bodyCarryIns.forEach((bIn, k) => {
                valMap.set(bIn.id, curCarries[k]);
            });

            for (const cap of captures) {
                const resolved = graph.hasNode(cap.originalName)
                    ? (graph.getNodeById(cap.originalName) as ValueNode)
                    : graph.hasNode(cap.id)
                      ? (graph.getNodeById(cap.id) as ValueNode)
                      : undefined;
                if (resolved) valMap.set(cap.id, resolved);
            }

            for (const cNode of bodyConstants) {
                const proto = {
                    ...cNode.constantValue,
                    dims: [...(cNode.constantValue.dims ?? [])],
                };
                const clonedC = builder.createConstant(`u_${loopOp.id}_i${it}_${cNode.id}`, proto);
                valMap.set(cNode.id, clonedC);
            }

            for (const innerOp of bodyOps) {
                const origOuts = innerOp.getOutputs();
                // Skip pure state passthrough Identity nodes wired only to Loop outputs
                if (
                    innerOp.type === "Identity" &&
                    origOuts.length === 1 &&
                    origOuts[0].is(TensorNode) &&
                    origOuts[0].as(TensorNode).type === "output"
                ) {
                    const src = innerOp.getInputs()?.[0];
                    if (src) valMap.set(origOuts[0].id, valMap.get(src.id) ?? src);
                    continue;
                }

                const mappedIns = (innerOp.getInputs() ?? []).map((inp) => {
                    const m = valMap.get(inp.id);
                    if (m) return m;
                    if (graph.hasNode(inp.id)) return graph.getNodeById(inp.id) as ValueNode;
                    return inp;
                });

                const expectedOuts = origOuts.map((o) => ({
                    type: o.literalType,
                    shape: Array.isArray(o.shape) ? toStaticShape(o.shape) : [],
                }));
                const clonedOuts = builder.createOp(
                    innerOp.type,
                    mappedIns,
                    { ...innerOp.attributes },
                    expectedOuts,
                );
                origOuts.forEach((origO, idx) => {
                    if (idx < clonedOuts.length) {
                        valMap.set(origO.id, clonedOuts[idx]);
                    }
                });
            }

            curCond = valMap.get(condOutSource.id) ?? condOutSource;
            curCarries = carryOutSources.map((src, k) => valMap.get(src.id) ?? curCarries[k]);
            scanOutSources.forEach((src, j) => {
                const stepVal = (valMap.get(src.id) ?? src) as ConcreteValueNode;
                const stepShape = toStaticShape(stepVal.shape);
                const [unsq] = builder.createOp("Unsqueeze", [stepVal, ax0], {}, [
                    { type: stepVal.literalType, shape: [1, ...stepShape] },
                ]);
                scanSteps[j].push(unsq);
            });
        }

        const finalOuterOutputs: ValueNode[] = [...curCarries];
        for (let j = 0; j < numScans; j++) {
            const steps = scanSteps[j];
            if (steps.length === 1) {
                finalOuterOutputs.push(steps[0]);
            } else {
                const elemTy = outerOutputs[numCarries + j]?.literalType ?? steps[0].literalType;
                const rawTargetShape =
                    outerOutputs.length > numCarries + j
                        ? outerOutputs[numCarries + j]?.shape
                        : undefined;
                const targetShape: KnownShape =
                    rawTargetShape !== undefined
                        ? toStaticShape(rawTargetShape)
                        : [tripCount, ...toStaticShape(steps[0].shape).slice(1)];
                const [stacked] = builder.createOp("Concat", steps, { axis: 0 }, [
                    { type: elemTy, shape: targetShape },
                ]);
                finalOuterOutputs.push(stacked);
            }
        }

        for (let idx = 0; idx < outerOutputs.length; idx++) {
            const oldOut = outerOutputs[idx];
            const newOut = finalOuterOutputs.length > idx ? finalOuterOutputs[idx] : undefined;
            if (newOut !== undefined && oldOut.id !== newOut.id) {
                builder.replaceAllUsesWith(oldOut, newOut);
            }
        }
        builder.removeNode(loopOp);
    }
}
