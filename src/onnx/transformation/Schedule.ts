import type OnnxGraph from "../ir/OnnxGraph.js";
import type OperationNode from "../ir/OperationNode.js";
import type { DecompositionOptions } from "../../DecompositionOptions.js";
import type { HistoryManager } from "./tracking/HistoryManager.js";
import OnnxGraphTransformer from "./index.js";
import { Recipes } from "./recipes.js";
import { CacheReadPass } from "./schedule/CacheReadPass.js";
import { ScheduleLoopTransformsPass } from "./schedule/ScheduleLoopTransformsPass.js";

export type StopPhase = "none" | "canonicalization" | "loop-lowering";
export type SchedulePresetName = "sfu-loops" | "flat-loops" | "cgra-tiled";

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

/** Set of all registered loop-lowering recipe names, derived from the recipe catalog. */
export const LOOP_LOWERING_RECIPE_NAMES = new Set<string>(
    Object.values(Recipes.loopLowering).map((RecipeClass) => new RecipeClass().name),
);

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
    /** Creates a pre-configured `Schedule` tuned for a target execution style. */
    public static preset(name: SchedulePresetName): Schedule {
        const s = new Schedule();
        switch (name) {
            case "sfu-loops":
                return s
                    .useRecipe("Softmax", "LowerSoftmaxLoop")
                    .useRecipe("LogSoftmax", "LowerLogSoftmaxLoop")
                    .useRecipe("LayerNormalization", "LowerLayerNormLoop")
                    .useRecipe("RMSNormalization", "LowerRMSNormLoop")
                    .useRecipe("Det", "LowerDetLoop")
                    .useRecipe("TopK", "LowerTopKScalarLoop")
                    .useRecipe("AveragePool", "LowerAveragePoolLoop");
            case "flat-loops":
                return s
                    .useRecipe("Softmax", "LowerSoftmax")
                    .useRecipe("LogSoftmax", "LowerLogSoftmax")
                    .useRecipe("LayerNormalization", "LowerLayerNormalization")
                    .useRecipe("RMSNormalization", "LowerRMSNormalization")
                    .useRecipe("Det", "LowerDet")
                    .useRecipe("TopK", "LowerTopK")
                    .useRecipe("AveragePool", "LowerAveragePool");
            case "cgra-tiled":
                return Schedule.preset("sfu-loops").split("MatMul", 8).split("Conv", 8);
        }
    }

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

    /** Executes the full `OnnxGraphTransformer` pipeline with this `Schedule` attached. */
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

    /** Convenience delegate to `CacheReadPass`. */
    public static applyCacheReads(
        graph: OnnxGraph.Class,
        config: ScheduleConfig,
        historyManager?: HistoryManager,
    ): boolean {
        return new CacheReadPass(config).run(graph, historyManager);
    }

    /** Convenience delegate to `ScheduleLoopTransformsPass`. */
    public static applyLoopTransforms(
        graph: OnnxGraph.Class,
        config: ScheduleConfig,
        historyManager?: HistoryManager,
    ): boolean {
        return new ScheduleLoopTransformsPass(config).run(graph, historyManager);
    }
}
