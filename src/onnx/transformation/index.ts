import type Graph from "@specs-feup/flow/graph/Graph";
import type OnnxGraph from "../OnnxGraph.js";
import { initializeSchemaRegistry } from "../schema/index.js";
import { PassManager } from "../PassManager.js";
import { CanonicalizationPass } from "./canonicalization/CanonicalizationPass.js";
import { InferShapesPass } from "./InferShapesPass.js";
import { LoopLoweringPass } from "./loop-lowering/LoopLoweringPass.js";
import { LoopTilingPass } from "./loop-lowering/LoopTilingPass.js";
import { PostLoopLoweringPass } from "./PostLoopLoweringPass.js";
import transformForCgra from "./cgra-decomposition-example/TransformForCgra.js";
import { HistoryManager } from "./tracking/HistoryManager.js";
import type { DecompositionOptions, IndexingMode } from "../../DecompositionOptions.js";
import { defaultDecompositionOptions } from "../../DecompositionOptions.js";
import { ShapeOptimizationPass } from "./optimization/ShapeOptimizationPass.js";
import { DeadCodeEliminationPass } from "./optimization/DeadCodeEliminationPass.js";
import { Schedule, resolveScheduleConfig, type ScheduleConfig } from "./Schedule.js";

/**
 * Top-level compilation pipeline driving an `OnnxGraph` through:
 * 1. Optional schedule `cacheRead` staging and **Canonicalization** (macro-op decomposition)
 * 2. **Loop Lowering & Fusion**, optional recursive post-loop lowering, and **Loop Tiling / Schedule transforms**
 * 3. **Optimization & Cleanup** (`DeadCodeEliminationPass`, `ShapeOptimizationPass`, and final `InferShapesPass`)
 */
export default class OnnxGraphTransformer implements Graph.Transformation<
    OnnxGraph.Class,
    OnnxGraph.Class
> {
    private canonicalize: boolean;
    private fuse: boolean;
    private recurse: boolean;
    private coalesce: boolean;
    private loopLowering: boolean;
    private optimize: boolean;
    private decomposeForCgra: boolean;
    private indexingMode: IndexingMode;
    private tileSize?: number | undefined;
    private targetOpset?: number | undefined;
    private irVersion?: number | undefined;
    private useScanOutputs: boolean;
    private fuseMultiOutput: boolean;
    private scheduleConfig?: ScheduleConfig | undefined;

    constructor(options: Partial<DecompositionOptions> = {}) {
        this.canonicalize = options.canonicalize ?? defaultDecompositionOptions.canonicalize;
        this.fuse = options.fuse ?? defaultDecompositionOptions.fuse;
        this.recurse = options.recurse ?? defaultDecompositionOptions.recurse;
        this.coalesce = options.coalesce ?? defaultDecompositionOptions.coalesce;
        this.loopLowering = options.loopLowering ?? defaultDecompositionOptions.loopLowering;
        this.optimize = options.optimize ?? defaultDecompositionOptions.optimize;
        this.decomposeForCgra =
            options.decomposeForCgra ?? defaultDecompositionOptions.decomposeForCgra;
        this.indexingMode = options.indexingMode ?? defaultDecompositionOptions.indexingMode;
        this.tileSize = options.tileSize ?? defaultDecompositionOptions.tileSize;
        this.targetOpset = options.targetOpset;
        this.irVersion = options.irVersion;
        this.useScanOutputs =
            options.useScanOutputs ?? defaultDecompositionOptions.useScanOutputs ?? false;
        this.fuseMultiOutput =
            options.fuseMultiOutput ?? defaultDecompositionOptions.fuseMultiOutput ?? false;
        this.scheduleConfig = resolveScheduleConfig(options.schedule);
    }

    /** Executes the configured transformation pass pipeline on `graph` in place. */
    public apply(graph: OnnxGraph.Class, printSummary: boolean = true): OnnxGraph.Class {
        initializeSchemaRegistry();
        if (this.targetOpset !== undefined) {
            graph.setTargetOpset(this.targetOpset, true);
        }
        if (this.irVersion !== undefined) {
            graph.setIrVersion(this.irVersion, true);
        }

        const pm = new PassManager();
        const historyManager = new HistoryManager(graph);

        // Phase 0: Standalone CGRA decomposition example (mutually exclusive with standard pipeline)
        if (this.decomposeForCgra) {
            pm.addPass({
                name: "TransformForCgraExample",
                run: (g) => {
                    transformForCgra(g);
                    return false;
                },
            });
            pm.run(graph, historyManager);
            return graph;
        }

        const scheduleCfg = this.scheduleConfig;
        if (scheduleCfg?.cacheReads !== undefined && scheduleCfg.cacheReads.length > 0) {
            pm.addPass({
                name: "ScheduleCacheRead",
                run: (g, hm) => Schedule.applyCacheReads(g, scheduleCfg, hm),
            });
        }

        // Phase 1: Normalization (Canonicalization)
        if (this.canonicalize) {
            pm.addPass(
                new CanonicalizationPass(undefined, {
                    loopLowering: this.loopLowering,
                    recurse: this.recurse,
                    ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                }),
            );
            if (this.optimize) {
                pm.addPass(new DeadCodeEliminationPass());
            }
            pm.addPass(new InferShapesPass());
        }

        // Phase 2: Loop Lowering, Post-Loop Lowering & Tiling
        if (this.loopLowering) {
            pm.addPass(
                new LoopLoweringPass({
                    coalesce: this.coalesce,
                    fuse: this.fuse,
                    indexingMode: this.indexingMode,
                    useScanOutputs: this.useScanOutputs,
                    fuseMultiOutput: this.fuseMultiOutput,
                    ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                }),
            );

            if (this.canonicalize && this.recurse) {
                pm.addPass(
                    new PostLoopLoweringPass({
                        loopLowering: this.loopLowering,
                        coalesce: this.coalesce,
                        fuse: this.fuse,
                        indexingMode: this.indexingMode,
                        useScanOutputs: this.useScanOutputs,
                        fuseMultiOutput: this.fuseMultiOutput,
                        ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                    }),
                );
                // Infer 0D shapes inside newly emitted Loop bodies, then recursively canonicalize
                // any remaining non-native scalar compute ops (Exp, Div, Sqrt, Log, etc.).
                pm.addPass(new InferShapesPass());
                pm.addPass(
                    new CanonicalizationPass(undefined, {
                        loopLowering: false,
                        recurse: true,
                        regionOnly: true,
                        scalar0DOnly: true,
                        ...(scheduleCfg !== undefined ? { schedule: scheduleCfg } : {}),
                    }),
                );
            }

            if (this.tileSize !== undefined && this.tileSize > 1) {
                pm.addPass(new LoopTilingPass({ tileSize: this.tileSize }));
            }
        }

        if (scheduleCfg !== undefined) {
            pm.addPass({
                name: "ScheduleLoopTransforms",
                run: (g, hm) => Schedule.applyLoopTransforms(g, scheduleCfg, hm),
            });
        }

        // Phase 3: Final Optimization & Shape Propagation
        if (this.optimize) {
            pm.addPass(new DeadCodeEliminationPass());
            pm.addPass(new ShapeOptimizationPass());
        }
        pm.addPass(new InferShapesPass());

        pm.run(graph, historyManager);
        if (printSummary) {
            historyManager.printSummary();
        }

        return graph;
    }
}
