import type OnnxGraph from "../../ir/OnnxGraph.js";
import { BaseGraphPass } from "../GraphPass.js";
import { LoopFusionMatcher } from "./LoopFusionMatcher.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { IndexingMode } from "../../../DecompositionOptions.js";
import type { ScheduleConfig } from "../schedule/Schedule.js";
import inferShapes, { propagateToRegion } from "../../schema/InferShapes.js";
import { assignRegionScopeTag, getRegionScopeTag } from "../../utils/index.js";

const DEFAULT_MAX_RECURSION_DEPTH = 8;

export interface LoopLoweringPassOptions {
    coalesce: boolean;
    fuse: boolean;
    indexingMode: IndexingMode;
    useScanOutputs?: boolean;
    fuseMultiOutput?: boolean;
    /** When true, recursively lowers tensor ops inside nested `Loop`/`If`/`Scan` regions. */
    recurse?: boolean;
    /** Internal flag marking nested-region invocations so scalar/plumbing ops are preserved. */
    inRegion?: boolean;
    /** Maximum region nesting depth for recursive loop lowering (default: 8). */
    maxDepth?: number;
    schedule?: ScheduleConfig;
}

/**
 * Lowers supported tensor operations (or fused chains) into explicit ONNX `Loop` nodes.
 * When `recurse: true`, recursively descends into control-flow regions (`Loop`, `If`, `Scan`)
 * so operations emitted inside outer loops (such as `MatMul`/`Add` inside `RNN`/`GRU`/`LSTM`
 * or native ONNX `Loop`/`If` subgraphs) are lowered into nested explicit `Loop`s.
 */
export class LoopLoweringPass extends BaseGraphPass {
    public override readonly name = "LoopLoweringV2";

    constructor(
        private options: LoopLoweringPassOptions = {
            coalesce: true,
            fuse: true,
            indexingMode: IndexingMode.Pointer,
            useScanOutputs: false,
            fuseMultiOutput: false,
        },
    ) {
        super();
    }

    public override run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean {
        return this.runAtDepth(graph, historyManager, 0);
    }

    private runAtDepth(
        graph: OnnxGraph.Class,
        historyManager: HistoryManager | undefined,
        depth: number,
    ): boolean {
        let changed = false;

        // 1. Discover and apply loop-lowering opportunities at the current graph level
        const matcher = new LoopFusionMatcher({
            ...this.options,
            inRegion: this.options.inRegion === true || depth > 0,
        });
        const opportunities = matcher.findOpportunities(graph);
        const builderPrefix = getRegionScopeTag(graph, "loop_lowering");

        for (const opp of opportunities) {
            const nodeIds = opp.targetNodeId.split(",");
            const rootNodeId = nodeIds[nodeIds.length - 1];

            if (graph.hasNode(rootNodeId)) {
                const builder = new TrackedGraphBuilder(
                    graph,
                    opp.id,
                    opp.description,
                    builderPrefix,
                );

                opp.apply(builder);
                changed = true;

                const patch = builder.commitPatch();
                if (historyManager) {
                    historyManager.pushPatch(patch);
                }
            }
        }

        // 2. Recursively lower tensor operations inside nested control-flow regions
        //    (skipping already-scalarized leaf Loops emitted by LoopChainLowerer).
        const maxDepth = this.options.maxDepth ?? DEFAULT_MAX_RECURSION_DEPTH;
        if (this.options.recurse === true && depth < maxDepth) {
            for (const op of graph.getOperationNodes().toArray()) {
                if (op.getMetadata("scalarizedLoop") === true) continue;
                for (const region of op.regions) {
                    assignRegionScopeTag(region, "ll");
                    propagateToRegion(graph, region);
                    inferShapes(region);
                    if (this.runAtDepth(region, historyManager, depth + 1)) {
                        changed = true;
                        inferShapes(region);
                    }
                }
            }
        }

        return changed;
    }
}
