import type OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";
import { LoopFusionMatcher } from "./LoopFusionMatcher.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";

export class LoopLoweringPass implements GraphPass {
    public readonly name = "LoopLoweringV2";

    constructor(
        private options: { coalesce: boolean; fuse: boolean } = { coalesce: true, fuse: true },
    ) {}

    public run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean {
        let changed = false;

        // 1. Initialize the matcher
        const matcher = new LoopFusionMatcher(this.options);

        // 2. Find all loop fusion chains
        const opportunities = matcher.findOpportunities(graph);

        // 3. Greedily apply them to maintain current compiler functionality
        for (const opp of opportunities) {
            // Because opp.targetNodeId is a comma-separated list of the chain,
            // we check if the root node (the last one) still exists.
            const nodeIds = opp.targetNodeId.split(",");
            const rootNodeId = nodeIds[nodeIds.length - 1];

            if (graph.hasNode(rootNodeId)) {
                //const builder = new GraphBuilder(graph, `lowering_${rootNodeId}`);
                const builder = new TrackedGraphBuilder(
                    graph,
                    opp.id,
                    opp.description,
                    "loop_lowering",
                );

                opp.apply(builder);
                changed = true;

                const patch = builder.commitPatch();
                historyManager.pushPatch(patch);
            }
        }

        return changed;
    }
}
