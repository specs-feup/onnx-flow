import type OnnxGraph from "../../ir/OnnxGraph.js";
import { topologicalSortOperationNodes } from "../../utils/index.js";
import type { GraphPass } from "../GraphPass.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { LoopTilingPass } from "../loop-lowering/LoopTilingPass.js";
import { matchesScheduleTarget, type ScheduleConfig } from "../Schedule.js";
import { LoopUnrollingPass } from "./LoopUnrollingPass.js";

/**
 * Applies post-lowering schedule directives to emitted ONNX `Loop` nodes:
 * 1. Targeted loop splitting / strip-mining via `LoopTilingPass`
 * 2. Vectorization metadata tagging (`vectorWidth`) across outer and nested loops
 * 3. Static loop unrolling via `LoopUnrollingPass`
 */
export class ScheduleLoopTransformsPass implements GraphPass {
    public readonly name = "ScheduleLoopTransforms";

    constructor(private readonly config: ScheduleConfig) {}

    public run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean {
        let changed = false;

        // 1. Targeted Split / Tiling
        for (const sp of this.config.splits ?? []) {
            if (sp.factor <= 1) continue;
            const tiler = new LoopTilingPass({ tileSize: sp.factor });
            for (const op of topologicalSortOperationNodes(graph)) {
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
        for (const vec of this.config.vectorizes ?? []) {
            if (vec.width <= 1) continue;
            const markMatching = (g: OnnxGraph.Class): void => {
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
        if (this.config.unrolls !== undefined && this.config.unrolls.length > 0) {
            const unroller = new LoopUnrollingPass(this.config.unrolls);
            if (unroller.run(graph, historyManager)) {
                changed = true;
            }
        }

        return changed;
    }
}
