import type OnnxGraph from "../../ir/OnnxGraph.js";
import OperationNode from "../../ir/OperationNode.js";
import TensorNode from "../../ir/TensorNode.js";
import ConstantNode from "../../ir/ConstantNode.js";
import type { ConcreteValueNode } from "../../ir/OnnxTypes.js";
import { toStaticShape } from "../../utils/index.js";
import type { GraphPass } from "../GraphPass.js";
import { TrackedGraphBuilder } from "../tracking/TrackedGraphBuilder.js";
import type { HistoryManager } from "../tracking/HistoryManager.js";
import { matchesScheduleTarget, type ScheduleConfig } from "../Schedule.js";

/**
 * Inserts explicit scratchpad staging `Identity` reads for tensors listed in
 * `ScheduleConfig.cacheReads`, optionally scoped to matching consumer operations.
 */
export class CacheReadPass implements GraphPass {
    public readonly name = "ScheduleCacheRead";

    constructor(private readonly config: ScheduleConfig) {}

    public run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean {
        const specs = this.config.cacheReads ?? [];
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
}
