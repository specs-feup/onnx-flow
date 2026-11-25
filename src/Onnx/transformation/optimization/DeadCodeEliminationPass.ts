import type OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import OperationNode from "../../OperationNode.js";
import RegionArgumentNode from "../../RegionArgumentNode.js";

export class DeadCodeEliminationPass implements GraphPass {
    public readonly name = "DeadCodeElimination";

    private collectCapturedIds(graph: OnnxGraph.Class, out: Set<string>): void {
        for (const op of graph.getOperationNodes()) {
            for (const region of op.regions) {
                for (const arg of region.getNodes().filterIs(RegionArgumentNode)) {
                    out.add(arg.id);
                    out.add(arg.originalName);
                }
                this.collectCapturedIds(region, out);
            }
        }
    }

    run(graph: OnnxGraph.Class): boolean {
        let changed = false;

        // First run DCE inside nested regions so dead captures inside regions are pruned before outer DCE
        for (const op of graph.getOperationNodes()) {
            for (const region of op.regions) {
                if (this.run(region)) {
                    changed = true;
                }
            }
        }

        const capturedIds = new Set<string>();
        this.collectCapturedIds(graph, capturedIds);

        let removing = true;

        // Iterate until we reach a fixed point (removing a node might orphan its parents)
        while (removing) {
            removing = false;
            const nodes = graph.nodes.toArray();

            for (const node of nodes) {
                if (!graph.hasNode(node.id)) continue;

                // Never remove nodes implicitly captured by nested Loop / If / Scan regions
                if (capturedIds.has(node.id)) continue;

                // Never remove the main graph's explicit inputs or outputs
                if (node.is(TensorNode)) {
                    const tn = node.as(TensorNode);
                    if (tn.type === "input" || tn.type === "output") continue;
                    // Only remove an unconsumed intermediate tensor if it has no live producer op
                    // (e.g. orphaned tensor; op outputs are removed together with their producer op
                    // so multi-output ops like Loop/Split never lose positional output tensors)
                    if (tn.outgoers.length === 0 && tn.incomers.length === 0) {
                        tn.remove();
                        removing = true;
                        changed = true;
                    }
                    continue;
                }

                if (node.is(ConstantNode)) {
                    if (node.outgoers.length === 0) {
                        node.remove();
                        removing = true;
                        changed = true;
                    }
                    continue;
                }

                if (node.is(OperationNode)) {
                    const op = node.as(OperationNode);
                    const outputs = op.getOutputs();
                    const allOutputsDead = outputs.every(
                        (out) =>
                            !capturedIds.has(out.id) &&
                            out.outgoers.length === 0 &&
                            (!out.is(TensorNode) ||
                                (out.as(TensorNode).type !== "input" &&
                                    out.as(TensorNode).type !== "output")),
                    );
                    if (allOutputsDead) {
                        op.remove();
                        for (const out of outputs) {
                            if (graph.hasNode(out.id)) out.remove();
                        }
                        removing = true;
                        changed = true;
                    }
                }
            }
        }

        return changed;
    }
}
