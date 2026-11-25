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
                    const type = node.as(TensorNode).type;
                    if (type === "input" || type === "output") continue;
                }

                // If a node has ZERO consumers (outgoers)
                if (node.outgoers.length === 0) {
                    if (node.is(TensorNode) || node.is(ConstantNode) || node.is(OperationNode)) {
                        node.remove();
                        removing = true;
                        changed = true;
                    }
                }
            }
        }

        return changed;
    }
}
