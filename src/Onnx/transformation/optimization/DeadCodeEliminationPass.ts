import type OnnxGraph from "../../OnnxGraph.js";
import type { GraphPass } from "../../PassManager.js";
import TensorNode from "../../TensorNode.js";
import ConstantNode from "../../ConstantNode.js";
import OperationNode from "../../OperationNode.js";

export class DeadCodeEliminationPass implements GraphPass {
    public readonly name = "DeadCodeElimination";

    run(graph: OnnxGraph.Class): boolean {
        let changed = false;
        let removing = true;

        // Iterate until we reach a fixed point (removing a node might orphan its parents)
        while (removing) {
            removing = false;
            const nodes = graph.nodes.toArray();

            for (const node of nodes) {
                if (!graph.hasNode(node.id)) continue;

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

        const ops = graph.getOperationNodes();
        for (const op of ops) {
            for (const region of op.regions) {
                if (this.run(region)) {
                    changed = true;
                }
            }
        }

        return changed;
    }
}
