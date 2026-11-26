import type OnnxGraph from "../../ir/OnnxGraph.js";
import { BaseGraphPass } from "../GraphPass.js";
import TensorNode from "../../ir/TensorNode.js";
import ConstantNode from "../../ir/ConstantNode.js";
import OperationNode from "../../ir/OperationNode.js";
import RegionArgumentNode from "../../ir/RegionArgumentNode.js";

/** Removes unreachable intermediate tensors, orphan constants, and dead operations recursively. */
export class DeadCodeEliminationPass extends BaseGraphPass {
    public override readonly name = "DeadCodeElimination";

    private collectCapturedIds(graph: OnnxGraph.Class, out: Set<string>): void {
        for (const op of graph.getOperationNodes()) {
            for (const region of op.regions) {
                for (const arg of region.getNodes().filterIs(RegionArgumentNode)) {
                    out.add(arg.id);
                    out.add(arg.originalName);
                }
                for (const innerOp of region.getOperationNodes()) {
                    for (const inp of innerOp.getInputs() ?? []) {
                        if (!region.hasNode(inp.id)) {
                            out.add(inp.id);
                        }
                    }
                }
                this.collectCapturedIds(region, out);
            }
        }
    }

    public override run(graph: OnnxGraph.Class): boolean {
        // First run DCE inside nested regions so dead captures inside regions are pruned before outer DCE
        let changed = this.runOnRegions(graph);

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
