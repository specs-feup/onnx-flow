import type OnnxGraph from "../../ir/OnnxGraph.js";
import type OperationNode from "../../ir/OperationNode.js";
import type TensorNode from "../../ir/TensorNode.js";
import { GraphBuilder } from "../../ir/GraphBuilder.js";
import { topologicalSortOperationNodes } from "../../utils/index.js";
import { BaseGraphPass } from "../GraphPass.js";
import { asRewritableIntermediate, collectCapturedIds } from "./OptimizationUtils.js";

const NON_PURE_OR_CONTROL_OPS = new Set([
    "Loop",
    "If",
    "Scan",
    "Dropout",
    "RandomUniform",
    "RandomUniformLike",
    "RandomNormal",
    "RandomNormalLike",
    "Bernoulli",
    "Multinomial",
]);

const COMMUTATIVE_BINARY_OPS = new Set(["Add", "Mul", "Max", "Min", "Equal", "And", "Or", "Xor"]);

function canonicalizeValue(val: unknown): unknown {
    if (val === null || typeof val !== "object") return val;
    if (Array.isArray(val)) return val.map(canonicalizeValue);
    const rec = val as Record<string, unknown>;
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(rec).sort()) {
        sorted[key] = canonicalizeValue(rec[key]);
    }
    return sorted;
}

/**
 * Eliminates duplicate pure single-output operations that share the exact same opType,
 * inputs (order-insensitive for commutative binary ops), and attributes within a graph
 * or nested control-flow region.
 */
export class CommonSubexpressionEliminationPass extends BaseGraphPass {
    public override readonly name = "CommonSubexpressionElimination";

    public override run(graph: OnnxGraph.Class): boolean {
        let changed = this.runOnRegions(graph);
        const capturedIds = collectCapturedIds(graph);

        // Single topological pass deduplicates whole expression DAGs bottom-up:
        // once a producer duplicate is merged, downstream consumers automatically share input IDs.
        const seenExprs = new Map<string, TensorNode.Class>();
        const ops = topologicalSortOperationNodes(graph);

        for (const op of ops) {
            if (!graph.hasNode(op.id)) continue;
            if (NON_PURE_OR_CONTROL_OPS.has(op.type)) continue;
            if (op.regions.length > 0) continue;
            if (op.type === "Identity" && op.getMetadata("cacheRead") === true) continue;

            const outputs = op.getOutputs();
            if (outputs.length !== 1) continue;
            const outTensor = asRewritableIntermediate(outputs[0], capturedIds);
            if (outTensor === undefined || outTensor.outgoers.length === 0) continue;

            const key = this.buildExpressionKey(op);
            const existingOut = seenExprs.get(key);

            if (
                existingOut !== undefined &&
                graph.hasNode(existingOut.id) &&
                existingOut.literalType === outTensor.literalType
            ) {
                const builder = new GraphBuilder(graph, `cse_${op.id}`);
                builder.replaceAllUsesWith(outTensor, existingOut);
                op.remove();
                if (graph.hasNode(outTensor.id) && outTensor.outgoers.length === 0) {
                    outTensor.remove();
                }
                changed = true;
            } else {
                seenExprs.set(key, outTensor);
            }
        }

        return changed;
    }

    private buildExpressionKey(op: OperationNode.Class): string {
        const rawInputIds = (op.getInputs() ?? []).map((inp) => inp.id);
        const inputIds =
            COMMUTATIVE_BINARY_OPS.has(op.type) && rawInputIds.length === 2
                ? [...rawInputIds].sort()
                : rawInputIds;
        const attrJson = JSON.stringify(canonicalizeValue(op.attributes));
        return `${op.type}|${inputIds.join(",")}|${attrJson}`;
    }
}
