import type OnnxGraph from "../../OnnxGraph.js";
import dequantizeLinearHandler from "./handlers/DequantizeLinear.js";
import averagePoolHandler from "./handlers/AveragePool.js";
import clipHandler from "./handlers/Clip.js";
import concatHandler from "./handlers/Concat.js";
import gemmHandler from "./handlers/Gemm.js";
import padHandler from "./handlers/Pad.js";
import sliceHandler from "./handlers/Slice.js";
import type OperationNode from "../../OperationNode.js";
import softmaxHandler from "./handlers/Softmax.js";
import expandHandler from "./handlers/Expand.js";
import quantizeLinearHandler from "./handlers/QuantizeLinear.js";
import type { GraphPass } from "../../PassManager.js";

export type Handler = (graph: OnnxGraph.Class, op: OperationNode.Class) => boolean;

// Registry by op type
export type HandlersRegistry = Record<string, Handler | undefined>;

export interface CanonicalizationOptions {
    maxPasses?: number;
    handlers?: HandlersRegistry;
}

function buildDefaultRegistry(): HandlersRegistry {
    return {
        // Register handlers here. Keys are op types.
        Slice: sliceHandler,
        Pad: padHandler,
        Clip: clipHandler,
        Gemm: gemmHandler,
        Concat: concatHandler,
        DequantizeLinear: dequantizeLinearHandler,
        QuantizeLinear: quantizeLinearHandler,
        AveragePool: averagePoolHandler,
        Softmax: softmaxHandler,
        Expand: expandHandler,
    };
}

export class CanonicalizationPass implements GraphPass {
    public readonly name = "CanonicalizationPass";
    private handlers: HandlersRegistry;
    private maxInternalPasses: number;

    constructor(options?: CanonicalizationOptions) {
        this.handlers = options?.handlers ?? buildDefaultRegistry();
        this.maxInternalPasses = options?.maxPasses ?? 10;
    }

    run(graph: OnnxGraph.Class): boolean {
        let anyChange = false;

        // Run to fixed point (or maxPasses) to allow chained rewrites
        for (let pass = 0; pass < this.maxInternalPasses; pass++) {
            let changed = false;

            // Snapshot to avoid visiting newly inserted nodes in the same pass
            const ops = graph.getOperationNodes().toArray();

            for (const op of ops) {
                if (!graph.hasNode(op.id)) continue;

                const handler = this.handlers[op.type];
                if (handler === undefined) continue;

                const didChange = handler(graph, op);
                if (didChange) {
                    changed = true;
                    anyChange = true;
                }
            }

            if (!changed) break;
        }

        return anyChange;
    }
}
