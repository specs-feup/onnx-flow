import OnnxGraph from "../../OnnxGraph.js";
import dequantizeLinearHandler from "./handlers/DequantizeLinear.js";
import averagePoolHandler from "./handlers/AveragePool.js";
import clipHandler from "./handlers/Clip.js";
import concatHandler from "./handlers/Concat.js";
import gemmHandler from "./handlers/Gemm.js";
import padHandler from "./handlers/Pad.js";
import sliceHandler from "./handlers/Slice.js";
import OperationNode from "../../OperationNode.js";
import softmaxHandler from "./handlers/Softmax.js";
import expandHandler from "./handlers/Expand.js";
import quantizeLinearHandler from "./handlers/QuantizeLinear.js";

export type Handler = (graph: OnnxGraph.Class, op: OperationNode.Class) => boolean;

// Registry by op type
export type HandlersRegistry = Record<string, Handler>;

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

export default function applyCanonicalization(
    graph: OnnxGraph.Class,
    options?: CanonicalizationOptions,
): OnnxGraph.Class {
    const opts: CanonicalizationOptions = {
        maxPasses: 10, // a couple passes for now
        handlers: buildDefaultRegistry(),
        ...options,
    };

    // Run to fixed point (or maxPasses) to allow chained rewrites
    for (let pass = 0; pass < (opts.maxPasses ?? 1); pass++) {
        let changed = false;

        // snapshot to avoid visiting newly inserted nodes in the same pass
        const ops = graph.getOperationNodes();

        for (const op of ops) {
            const type = op.type;
            const handler = opts.handlers[type];
            if (!handler) continue;

            const didChange = handler(graph, op);
            if (didChange) changed = true;
        }

        if (!changed) break;
    }
    return graph;
}
