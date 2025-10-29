import OnnxGraph from "../../OnnxGraph.js";
import dequantizeLinearHandler from "./handlers/DequantizeLinear.js";
import averagePoolHandler from "./handlers/AveragePool.js";
import clipHandler from "./handlers/Clip.js";
import concatHandler from "./handlers/Concat.js";
import convHandler from "./handlers/Conv.js";
import gemmHandler from "./handlers/Gemm.js";
import padHandler from "./handlers/Pad.js";
import sliceHandler from "./handlers/Slice.js";
import OperationNode from "../../OperationNode.js";

export type Handler = (graph: OnnxGraph.Class, op: OperationNode.Class) => boolean;
// Registry by op type
export type HandlersRegistry = Record<string, Handler>;

export interface PreDecomposeOptions {
  maxPasses?: number;
  handlers?: HandlersRegistry;
}

function buildDefaultRegistry(): HandlersRegistry {
  return {
    // Register handlers here. Keys are op types.
    Conv: convHandler,
    Slice: sliceHandler,
    Pad: padHandler,
    Clip: clipHandler,
    Gemm: gemmHandler,
    Concat: concatHandler,
    DequantizeLinear: dequantizeLinearHandler,
    AveragePool: averagePoolHandler,
  };
}

export default function applyPreDecomposition(
  graph: OnnxGraph.Class,
  options?: PreDecomposeOptions
): OnnxGraph.Class {
  const opts: PreDecomposeOptions = {
    maxPasses: 10,         // a couple passes are usually enough
    handlers: buildDefaultRegistry(),
    ...options,
  };

  // Run to fixed point (or maxPasses) to allow chained rewrites
  for (let pass = 0; pass < (opts.maxPasses ?? 1); pass++) {
    let changed = false;

    // snapshot to avoid visiting newly inserted nodes in the same pass
    const ops = graph.getOperationNodes();

    
    console.log(
      `PASS ${pass + 1} | OPS:`,
      (Array.isArray(ops) ? ops : Array.from(ops ?? []))
        .map(o => o?.type ?? o?.opType ?? '(unknown)')
        .join(', ')
    );
    

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
