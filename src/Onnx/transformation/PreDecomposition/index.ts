import OnnxGraph from "../../OnnxGraph.js";
import clipHandler from "./handlers/Clip.js";
import convHandler from "./handlers/Conv.js";
import padHandler from "./handlers/Pad.js";
import sliceHandler from "./handlers/Slice.js";
import { HandlersRegistry, PreDecomposeOptions } from "./types.js";


function buildDefaultRegistry(): HandlersRegistry {
  return {
    // Register handlers here. Keys are op types.
    Conv: convHandler,
    Slice: sliceHandler,
    Pad: padHandler,
    Clip: clipHandler,
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
