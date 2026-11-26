import type OnnxGraph from "../ir/OnnxGraph.js";
import type { HistoryManager } from "./tracking/HistoryManager.js";

/**
 * Contract for a single compiler pass over an `OnnxGraph`.
 * Passes return `true` when they mutate the graph so `PassManager` can
 * detect convergence during fixed-point iteration.
 */
export interface GraphPass {
    readonly name: string;

    /**
     * Executes the pass on `graph`, optionally recording reversible patches into `historyManager`.
     * @returns `true` if the graph was mutated, `false` otherwise.
     */
    run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean;
}

/**
 * Optional abstract base class for `GraphPass` implementations that need
 * recursive traversal into nested control-flow regions (`Loop`, `If`, `Scan`).
 */
export abstract class BaseGraphPass implements GraphPass {
    public abstract readonly name: string;

    public abstract run(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean;

    /** Runs this pass recursively over every child region of operations in `graph`. */
    protected runOnRegions(graph: OnnxGraph.Class, historyManager?: HistoryManager): boolean {
        let changed = false;
        for (const op of graph.getOperationNodes()) {
            for (const region of op.regions) {
                if (this.run(region, historyManager)) {
                    changed = true;
                }
            }
        }
        return changed;
    }
}
