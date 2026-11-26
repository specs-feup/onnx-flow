import type OnnxGraph from "../ir/OnnxGraph.js";
import { HistoryManager } from "./tracking/HistoryManager.js";
import type { GraphPass } from "./GraphPass.js";

export type { GraphPass } from "./GraphPass.js";
export { BaseGraphPass } from "./GraphPass.js";

/**
 * Orchestrates an ordered pipeline of `GraphPass` instances over an `OnnxGraph`,
 * supporting optional fixed-point iteration up to `maxIterations`.
 */
export class PassManager {
    private passes: GraphPass[] = [];

    /** Appends `pass` to the pipeline and returns `this` for fluent chaining. */
    public addPass(pass: GraphPass): this {
        this.passes.push(pass);
        return this;
    }

    /** Returns the registered passes in execution order. */
    public getPasses(): readonly GraphPass[] {
        return this.passes;
    }

    /** Executes all registered passes on `graph` until convergence or `maxIterations`. */
    public run(
        graph: OnnxGraph.Class,
        historyManager: HistoryManager = new HistoryManager(graph),
        maxIterations: number = 1,
        verbose: boolean = false,
    ): HistoryManager {
        for (let i = 0; i < maxIterations; i++) {
            let changed = false;
            for (const pass of this.passes) {
                if (verbose) console.log(`[PassManager] Running ${pass.name}...`);
                if (pass.run(graph, historyManager)) {
                    changed = true;
                }
            }
            if (!changed) {
                if (verbose && maxIterations > 1) {
                    console.log(`[PassManager] Reached fixed point after ${i + 1} iterations.`);
                }
                break;
            }
        }
        return historyManager;
    }
}
