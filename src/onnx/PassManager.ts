import type OnnxGraph from "./OnnxGraph.js";
import { HistoryManager } from "./transformation/tracking/HistoryManager.js";

export interface GraphPass {
    readonly name: string;
    /**
     * Executes the pass on the given graph.
     * @returns {boolean} True if the graph was mutated, false otherwise.
     */
    run(graph: OnnxGraph.Class, historyManager: HistoryManager): boolean;
}

export class PassManager {
    private passes: GraphPass[] = [];

    public addPass(pass: GraphPass): this {
        this.passes.push(pass);
        return this; // Fluent chaining: pm.addPass(...).addPass(...)
    }

    public getPasses(): readonly GraphPass[] {
        return this.passes;
    }

    public run(
        graph: OnnxGraph.Class,
        historyManager: HistoryManager = new HistoryManager(graph),
        maxIterations: number = 1,
        verbose: boolean = true,
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
