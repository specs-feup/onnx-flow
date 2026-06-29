import type OnnxGraph from "./OnnxGraph.js";
import type { HistoryManager } from "./transformation/tracking/HistoryManager.js";

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

    public addPass(pass: GraphPass): void {
        this.passes.push(pass);
    }

    public run(
        graph: OnnxGraph.Class,
        historyManager: HistoryManager,
        maxIterations: number = 1,
    ): void {
        for (let i = 0; i < maxIterations; i++) {
            let changed = false;

            for (const pass of this.passes) {
                console.log(`[PassManager] Running ${pass.name}...`);
                if (pass.run(graph, historyManager)) {
                    changed = true;
                }
            }

            if (!changed) {
                console.log(`[PassManager] Reached fixed point after ${i + 1} iterations.`);
                break;
            }
        }
    }
}
