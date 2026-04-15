import type OnnxGraph from "./OnnxGraph.js";

export interface GraphPass {
    readonly name: string;
    /**
     * Executes the pass on the given graph.
     * @returns {boolean} True if the graph was mutated, false otherwise.
     */
    run(graph: OnnxGraph.Class): boolean;
}

export class PassManager {
    private passes: GraphPass[] = [];

    public addPass(pass: GraphPass): void {
        this.passes.push(pass);
    }

    public run(graph: OnnxGraph.Class, maxIterations: number = 1): void {
        for (let i = 0; i < maxIterations; i++) {
            let changed = false;

            for (const pass of this.passes) {
                console.log(`[PassManager] Running ${pass.name}...`);
                if (pass.run(graph)) {
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
