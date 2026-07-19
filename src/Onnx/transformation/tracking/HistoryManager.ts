import fs from "fs";
import type OnnxGraph from "../../OnnxGraph.js";
import type { MutationPatch } from "./GraphActions.js";
import inferShapes from "../../InferShapes.js";

export class HistoryManager {
    private undoStack: MutationPatch[] = [];
    private redoStack: MutationPatch[] = [];

    constructor(private graph: OnnxGraph.Class) {}

    /**
     * Pushes a new mutation patch onto the history stack.
     * This destroys any "redo" future, as we have branched into a new timeline.
     */
    public pushPatch(patch: MutationPatch): void {
        this.undoStack.push(patch);
        this.redoStack = [];
    }

    /**
     * Reverts the most recent transformation.
     * @returns The patch that was undone, or null if nothing to undo.
     */
    public undo(): MutationPatch | null {
        const patch = this.undoStack.pop();
        if (!patch) return null;

        patch.revert(this.graph);
        this.redoStack.push(patch);

        inferShapes(this.graph);
        return patch;
    }

    /**
     * Re-applies the most recently undone transformation.
     * @returns The patch that was redone, or null if nothing to redo.
     */
    public redo(): MutationPatch | null {
        const patch = this.redoStack.pop();
        if (!patch) return null;

        patch.apply(this.graph);
        this.undoStack.push(patch);

        inferShapes(this.graph);
        return patch;
    }

    /** Returns the full history of applied transformations */
    public getHistory(): MutationPatch[] {
        return this.undoStack;
    }

    /** Clears all history (useful when loading a completely new graph) */
    public clear(): void {
        this.undoStack = [];
        this.redoStack = [];
    }

    public printSummary(): void {
        console.log("\n========================================");
        console.log("       COMPILER HISTORY TIMELINE        ");
        console.log("========================================");

        if (this.undoStack.length === 0) {
            console.log(" [Empty History]");
            return;
        }

        this.undoStack.forEach((patch, index) => {
            console.log(`[Step ${index + 1}] Opportunity: ${patch.opportunityId}`);
            console.log(`         Description: ${patch.description}`);

            // Tally up what happened in this patch
            let nodesAdded = 0,
                nodesRemoved = 0,
                edgesAdded = 0,
                edgesRemoved = 0;
            for (const action of patch.actions) {
                if (action.type === "ADD_NODE") nodesAdded++;
                if (action.type === "REMOVE_NODE") nodesRemoved++;
                if (action.type === "ADD_EDGE") edgesAdded++;
                if (action.type === "REMOVE_EDGE") edgesRemoved++;
            }
            console.log(
                `         -> +${nodesAdded} Nodes | -${nodesRemoved} Nodes | +${edgesAdded} Edges | -${edgesRemoved} Edges\n`,
            );
        });
        console.log("========================================\n");
    }

    public exportHistoryToJson(filePath: string): void {
        const data = JSON.stringify(this.getHistory(), null, 2);
        fs.writeFileSync(filePath, data);
    }
}
