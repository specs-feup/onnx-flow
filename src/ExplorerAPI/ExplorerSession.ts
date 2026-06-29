import { convertFlowGraphToOnnxJson } from "../flow2json.js";
import type OnnxGraph from "../Onnx/OnnxGraph.js";
import type { MutationPatch } from "../Onnx/transformation/tracking/GraphActions.js";
import type { HistoryManager } from "../Onnx/transformation/tracking/HistoryManager.js";
import { TrackedGraphBuilder } from "../Onnx/transformation/tracking/TrackedGraphBuilder.js";
import type { TransformationOpportunity } from "../Onnx/transformation/TransformationOpportunity.js";
import type { TransformationRegistry } from "../Onnx/transformation/TransformationRegistry.js";
import { safeWriteJson } from "../Onnx/Utils.js";

export class ExplorerSession {
    constructor(
        private graph: OnnxGraph.Class,
        private history: HistoryManager,
        private registry: TransformationRegistry,
    ) {}

    /** 1. For the graph view: Get current structure */
    public exportGraphJson(fileName: string, name: string): void {
        const onnxJson = convertFlowGraphToOnnxJson(this.graph, name);
        safeWriteJson(`${fileName}.json`, onnxJson);
    }

    /** 2. For the sidebar: Find what can be done right now */
    public getAvailableOpportunities(): TransformationOpportunity[] {
        return this.registry.findAllOpportunities(this.graph);
    }

    /** 3. For the interaction: Apply an action */
    public applyAction(opportunityId: string): void {
        const opp = this.getAvailableOpportunities().find((o) => o.id === opportunityId);
        if (!opp) throw new Error("Opportunity expired");

        const builder = new TrackedGraphBuilder(this.graph, opp.id, opp.description);
        opp.apply(builder);
        this.history.pushPatch(builder.commitPatch());
    }

    /** 4. For the history panel: Time travel */
    public undo(): MutationPatch | null {
        return this.history.undo();
    }
    public redo(): MutationPatch | null {
        return this.history.redo();
    }
    public getHistoryLog(): MutationPatch[] {
        return this.history.getHistory();
    }
}
