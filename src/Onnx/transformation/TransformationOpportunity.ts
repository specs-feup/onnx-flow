import type { GraphBuilder } from "../GraphBuilder.js";

export class TransformationOpportunity {
    public readonly id: string;

    constructor(
        public readonly recipeName: string,
        public readonly targetNodeId: string,
        public readonly description: string,
        private readonly applyCallback: (builder: GraphBuilder) => void,
    ) {
        // Auto-generate a deterministic ID so the recipes don't have to
        this.id = `${recipeName}_${targetNodeId}`;
    }

    /**
     * Executes the transformation.
     */
    public apply(builder: GraphBuilder): void {
        this.applyCallback(builder);
    }
}
