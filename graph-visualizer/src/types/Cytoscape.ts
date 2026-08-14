/**
 * @file Cytoscape.ts
 * @description Type definitions representing Cytoscape graph structures used across the visualizer.
 */

/**
 * Represents the normalized Cytoscape graph data payload containing nodes and edges.
 */
export type CytoscapeData = {
    /**
     * Graph elements container.
     */
    elements: {
        /**
         * List of graph node objects conforming to Cytoscape element definitions.
         */
        nodes: Array<Record<string, unknown>>;
        /**
         * List of graph edge objects connecting source and target nodes.
         */
        edges: Array<Record<string, unknown>>;
    };
};