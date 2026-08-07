export type CytoscapeData = {
    elements: {
        nodes: Array<Record<string, unknown>>;
        edges: Array<Record<string, unknown>>;
    };
};