import fs from 'fs';

// Helper function to convert a Cytoscape graph to DOT format
export function cytoscapeToDot(cyGraph) {
    const graphName = (cyGraph.renderer.name & cyGraph.renderer.name != "null") ? cyGraph.renderer.name : "G";
    const nodes = cyGraph.elements.nodes;
    let edges = cyGraph.elements.edges;

    // Sort edges by the 'order' attribute
    edges = edges.sort((a, b) => (a.data.order || 0) - (b.data.order || 0));

    // Start the DOT format string
    let dotString = `digraph ${graphName} {\n`;

    // Group nodes by parent/subgraph
    const subgraphs = new Map();
    const independentNodes = [];

    nodes.forEach(node => {
        const id = node.data.id;
        const opcode = node.classes ? (node.classes == "operation" ? node.data.opType : node.classes) : "";
        const opcodeString = opcode=="" ? "" : ` [opcode="${opcode}"]`;
        const constValue = opcode=="constant" ? (node.data.value ? node.data.value : (node.data.label ? node.data.label : "")) : "";
        const valueString = constValue=="" ? "" : `[value="${constValue}"]`;
        const labelString = (opcode=="input" || opcode=="output" || opcode=="constant") ? `[label="${opcode +" "+ id}"]` : "";

        const parent = node.data.parent;

        if (parent) {
            if (!subgraphs.has(parent)) {
                subgraphs.set(parent, []);
            }
            subgraphs.get(parent).push({ id, opcodeString, valueString, labelString });
        } else {
            independentNodes.push({ id, opcodeString, valueString, labelString });
        }
    });

    // Add independent nodes (nodes without parents)
    independentNodes.forEach(({ id, opcodeString, valueString, labelString }) => {
        dotString += `   "${id}"${opcodeString}${valueString}${labelString};\n`;
    });

    // Add subgraphs
    subgraphs.forEach((nodes, parentId) => {
        dotString += `   subgraph "cluster_${parentId}" {\n`;
        dotString += `      label="${parentId}";\n`;
        nodes.forEach(({ id, opcodeString, valueString, labelString }) => {
            dotString += `      "${id}"${opcodeString}${valueString}${labelString};\n`;
        });
        dotString += `   }\n`;
    });

    // Add edges
    edges.forEach(edge => {
        const source = edge.data.source;
        const target = edge.data.target;
        const dims = edge.data.label ? edge.data.label : ((edge.data.dims && Array.isArray(edge.data.dims)) ? edge.data.dims.map(dim => dim.dimValue).join(",") : "");
        const dimsString = dims=="" ? "" : ` [dims="${dims}"]`;
        const labelString = dims=="" ? "" : ` [label="${dims}"]`;

        dotString += `   "${source}" -> "${target}"${labelString}${dimsString};\n`;
    });

    // Close the DOT format string
    dotString += "}\n";

    return dotString;
}

// Main function to read a Cytoscape JSON and convert to DOT
async function main() {
    const inputPath = process.argv[2];
    const outputPath = process.argv[3];

    if (!inputPath || !outputPath) {
        console.error("Usage: node cytoscape2dot.js <input-cytoscape-json> <output-dot-file>");
        process.exit(1);
    }

    try {
        // Read the Cytoscape JSON file
        const cyGraph = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));

        // Convert the Cytoscape graph to DOT format
        const dotGraph = cytoscapeToDot(cyGraph);

        // Write the DOT graph to the output file
        fs.writeFileSync(outputPath, dotGraph);
        console.log(`DOT graph successfully written to ${outputPath}`);
    } catch (error) {
        console.error("Error processing the Cytoscape graph:", error);
    }
}

// Only run `main` if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    main();
}
