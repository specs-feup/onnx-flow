import fs from 'fs';

// Helper function to convert a Cytoscape graph to DOT format
function cytoscapeToDot(cyGraph) {
    const graphName = (cyGraph.renderer.name & cyGraph.renderer.name != "null") ? cyGraph.renderer.name : "G";
    const nodes = cyGraph.elements.nodes;
    let edges = cyGraph.elements.edges;

    // Sort edges by the 'order' attribute
    edges = edges.sort((a, b) => (a.data.order || 0) - (b.data.order || 0));

    // Start the DOT format string
    let dotString = `digraph ${graphName} {\n`;

    // Add nodes
    nodes.forEach(node => {
        const id = node.data.id;
        const opcode = node.classes ? (node.classes == "operation" ? node.data.opType : node.classes) : "";
        const opcodeString = opcode=="" ? "" : ` [opcode="${opcode}"]`;
        const constValue = opcode=="constant" ? (node.data.value ? node.data.value : (node.data.label ? node.data.label : "")) : "";
        const valueString = constValue=="" ? "" : `[value="${constValue}"]`;

        dotString += `"${id}"${opcodeString}${valueString};\n`;
    });

    // Add edges
    edges.forEach(edge => {
        const source = edge.data.source;
        const target = edge.data.target;
        const dims = edge.data.label ? edge.data.label : ((edge.data.dims && Array.isArray(edge.data.dims)) ? edge.data.dims.map(dim => dim.dimValue).join(",") : "");
        const dimsString = dims=="" ? "" : ` [dims="${dims}"]`;
        dotString += `"${source}" -> "${target}${dimsString};\n`;
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

main();
