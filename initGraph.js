function addInputNodes(data) {
    let inputs = []
    data.graph.input.forEach(input => {
      inputs.push({
        data: {
            id: input.name,
            label: input.name,
            elemType: input.type.tensorType.elemType,
            dimensions: input.type.tensorType.shape.dim
        },
        classes: 'input'
      })
    })
    return inputs
}

function addOutputNodes(data) {
    let outputs = []
    data.graph.output.forEach(output => {
      outputs.push({
        data: {
            id: output.name,
            label: output.name,
            elemType: output.type.tensorType.elemType,
            dimensions: output.type.tensorType.shape.dim
        },
        classes: 'output'
      })
    })
    return outputs
}


function addNodes(data, inputs, outputs, mapNodeAndOutput, edges) {
    let oppNodes = []
    data.graph.node.forEach((node, index) => {
        oppNodes.push({
            data: {
                id: index.toString(),
                label: node.opType,
                inputs: node.input,
                outputs: node.output
            },
            classes: 'operation'
        })
        node.input.forEach(input => {
            const inputFound = inputs.find(inp => inp.data.id === input)
            if (inputFound) {
                edges.push({
                    data: {
                        source: input,
                        target: index.toString(),
                        dims: inputFound.dimensions, 
                        elemType: inputFound.elemType
                    }
                })
            }
        })
        node.output.forEach(output => {
            const outputFound = outputs.find(out => out.data.id === output)
            if (outputFound) {
                edges.push({
                    data: {
                        source: index.toString(),
                        target: output,
                        dims: outputFound.dimensions, 
                        elemType: outputFound.elemType
                    }
                })
            }
            else mapNodeAndOutput.push({nodeId: index.toString(), output: output})
        })
    })
    return oppNodes
}


function addEdges(nodes, mapNodeAndOutput, edges) {
    nodes.forEach(node =>{
        node.data.inputs.forEach(input => {
            const nodeWithCorrespondingOutput = mapNodeAndOutput.find(elem => elem.output === input)
            if (nodeWithCorrespondingOutput) {
                edges.push({
                    data: {
                        source: nodeWithCorrespondingOutput.nodeId,
                        target: node.data.id,
                        dims: 'None', 
                        elemType: 'None'
                    }
                })
            }
        })
    })
}


function initializeCytoscapeGraph (nodes, inputs, outputs, edges, sty) {
    const cy = cytoscape({
        container: document.getElementById('cy'),
        elements: {
            nodes: nodes.concat(inputs, outputs),
            edges:edges
        },
        style: sty,
        layout: {
        name: 'grid',
        rows: 1,
        position: function(node) {
            // Adjust the position of nodes based on their type
            if (node.hasClass('input')) {
                return { row: 0, col: node.data('index') };
            } else if (node.hasClass('output')) {
                return { row: 2, col: node.data('index') };
            } else {
                return { row: 1, col: node.data('index') };
            }
        },
        avoidOverlap: true,
        }
        
    })
    return cy;
}


export function createGraph(data) {

    let inputs = addInputNodes(data[0])
    let outputs = addOutputNodes(data[0])
    let edges = []
    let mapNodeAndOutput = []
    let nodes = addNodes(data[0], inputs, outputs, mapNodeAndOutput, edges)
    addEdges(nodes, mapNodeAndOutput, edges)
    return initializeCytoscapeGraph(nodes, inputs, outputs, edges, data[1])
} 