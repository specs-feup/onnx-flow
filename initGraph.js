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

function addNodes(data, mapNodeAndOutput) {
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
        node.output.forEach(output => {
            mapNodeAndOutput.push({nodeId: index.toString(), output: output})
        })
    })
    return oppNodes
}

function addEdges(nodes, inputs, outputs, mapNodeAndOutput, edges) {
    nodes.forEach(node => {
        node.data.inputs.forEach(input => {
            const inputFound = inputs.find(inp => inp.data.id === input)
            if (inputFound) {
                edges.push({
                    data: {
                        source: input,
                        target: node.data.id,
                        dims: inputFound.data.dimensions, 
                        elemType: inputFound.data.elemType
                    }
                })
            }
            else {
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
            }
        })
        node.data.outputs.forEach(output => {
            const outputFound = outputs.find(out => out.data.id === output)
            if (outputFound) {
                edges.push({
                    data: {
                        source: node.data.id,
                        target: output,
                        dims: outputFound.data.dimensions, 
                        elemType: outputFound.data.elemType
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

function bfs(cy,origin) {
    let bfs = cy.nodes('.operation').bfs({
        roots: origin,
        visit: function(v){
            let incomingEdges = v.incomers().filter('edge');
            if (incomingEdges.every(edge => edge.data('dims') !== 'None')) {
                let firstEdgeDims = incomingEdges[0].data('dims');
                let firstEdgeElemType = incomingEdges[0].data('elemType');
                v.outgoers().filter('edge').forEach(edge => {
                    edge.data('dims', firstEdgeDims);
                    edge.data('elemType', firstEdgeElemType);
                });
            }
        },
        directed: true
      });
}



export function createGraph(data) {

    let inputs = addInputNodes(data[0])
    let outputs = addOutputNodes(data[0])
    let edges = []
    let mapNodeAndOutput = []
    let nodes = addNodes(data[0], mapNodeAndOutput)
    addEdges(nodes, inputs, outputs, mapNodeAndOutput, edges)
    let cy =  initializeCytoscapeGraph(nodes, inputs, outputs, edges, data[1])
    let origin = cy.nodes('.operation').filter(node => node.incomers().filter('edge').every(edge => edge.data('dims') !== 'None'));

    bfs(cy, origin)
    
    cy.edges().forEach(edge => {
        console.log(edge.data())
    })
    
    return cy
} 


//separar addNodes (não misturar coisas diferentes) e talvez usar selectors nos métodos anteriores
//começar dos inputs na bfs 
//ou fazer recursivamente